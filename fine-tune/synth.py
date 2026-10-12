"""Seeded synthetic request generator -> data/raw/synth.jsonl  (tops up to TARGET, safe to re-run).

    python synth.py 30      # pilot: look at the output before scaling
    python synth.py 2500    # tops the file up to 2500 records

What is sampled IN CODE (so the LLM only has to phrase things): which intents, how many (= thread count),
the concrete entities, the writing style, and an optional "twist" (dependency / negation / vague / irrelevant
detail). The LLM never sees tool names. Every record keeps its seeds in `meta`, so later you can compare the
teacher's plan against the intended structure.

If data/raw/mine.txt exists, a few of your hand-written requests are shown as realism examples.
Env: GEN_URL, GEN_KEY, WORKERS.   Extend the LISTS / INTENTS / STYLES below; they are plain data.
"""
import hashlib, os, random, re, sys, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests

from common import DATA, append_jsonl, norm_key, read_jsonl

API_URL = os.environ.get("GEN_URL", "http://localhost:8080/v1/chat/completions")
API_KEY = os.environ.get("GEN_KEY", "")
MODEL = "local-model"
WORKERS = int(os.environ.get("WORKERS", 4))
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
SEED, FEWSHOT = 0, 2
TEMP, TOP_P, MAX_TOKENS, RETRIES, TIMEOUT = 0.9, 0.95, 200, 3, 180
OUT, MINE = DATA / "raw" / "synth.jsonl", DATA / "raw" / "mine.txt"

# ---------- entities (deliberately different from the cases.py benchmark) ----------
LISTS = {
    "file": [
        "my linear algebra lecture notes", "the pitch deck for my startup", "my apartment lease agreement",
        "grandma's lasagna recipe", "last month's electricity bill", "the wedding guest list",
        "my gym workout plan", "the draft of my cover letter", "the Kubernetes cheat sheet",
        "my flight confirmation to Madrid", "the meeting notes from the budget review", "my reading list for this year",
        "the project proposal for the robotics club", "my car insurance policy", "the lecture slides on graph theory",
        "a screenshot of yesterday's error message", "my sourdough starter log", "the quarterly sales report",
        "my podcast episode script", "the contract with the freelance designer", "my hiking packing list",
        "the README of my side project", "my chemistry lab report", "the Spanish vocabulary list",
        "the customer feedback spreadsheet", "the photo of my vaccination card", "my thesis chapter on optimisation",
        "the minutes of the club meeting", "my guitar chord sheet for Wonderwall", "the quarterly OKR document",
    ],
    "script": [
        "the script that renames my photos by date", "my web scraper for apartment listings",
        "the Monte Carlo pi estimator", "my snake game", "the script that syncs my dotfiles", "my sudoku solver",
        "the markdown to PDF converter", "my tic-tac-toe minimax program", "the script that cleans my downloads folder",
        "the CSV deduplication script", "my pomodoro timer", "the k-means clustering demo", "my discord bot",
        "the image resizer",
    ],
    "program": [
        "vlc", "git", "neovim", "tmux", "docker", "nodejs", "imagemagick", "wget", "jq", "ripgrep", "tree", "gimp",
        "postgresql", "cmake", "pandoc", "yt-dlp", "sqlite3", "zsh", "obs-studio", "ncdu", "the flask python package",
        "pytest", "matplotlib for python", "pandas for python",
    ],
    "city": [
        "Madrid", "Tokyo", "Naples", "Vienna", "Reykjavik", "Lagos", "Toronto", "Buenos Aires", "Bologna", "Athens",
        "Seoul", "Cape Town", "Dublin", "Prague", "Mumbai", "Seattle", "Milan", "Istanbul", "Sydney", "Florence",
    ],
    "when": ["today", "tomorrow", "this weekend", "this evening", "next Monday", "right now"],
    "web_question": [
        "the latest stable release of Node.js", "who won the most recent Formula 1 race",
        "the current EUR to GBP exchange rate", "the opening hours of the Uffizi museum",
        "how long to boil an egg for a soft yolk", "the difference between TCP and UDP", "the population of Canada",
        "the best-rated sci-fi novels of the last decade", "when the next solar eclipse is",
        "the side effects of too much caffeine", "how to convert a PDF to text on Linux",
        "the release date of the next Python version", "the highest mountain in South America",
        "recent news about the Mars rover", "the cheapest way to get from Bologna to Florence by train",
        "how many calories are in an avocado", "what's new in the latest Linux kernel", "the rules of cricket",
    ],
    "site": [
        "the Python documentation page for the itertools module", "the official PyTorch tutorials site",
        "the Wikipedia page on the Byzantine Empire", "the Hacker News front page",
        "the arXiv listing for new machine learning papers", "the Stack Overflow question about list comprehensions",
        "the BBC News homepage", "the GitHub page of the llama.cpp project", "the MDN page about CSS grid",
        "the official website of the Louvre", "the online Rust book", "the Trenitalia ticket page",
        "the NASA image of the day page", "the Hugging Face models page",
    ],
    "chat_task": [
        "ask for a joke about programmers", "ask for a short riddle", "ask how a hash table works, in simple terms",
        "ask for three name ideas for a pet turtle", "ask for advice on staying focused while studying",
        "ask for a fun fact about octopuses", "ask for a short bedtime story about a dragon",
        "ask what the difference between 'affect' and 'effect' is", "ask for a quick mental math trick",
        "ask for tips to prepare for a job interview", "ask for a summary of the plot of Romeo and Juliet",
    ],
    "terminal_task": [
        "see how much free disk space is left", "list the running docker containers",
        "check which version of python is installed", "show the ten biggest files in the home folder",
        "see which process is using port 8080", "check how long the computer has been running",
        "count the lines in all python files in the current folder", "show the current git branch of the project",
        "check the memory usage", "create a new folder called experiments inside Documents",
        "delete everything in the tmp folder",
    ],
    "file_action": [
        "summarize it", "tell me the main points", "explain what it does", "tell me if anything looks wrong",
        "list the action items in it", "check whether it mentions a deadline", "tell me what's missing",
    ],
}

# ---------- intents: weight + what the user wants (no tool names!) ----------
INTENTS = {
    "find_open": (3, "find {file} on the user's computer and open it (a local file, not a website)"),
    "find_run": (2, "find {script} on the user's computer and run it"),
    "find_read": (3, "find {file} on the user's computer, read it, and {file_action}"),
    "install": (2, "install {program}"),
    "terminal": (2, "get this done on the computer: {terminal_task}"),
    "weather": (2, "get the weather in {city} {when}"),
    "lookup": (3, "look up on the web: {web_question}"),
    "open_url": (2, "open {site} in the web browser (a website, not a local file)"),
    "chat": (0, "{chat_task}"),  # weight 0: only used via the chat rules in make_job
}

# ---------- styles (weight, instruction) ----------
STYLES = {
    "plain": (35, "Plain and natural, the way someone types to an assistant. Normal punctuation."),
    "terse": (15, "Very terse and command-like, as few words as possible, e.g. 'open the gym plan'."),
    "typos": (12, "All lowercase, no punctuation to speak of, a couple of realistic typos."),
    "voice": (12, "As if transcribed from speech: no punctuation, filler words like 'um' or 'so', run-on sentences."),
    "polite": (8, "Polite and a bit wordy, with a courtesy phrase."),
    "rambling": (8, "Thinking out loud, a little rambling, but still clear about what is wanted."),
    "italian": (10, "Natural, plain Italian (keep program, file and place names as they are)."),
}

# ---------- twists (probability; None = no twist). Difficulty by mechanism, not adjectives ----------
TWISTS = [
    (None, 55, ""),
    ("dependency", 15, "Make one part depend on the result of an earlier part (e.g. act on what was found or looked up)."),
    ("negation", 10, "Include one explicit negative constraint: something the user does NOT want done."),
    ("vague", 10, "Leave one detail vague, e.g. refer to something by a rough description instead of its exact name."),
    ("irrelevant_detail", 10, "Add a short piece of irrelevant personal context that doesn't change what must be done."),
]
MAX_WORDS = {1: 22, 2: 40, 3: 60, 4: 80}

SYSTEM = """You write realistic messages that a person types or says to a personal assistant on their computer. \
The assistant can search the user's files, open and read files, run terminal commands, search the web, open web pages and chat.
You will get a list of things the user wants, a writing style, and sometimes a twist. Write ONE message in which the user asks for all of them.
Rules:
- First person, as the user. Do not answer the request and do not add commentary.
- Keep concrete details (names, places, programs) exactly as given.
- Do not number the items or mention tools, steps or threads: it must read like one natural message.
- Respect the word limit.
- Output only the message, on a single line."""


# ---------- sampling ----------
def sample_intents(rng, n):
    pool = {k: v[0] for k, v in INTENTS.items() if v[0]}
    names = []
    for _ in range(n):
        k = rng.choices(list(pool), list(pool.values()))[0]
        names.append(k)
        del pool[k]
    return names


def make_job(rng, mine):
    if rng.random() < 0.12:
        names = ["chat"]
    else:
        names = sample_intents(rng, rng.choices([1, 2, 3, 4], [30, 35, 25, 10])[0])
        if len(names) >= 2 and rng.random() < 0.15:
            names[-1] = "chat"
    slots = []

    def pick(m):
        v = rng.choice(LISTS[m.group(1)])
        slots.append(f"{m.group(1)}={v}")
        return v

    tasks = [re.sub(r"\{(\w+)\}", pick, INTENTS[n][1]) for n in names]
    style = rng.choices(list(STYLES), [v[0] for v in STYLES.values()])[0]
    twist = rng.choices(TWISTS, [t[1] for t in TWISTS])[0]
    if twist[0] == "dependency" and len(names) == 1:
        twist = TWISTS[0]
    words = MAX_WORDS[len(names)] // (2 if style == "terse" else 1)
    return dict(intents=names, tasks=tasks, style=style, twist=twist, max_words=max(words, 6), slots=slots,
                lang="it" if style == "italian" else "en", examples=rng.sample(mine, min(FEWSHOT, len(mine))))


def build_messages(job):
    items = "\n".join(f"{i}. {t}" for i, t in enumerate(job["tasks"], 1))
    user = f"Things the user wants (mention them in any natural order):\n{items}\n\nStyle: {STYLES[job['style']][1]}\n"
    if job["twist"][0]:
        user += f"Twist: {job['twist'][2]}\n"
    user += f"Word limit: at most {job['max_words']} words.\n"
    if job["examples"]:
        user += ("\nReal examples of how this user phrases things (match the realism, do NOT reuse their content):\n"
                 + "\n".join(f"- {e}" for e in job["examples"]) + "\n")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user + "\nWrite the message."}]


# ---------- generation + validation ----------
def call(messages):
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {API_KEY}"} if API_KEY else {})}
    payload = dict(model=MODEL, messages=messages, temperature=TEMP, top_p=TOP_P, max_tokens=MAX_TOKENS,
                   stream=False, cache_prompt=True, chat_template_kwargs={"enable_thinking": False})
    for attempt in range(RETRIES):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except (requests.RequestException, KeyError, IndexError, ValueError) as e:
            print(f"attempt {attempt + 1}/{RETRIES} failed: {e!r}", flush=True)
            if attempt + 1 < RETRIES:
                import time
                time.sleep(2 * (attempt + 1))
    return None


def tidy(text):
    text = " ".join(text.strip().splitlines()).strip()
    text = re.sub(r"^(here('s| is)?( the)?( message| request)?|message|request|prompt)\s*:\s*", "", text, flags=re.I)
    return text.strip(" \"'`“”«»")


def reject_reason(text, job):
    n = len(text.split())
    if n < 2 or n > job["max_words"] * 1.5:
        return "length"
    if re.search(r"\b(file_search|open_file|read_file|deep_think|web_search|open_url)\b", text):
        return "tool_name_leak"
    if re.match(r"(sure|okay|certainly|here|as an ai)\b", text, re.I):
        return "meta_text"
    return None


lock, seen, stats = threading.Lock(), set(), Counter()


def work(job):
    raw = call(build_messages(job))
    text = tidy(raw) if raw else ""
    reason = ("api_error" if raw is None else reject_reason(text, job)) or (
        "duplicate" if norm_key(text) in seen else None)
    with lock:
        if reason:
            stats[reason] += 1
            return
        seen.add(norm_key(text))
        stats["saved"] += 1
        append_jsonl(OUT, dict(
            id="synth_" + hashlib.sha1(text.encode()).hexdigest()[:10], text=text, source="synth",
            lang=job["lang"], license="own",
            meta=dict(style=job["style"], intents=job["intents"], n_intents=len(job["intents"]),
                      twist=job["twist"][0], slots=job["slots"], max_words=job["max_words"])))
        if stats["saved"] % 25 == 0:
            print(f"saved {stats['saved']}  rejects {dict(stats)}", flush=True)


if __name__ == "__main__":
    existing = read_jsonl(OUT)
    seen.update(norm_key(r["text"]) for r in existing)
    need = TARGET - len(existing)
    if need <= 0:
        sys.exit(f"{OUT} already has {len(existing)} >= {TARGET}")
    mine = [l.strip() for l in open(MINE, encoding="utf-8") if l.strip()] if MINE.exists() else []
    rng = random.Random(SEED + len(existing))  # re-runs sample new combinations
    jobs = [make_job(rng, mine) for _ in range(need)]
    print(f"{len(existing)} existing, generating {need} more with {WORKERS} workers", flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(work, jobs))
    print(f"\nfinished: {dict(stats)}  (re-run to top up to {TARGET})")
    for r in read_jsonl(OUT)[-5:]:
        print(f"  [{r['meta']['style']}/{r['meta']['twist']}/{'+'.join(r['meta']['intents'])}] {r['text']}")