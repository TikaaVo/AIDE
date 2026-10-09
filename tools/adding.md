# Adding New Tools to AIDE

This guide walks you through adding a new tool to AIDE from scratch.  
You don't need to touch the planner, executor, or orchestrator — the architecture is designed so that **adding a tool is entirely self-contained**.

---

## How Tools Fit Into AIDE

When a user sends a message, AIDE runs this pipeline:

```
User message
     │
     ▼
  Planner          → reads tool names + descriptions → produces an ordered list of steps
     │
     ▼
  Executor         → runs each step; if step B depends on step A, an LLM resolver
     │                translates A's output into the right input for B (using input_schema)
     ▼
  Orchestrator     → summarises all results and replies to the user
```

Everything the planner and executor need to know about a tool comes **from the tool class itself**.  
There is no config file to update, no prompt to edit — just the class and one line in `registry.py`.

---

## Step 1 — Create your tool file

Create a new file in `tools/`. Name it after what the tool does, e.g. `tools/web_search.py`.

```python
# tools/web_search.py

from tools.base import BaseTool, ToolCallResult


class WebSearchTool(BaseTool):
    name = "web_search"
    
    description = (
        "Search the web for current information, news, facts, or anything "
        "that requires up-to-date knowledge beyond what the assistant knows."
    )

    input_schema = (
        "A concise search query string, 3–8 words, suitable for a search engine. "
        "Example: 'Python 3.13 release date' or 'best noise-cancelling headphones 2025'."
    )

    ack_label = "searching the web"   # shown to user while working, e.g. "Searching the web..."

    example_phrases = [
        "search the web for X",
        "look up X online",
        "what's the latest news on X",
        "find current information about X",
        "google X for me",
    ]

    def run(self, query: str, data: dict) -> ToolCallResult:
        # query  → the input string the planner or LLM resolver produced
        # data   → session data bag (results from prior steps, context["path"], etc.)

        try:
            results = _do_actual_search(query)   # your implementation here

            if not results:
                return ToolCallResult(
                    success=False,
                    output=f"No results found for '{query}'.",
                )

            output = "\n".join(f"- {r['title']}: {r['snippet']}" for r in results[:5])

            return ToolCallResult(
                success=True,
                output=output,
                data={"results": results},   # structured data for downstream tools
            )

        except Exception as e:
            return ToolCallResult(
                success=False,
                output=f"Web search failed: {e}",
            )
```

---

## The Fields You Must Fill In

### `name` (str)
The internal identifier. **Must be unique** across all tools.  
The planner uses this exact string in its JSON output, e.g. `{"tool": "web_search", ...}`.  
Keep it short, lowercase, underscored.

---

### `description` (str)
**The most important field.** The planner reads this to decide *when* to use the tool.  
Write it as if explaining to a smart person who has never seen your code:

- ✅ Describe *what the tool does* and *when it should be chosen*
- ✅ Mention what kind of input it expects and what kind of output it gives
- ❌ Don't describe implementation details ("calls the requests library")
- ❌ Don't be vague ("does stuff with the web")

**Good:**
```python
description = (
    "Search the web for current information, news, or facts. Use when the user "
    "asks about something that may have changed recently or that the assistant "
    "cannot know from training data alone."
)
```

**Bad:**
```python
description = "Web search tool."
```

---

### `input_schema` (str)
Describes what the tool's `run()` method expects as its `query` argument.  
This is read by the **LLM resolver** when this tool is chained after another tool.  
The resolver uses it to translate the previous tool's output into the right input.

- ✅ Describe format, length, and any constraints
- ✅ Include an example if the format is non-obvious
- ❌ Don't describe what the tool *does* (that's `description`)

**Good:**
```python
input_schema = (
    "A concise search query string, 3–8 words. "
    "Example: 'Python 3.13 new features' or 'weather Paris tomorrow'."
)
```

If your tool expects a **file path**:
```python
input_schema = (
    "An absolute file path (e.g. /home/user/documents/report.pdf). "
    "Must point to an existing file on disk."
)
```

If your tool is always called directly (never chained), this field still needs to exist but a short generic string is fine.

---

### `ack_label` (str)
A short phrase shown to the user while the tool is running.  
The orchestrator formats it as: `"<ack_label>..."` (e.g. `"searching the web..."`).  
Use lowercase, present participle. Leave as `""` for instant tools.

---

### `example_phrases` (list[str])
A list of user utterances that *should* trigger this tool.  
The planner is an LLM so these are hints, not hard rules — but they help a lot.

- Include varied phrasings, not just synonyms of the same sentence
- Include both short commands and longer natural-language requests
- 10–25 examples is a good range

---

### `is_conversational` (bool, default `False`)
Set to `True` **only** if the tool is purely a chat/LLM step and `run()` should never be called.  
The orchestrator detects this and routes directly to `_handle_conversational()`.  
The only built-in example is `ChatTool`. You almost certainly want `False`.

---

## Step 2 — Implement `run()`

```python
def run(self, query: str, data: dict) -> ToolCallResult:
    ...
```

### Arguments

| Argument | Type | What it contains |
|----------|------|-----------------|
| `query` | `str` | The instruction for this step. For the first step in a plan, this comes from the planner. For subsequent steps that depend on a prior step, this is the output of the **LLM resolver** — a string formatted according to your `input_schema`. |
| `data` | `dict` | The session data bag. Contains structured output from prior steps (e.g. `data["hits"]` from `file_search`, `data["path"]` from a resolved file). Safe to call `.get()` on anything. |

### Return: `ToolCallResult`

```python
@dataclass
class ToolCallResult:
    success: bool               # did the tool succeed?
    output: str                 # human-readable text — the summariser LLM reads this
    data: dict                  # structured data for downstream tools or context
    requires_confirmation: bool # set True to pause and ask the user before continuing
    confirmation_prompt: str    # the question shown to the user if above is True
```

**`output`** is what the summariser LLM (and the resolver LLM) will read.  
Write it as if describing results to a human — full paths, previews, clear status messages.  
The more informative it is, the better the resolver and summariser will perform.

**`data`** is for downstream tools. Put structured results here.  
Convention: if your tool returns file hits, use `{"hits": [{"path": ..., "name": ...}, ...]}`.  
If it returns a single path, use `{"path": "/absolute/path"}`.

### Confirmation gate
If your tool does something irreversible (delete a file, send a message, run a command):

```python
return ToolCallResult(
    success=False,
    output="",
    requires_confirmation=True,
    confirmation_prompt=(
        f"I'm about to delete {target}. "
        "This can't be undone. Are you sure? (yes / no)"
    ),
)
```

The executor will pause, show the prompt to the user, and wait for `"yes"` before continuing.

---

## Step 3 — Register the tool

Open `tools/registry.py` and add your tool:

```python
from tools.base import BaseTool
from tools.file_search import FileSearchTool
from tools.open_file import OpenFileTool
from tools.terminal import TerminalTool
from tools.chat import ChatTool
from tools.web_search import WebSearchTool   # ← add this import


def build_registry() -> dict[str, BaseTool]:
    tools: list[BaseTool] = [
        FileSearchTool(),
        OpenFileTool(),
        TerminalTool(),
        ChatTool(),
        WebSearchTool(),                      # ← add this line
    ]
    return {tool.name: tool for tool in tools}
```

That's it. AIDE will pick it up on next launch.

---

## Step 4 — Test it

Run AIDE normally and try some of your `example_phrases`. Watch the terminal output:

```
[Orchestrator] Plan: ['web_search']
[Executor] → web_search: 'Python 3.13 new features'
[Executor]   success=True
```

If the planner keeps choosing the wrong tool (or no tool), your `description` probably needs to be more specific about *when* to choose it vs. similar tools.

---

## Chaining tools together

If your tool is designed to be used *after* another tool (like `open_file` after `file_search`), you don't wire this up manually. The planner generates a `depends_on` link, and the executor automatically invokes the LLM resolver to translate the first tool's `output` into the right `query` for your tool, guided by your `input_schema`.

**Example chain:**  
`file_search` → `open_file`

1. `file_search.run()` returns a `ToolCallResult` with `output` listing five files and an `output` string describing them.
2. The executor calls the resolver with: the original user request + `file_search`'s output text + `open_file.input_schema`.
3. The resolver produces `"/home/user/docs/2236B.cpp"`.
4. `open_file.run("/home/user/docs/2236B.cpp", data)` is called.

For this to work well, make sure your **`output` string is rich and readable** — the resolver reads it as plain text.

---

## Quick reference checklist

```
tools/your_tool.py
  ✓ name            — unique, lowercase, underscored
  ✓ description     — clear, specific, tells the planner WHEN to use it
  ✓ input_schema    — tells the resolver WHAT format query should be in
  ✓ ack_label       — short present-participle phrase (or "" if instant)
  ✓ example_phrases — 10–25 varied natural-language triggers
  ✓ run()           — returns ToolCallResult(success, output, data)

tools/registry.py
  ✓ import added
  ✓ instance added to the list in build_registry()
```