from core.llm import LLM, clean_reply, extract_json_objects


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_chat_payload_and_think_stripping(monkeypatch):
    sent = {}

    def fake_post(url, json, timeout):
        sent.update(url=url, body=json)
        return FakeResponse({"choices": [{"message": {"content": "<think>hmm</think>  Hello! "}}]})

    monkeypatch.setattr("core.llm.requests.post", fake_post)
    llm = LLM("http://host:8081/", "Bonsai2-27b")
    reply = llm.ask("hi", system="be nice", think=True, max_tokens=7)

    assert reply == "Hello!"
    assert sent["url"] == "http://host:8081/v1/chat/completions"
    assert sent["body"]["model"] == "Bonsai2-27b"
    assert sent["body"]["max_tokens"] == 7
    assert sent["body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert [m["role"] for m in sent["body"]["messages"]] == ["system", "user"]


def test_embed_reads_openai_style_response(monkeypatch):
    monkeypatch.setattr(
        "core.llm.requests.post",
        lambda url, json, timeout: FakeResponse({"data": [{"embedding": [0.1, 0.2]}]}),
    )
    assert LLM("http://x", "nomic-embed-text").embed("hello") == [0.1, 0.2]


def test_extract_json_objects_handles_fences_chatter_and_siblings():
    raw = 'Sure!\n```json\n{"threads": [1]}\n```\nand also {"threads": [2]} done'
    assert list(extract_json_objects(raw)) == [{"threads": [1]}, {"threads": [2]}]


def test_extract_json_objects_skips_garbage():
    assert list(extract_json_objects("no json {here")) == []


def test_clean_reply():
    assert clean_reply("```bash\nls -l\n```") == "ls -l"
    assert clean_reply("'/home/me/a.py'") == "/home/me/a.py"
    assert clean_reply("echo 'hi'") == "echo 'hi'"   # inner quotes survive


def test_complete_returns_server_stats(monkeypatch):
    payload = {
        "choices": [{"message": {"content": "hi"}}],
        "usage": {"completion_tokens": 3},
        "timings": {"predicted_per_second": 42.0},
    }
    monkeypatch.setattr("core.llm.requests.post", lambda url, json, timeout: FakeResponse(payload))
    text, stats = LLM("http://x", "m").complete([{"role": "user", "content": "q"}])
    assert text == "hi"
    assert stats["usage"]["completion_tokens"] == 3 and stats["timings"]["predicted_per_second"] == 42.0
