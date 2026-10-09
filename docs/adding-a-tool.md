# Adding a tool

A tool is one class plus one line in the registry. The planner, executor and orchestrator don't change.

## 1. Write the class

```python
# tools/word_count.py
from pathlib import Path

from tools.base import BaseTool, ToolCallResult, resolve_path


class WordCountTool(BaseTool):
    name = "word_count"
    description = (
        "Count the words in a local text file. Use when the user asks how long a file is. "
        "If the path isn't known, run file_search first."
    )
    input_schema = "An absolute path to an existing text file, e.g. /home/user/notes.txt."

    def run(self, query: str, data: dict) -> ToolCallResult:
        path = resolve_path(query, data)
        if not path:
            return ToolCallResult(False, "No file path available. Run a file search first.")
        count = len(Path(path).read_text(errors="ignore").split())
        return ToolCallResult(True, f"{Path(path).name} has {count} words.", {"path": path, "words": count})
```

### The fields

| Field | Read by | Purpose |
|---|---|---|
| `name` | planner | Unique identifier the planner puts in its JSON. Lowercase, underscored. |
| `description` | planner | **The important one.** Say what the tool does and *when to pick it*, including how it differs from similar tools. |
| `input_schema` | resolver | Format of the `query` string. Used when this tool runs after another one, so the resolver can turn the previous output into valid input. Include an example if the format isn't obvious. |
| `is_conversational` | orchestrator | Leave `False`. Only `chat` sets it. |

### `run(query, data)`

- `query`: the step's input. For a first step it's the planner's text; for a step with `depends_on` it's
  whatever the resolver produced from the earlier outputs, shaped by your `input_schema`.
- `data`: session data merged with the structured `data` of the steps this one depends on.
  Conventions: `hits` (file search), `results` (web search), `path`, `url`.

Return `ToolCallResult(success, output, data)`. `output` is text that the resolver and summarizer read, so
make it informative (full paths, clear status). `data` is for downstream tools.

### Asking for confirmation

For anything destructive or irreversible:

```python
if not data.get("confirmed"):
    return ToolCallResult(
        False, "",
        requires_confirmation=True,
        confirmation_prompt=f"I'm about to delete {target}. This can't be undone. Proceed?",
    )
# ... do the thing
```

The executor pauses the thread, the orchestrator asks the user, and on "yes" the same step runs again
with `data["confirmed"] = True`.

Tools that need the model take an `LLM` in `__init__` (see `tools/summarize.py`).

## 2. Register it

In `tools/registry.py`, import the class and add an instance to the list in `build_registry()`.

## 3. Test it

Tools are plain classes, so they're easy to test without a model: call `run()` directly, or use
`EchoTool` and `FakeLLM` from `tests/conftest.py` to test how the planner and executor treat your tool.

```python
def test_word_count(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("one two three")
    assert WordCountTool().run(str(f), {}).data["words"] == 3
```

If the planner picks the wrong tool (or ignores yours), sharpen `description`. That's almost always the fix.
