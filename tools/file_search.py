import hashlib
import logging
import os
import time
from pathlib import Path

import chromadb
import requests
from pypdf import PdfReader

from core.config import ROOT, load_config
from core.llm import LLM
from tools.base import BaseTool, ToolCallResult

log = logging.getLogger("aide.file_search")

# Full text is embedded for these.
TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".html", ".css", ".json", ".yaml", ".yml", ".toml",
    ".csv", ".rst", ".c", ".cpp", ".h", ".java", ".rs", ".go", ".sh", ".tex", ".org", ".xml",
}
# Only the file name and path are embedded for these.
NAME_ONLY_EXTENSIONS = {
    ".docx", ".xlsx", ".pptx", ".odt", ".ods", ".png", ".jpg", ".jpeg", ".gif",
    ".mp4", ".mp3", ".zip", ".tar", ".gz",
}
MAX_FILE_SIZE = 10_000_000   # bytes; larger text files are skipped
PDF_PAGE_LIMIT = 10
EMBED_CHARS = 2_000          # how much of a file is embedded
PREVIEW_CHARS = 300
MAX_RESULTS = 10


class FileSearchTool(BaseTool):
    name = "file_search"
    description = (
        "Semantically search the user's files (by name and content) and return the best matches "
        "with their paths. Use it to find or locate a file, or before open_file / read_file."
    )
    input_schema = (
        "A few keywords for a semantic file search, taken from the user's request and context. "
        "Keywords only: no quotes, punctuation, grammar or explanation."
    )

    def __init__(self, embedder: LLM, db_path: Path = ROOT / ".chroma"):
        cfg = load_config()["paths"]
        self.embedder = embedder
        self.roots = [Path(p).expanduser() for p in cfg["file_search_roots"]]
        self.exclude = set(cfg.get("file_search_exclude", []))
        self.collection = chromadb.PersistentClient(path=str(db_path)).get_or_create_collection(
            name="files", metadata={"hnsw:space": "cosine"}
        )
        log.info("index has %d documents", self.collection.count())

    # ---- indexing ----

    @staticmethod
    def _file_id(path: Path) -> str:
        return hashlib.md5(str(path.resolve()).encode()).hexdigest()

    @staticmethod
    def _read_text(path: Path) -> str | None:
        try:
            if path.stat().st_size > MAX_FILE_SIZE:
                return None
            return path.read_text(encoding="utf-8", errors="ignore")
        except OSError:   # permissions, broken symlink, ...
            return None

    @staticmethod
    def _read_pdf(path: Path) -> str:
        try:
            pages = (page.extract_text() or "" for page in PdfReader(str(path)).pages[:PDF_PAGE_LIMIT])
            return "\n\n".join(p.strip() for p in pages if p.strip())
        except Exception as e:   # pypdf raises many things on damaged files
            log.warning("could not read PDF %s: %s", path.name, e)
            return ""

    def _extract(self, path: Path) -> tuple[str, str] | None:
        """Return (text to embed, preview), or None if the file isn't indexable."""
        suffix = path.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            body = self._read_text(path)
            if body is None:
                return None
        elif suffix == ".pdf":
            body = self._read_pdf(path)
        elif suffix in NAME_ONLY_EXTENSIONS:
            body = ""
        else:
            return None

        if body:
            return f"Filename: {path.name}\n\n{body[:EMBED_CHARS]}", body[:PREVIEW_CHARS]
        return f"Filename: {path.name}\nPath: {path}", ""

    def _index_file(self, path: Path) -> bool:
        """Embed one file unless unchanged. Returns True if it was (re)indexed."""
        try:
            # Integer nanoseconds: Chroma rounds float metadata, which breaks equality checks.
            mtime = path.stat().st_mtime_ns
        except OSError:   # vanished or unreadable
            return False

        file_id = self._file_id(path)
        existing = self.collection.get(ids=[file_id], include=["metadatas"])
        if existing["ids"] and existing["metadatas"][0].get("mtime_ns") == mtime:
            return False

        extracted = self._extract(path)
        if extracted is None:
            return False
        text, preview = extracted

        self.collection.upsert(
            ids=[file_id],
            embeddings=[self.embedder.embed(text)],
            documents=[text],
            metadatas=[{
                "path": str(path.resolve()),
                "name": path.name,
                "suffix": path.suffix.lower(),
                "mtime_ns": mtime,
                "preview": preview,
            }],
        )
        return True

    def index(self) -> None:
        """Walk the search roots and (re)index new or changed files. Safe to run in a thread."""
        log.info("indexing...")
        start, indexed = time.time(), 0

        for root in self.roots:
            if not root.exists():
                log.warning("skipping missing root %s", root)
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in self.exclude]
                for filename in filenames:
                    try:
                        indexed += self._index_file(Path(dirpath) / filename)
                    except requests.RequestException as e:
                        # Same failure would repeat for every file, so stop here.
                        log.error("embedding endpoint unavailable (%s); indexing stopped", e)
                        return

        log.info("indexed %d files in %.1fs (%d total)", indexed, time.time() - start, self.collection.count())

    # ---- querying ----

    def _query(self, keywords: str) -> list[dict]:
        count = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(
            query_embeddings=[self.embedder.embed(keywords)],
            n_results=min(MAX_RESULTS, count),
            include=["metadatas", "distances"],
        )
        return [
            {
                "path": meta["path"],
                "name": meta["name"],
                "similarity": round(1 - dist / 2, 3),   # cosine distance [0, 2] -> similarity [0, 1]
                "preview": meta.get("preview", ""),
            }
            for meta, dist in zip(results["metadatas"][0], results["distances"][0])
        ]

    def run(self, query: str, data: dict) -> ToolCallResult:
        try:
            hits = self._query(query)
        except requests.RequestException as e:
            return ToolCallResult(False, f"File search unavailable: the embedding endpoint failed ({e}).")
        if not hits:
            return ToolCallResult(False, "No files found. The index may still be building; try again in a moment.")

        output = "\n".join(
            f"{i}. {h['name']} ({h['similarity']:.0%} match)\n"
            f"   Path: {h['path']}\n"
            f"   Preview: {h['preview'][:150] or 'N/A'}"
            for i, h in enumerate(hits, 1)
        )
        return ToolCallResult(True, output, {"hits": hits})
