"""Central configuration. Every value can be overridden with an environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Minimal .env support (KEY=value lines); real environment variables take precedence.
if (ROOT / ".env").exists():
    for _line in (ROOT / ".env").read_text().splitlines():
        _key, _sep, _value = _line.partition("=")
        if _sep and not _key.strip().startswith("#") and _value.split("#")[0].strip():
            os.environ.setdefault(_key.strip(), _value.split("#")[0].strip())

DATA_DIR = ROOT / "data"
INDEX_DIR = Path(os.getenv("REGBOT_INDEX_DIR", ROOT / "index"))


@dataclass(frozen=True)
class SourceDoc:
    doc_id: str
    title: str
    short: str  # label used in citations
    path: Path


# The only two documents the chatbot is allowed to know about.
SOURCES: tuple[SourceDoc, ...] = (
    SourceDoc(
        doc_id="general",
        title="University of London General Regulations 2026-2027",
        short="General Regs",
        path=DATA_DIR / "general-regs-2026-27.pdf",
    ),
    SourceDoc(
        doc_id="cs",
        title="Programme Regulations 2026-2027: Computer Science (BSc, Graduate Diploma, Graduate Certificate)",
        short="CS Programme Regs",
        path=DATA_DIR / "prog-regs-computer-science-bsc-2026-27.pdf",
    ),
)

MODEL = os.getenv("REGBOT_MODEL", "claude-opus-5-5")
# Effort controls answer depth vs. latency/cost: low | medium | high | xhigh | max
EFFORT = os.getenv("REGBOT_EFFORT", "medium")
MAX_TOKENS = int(os.getenv("REGBOT_MAX_TOKENS", "16000"))

TOP_K = int(os.getenv("REGBOT_TOP_K", "8"))
MAX_CHUNK_CHARS = int(os.getenv("REGBOT_MAX_CHUNK_CHARS", "1800"))
# Optional dense embeddings (requires `pip install sentence-transformers`).
EMBED_MODEL = os.getenv("REGBOT_EMBED_MODEL", "")
