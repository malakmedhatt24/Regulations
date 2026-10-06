"""Retrieve-then-generate pipeline shared by the web app and the CLI."""

from __future__ import annotations

from collections.abc import Iterator

from . import config
from .generator import make_generator
from .ingest import load_chunks
from .retriever import Hit, HybridRetriever


class RegBot:
    def __init__(self, top_k: int = config.TOP_K):
        self.top_k = top_k
        self.retriever = HybridRetriever(load_chunks())
        self.generator = make_generator()

    def retrieve(self, question: str, history: list[dict] | None = None) -> list[Hit]:
        hits = self.retriever.search(question, self.top_k)
        # Follow-ups ("what if I fail it again?") lean on the previous question for context.
        prev = [m["content"] for m in (history or []) if m["role"] == "user"]
        if prev:
            seen = {h.chunk.id for h in hits}
            for h in self.retriever.search(f"{prev[-1]} {question}", self.top_k):
                if h.chunk.id not in seen and len(hits) < self.top_k + 3:
                    hits.append(h)
                    seen.add(h.chunk.id)
        return hits

    def answer(self, question: str, history: list[dict] | None = None) -> tuple[list[Hit], Iterator[str]]:
        history = history or []
        hits = self.retrieve(question, history)
        return hits, self.generator.stream(question, hits, history)
