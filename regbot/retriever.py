"""Hybrid retriever over the regulation chunks.

Two lexical rankers that need no model downloads are fused with Reciprocal Rank Fusion:
  * BM25 over stemmed words (strong for exact regulation vocabulary), and
  * TF-IDF cosine over word uni+bigrams (rewards matching phrases).
If REGBOT_EMBED_MODEL is set and `sentence-transformers` is installed, a dense
semantic ranker joins the fusion as a third signal.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from . import config
from .ingest import Chunk

_STOP = set(
    """a an and are as at be been but by can could do does for from had has have how i if in
    into is it its may me might my no not of on or our shall should so than that the their
    them then there these they this to us was we were what when where which who will with
    would you your yours am im i'm get got many much often need want know tell please also
    any some about""".split()
)

# Student phrasing -> wording the regulations actually use.
_SYNONYMS = {
    "cheat": "assessment offence plagiarism collusion",
    "cheating": "assessment offence plagiarism collusion contract cheating",
    "chatgpt": "artificial intelligence ai",
    "ai": "artificial intelligence",
    "pause": "interrupt interruption",
    "break": "interrupt interruption",
    "quit": "cancel registration",
    "drop": "cancel registration",
    "dropout": "cancel registration",
    "attempt": "attempt number permitted",
    "attempts": "attempt number permitted",
    "resit": "resit attempt reassessment",
    "retake": "resit attempt repeat",
    "fail": "fail attempts resit",
    "failed": "fail attempts resit",
    "grade": "mark",
    "grades": "marks",
    "gpa": "classification award",
    "honours": "classification",
    "late": "deadline late",
    "sick": "illness mitigating circumstances",
    "ill": "illness mitigating circumstances",
    "exam": "examination",
    "exams": "examination",
    "extend": "extension",
    "complain": "complaint appeal",
    "appeal": "appeal complaint",
    "remark": "recheck marks",
    "transfer": "transfer registration",
    "credit": "credit recognition prior learning",
    "rpl": "recognition prior learning",
    "dissertation": "final project",
    "thesis": "final project",
    "capstone": "final project",
}

_TOKEN = re.compile(r"cm\d{4}|\d+\.\d+|\d+%|[a-z][a-z'’-]*|\d+")


def _stem(w: str) -> str:
    for suf in ("ational", "ations", "ation", "ments", "ment", "ings", "ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def tokenize(text: str, expand: bool = False) -> list[str]:
    words = [w.strip("'’-") for w in _TOKEN.findall(text.lower())]
    words = [w for w in words if w and w not in _STOP]
    if expand:
        extra = [s for w in words if w in _SYNONYMS for s in _SYNONYMS[w].split()]
        words += extra
    return [_stem(w) for w in words]


@dataclass
class Hit:
    chunk: Chunk
    score: float


class _BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(d) for d in docs]
        self.lens = np.array([len(d) for d in docs], dtype=float)
        self.avg = self.lens.mean()
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, q: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tfs))
        norm = self.k1 * (1 - self.b + self.b * self.lens / self.avg)
        for t in set(q):
            idf = self.idf.get(t)
            if idf is None:
                continue
            tf = np.array([d.get(t, 0) for d in self.tfs], dtype=float)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out


class _TfIdf:
    def __init__(self, docs: list[list[str]]):
        grams = [self._grams(d) for d in docs]
        vocab = sorted({g for gs in grams for g in gs})
        self.index = {g: i for i, g in enumerate(vocab)}
        n = len(docs)
        df = Counter(g for gs in grams for g in set(gs))
        self.idf = np.array([math.log((1 + n) / (1 + df[g])) + 1 for g in vocab])
        self.matrix = np.vstack([self._vec(gs) for gs in grams])

    @staticmethod
    def _grams(toks: list[str]) -> list[str]:
        return toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]

    def _vec(self, grams: list[str]) -> np.ndarray:
        v = np.zeros(len(self.index))
        for g, c in Counter(grams).items():
            if (i := self.index.get(g)) is not None:
                v[i] = (1 + math.log(c)) * self.idf[i]
        n = np.linalg.norm(v)
        return v / n if n else v

    def scores(self, q: list[str]) -> np.ndarray:
        return self.matrix @ self._vec(self._grams(q))


class _Dense:
    def __init__(self, texts: list[str], model_name: str):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model = SentenceTransformer(model_name)
        self.matrix = self.model.encode(texts, normalize_embeddings=True)

    def scores(self, query: str) -> np.ndarray:
        return self.matrix @ self.model.encode([query], normalize_embeddings=True)[0]


class HybridRetriever:
    RRF_K = 60

    def __init__(self, chunks: list[Chunk], embed_model: str = config.EMBED_MODEL):
        self.chunks = chunks
        docs = [tokenize(f"{c.header}\n{c.text}") for c in chunks]
        self.bm25 = _BM25(docs)
        self.tfidf = _TfIdf(docs)
        self.dense = _Dense([f"{c.header}\n{c.text}" for c in chunks], embed_model) if embed_model else None

    def _rankings(self, query: str) -> list[np.ndarray]:
        q = tokenize(query, expand=True)
        ranks = [self.bm25.scores(q), self.tfidf.scores(q)]
        if self.dense is not None:
            ranks.append(self.dense.scores(query))
        return ranks

    def search(self, query: str, k: int = config.TOP_K) -> list[Hit]:
        fused = np.zeros(len(self.chunks))
        for scores in self._rankings(query):
            order = np.argsort(-scores)
            ranked = order[scores[order] > 0]
            fused[ranked] += 1.0 / (self.RRF_K + np.arange(1, len(ranked) + 1))

        # A question that cites a regulation number ("what does 4.27 say?") should get it.
        for ref in re.findall(r"\b(\d{1,2}\.\d{1,2})\b", query):
            for i, c in enumerate(self.chunks):
                if c.reg == ref:
                    fused[i] += 1.0

        order = [i for i in np.argsort(-fused) if fused[i] > 0][:k]
        return [Hit(self.chunks[i], float(fused[i])) for i in order]
