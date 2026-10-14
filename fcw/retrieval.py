"""Vector retrieval over the context notes (FAISS, exact inner-product search).

Encoder:
  * "minilm" -- sentence-transformers all-MiniLM-L6-v2 (the intended backend)
  * "tfidf"  -- scikit-learn TF-IDF, used automatically if MiniLM can't be loaded
                (no package / no network to download the model)
Both produce L2-normalised vectors, so inner product == cosine similarity.

With route filtering on (config.RETRIEVAL_ROUTE_FILTER), only notes that apply to
the route -- its own notes plus "All Routes" notes -- compete for the top-k slots.

Retrieval only *proposes* candidate notes. Whether a note justifies a flag is
decided by the deterministic rules in justify.py, never by similarity score.
"""
from __future__ import annotations

import calendar
import logging
from dataclasses import dataclass

import faiss
import numpy as np

from . import config
from .notes import Note

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Hit:
    note: Note
    score: float
    rank: int


class _MiniLMEncoder:
    name = "minilm"

    def __init__(self, texts: list[str]):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self._model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vecs, dtype="float32")


class _TfidfEncoder:
    name = "tfidf"

    def __init__(self, texts: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
        self._vec.fit(texts)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._vec.transform(texts).toarray().astype("float32")  # rows are L2-normalised


def _make_encoder(backend: str, texts: list[str]):
    if backend in ("auto", "minilm"):
        try:
            return _MiniLMEncoder(texts)
        except Exception as exc:  # ImportError, download failure, ...
            if backend == "minilm":
                raise
            log.warning("MiniLM unavailable (%s); falling back to TF-IDF vectors.", type(exc).__name__)
    if backend in ("auto", "tfidf"):
        return _TfidfEncoder(texts)
    raise ValueError(f"unknown retriever backend {backend!r}")


def build_query(route: str, week_of: str) -> str:
    y, m, _ = (int(p) for p in week_of.split("-"))
    return f"{route} route freight cost rise in the week of {week_of} ({calendar.month_name[m]} {y})"


class NoteRetriever:
    def __init__(self, notes: list[Note], backend: str = "auto"):
        self.notes = notes
        texts = [n.index_text() for n in notes]
        self._encoder = _make_encoder(backend, texts)
        self.backend = self._encoder.name
        vectors = self._encoder.encode(texts)
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)

    def search(self, query: str, k: int = config.RETRIEVAL_TOP_K, route: str | None = None) -> list[Hit]:
        """Top-k notes by cosine similarity.

        If `route` is given, notes for other routes are filtered out first (metadata
        filter). They could never pass the justification gate, so this just stops them
        from taking top-k slots away from notes that apply to the route (e.g. "All Routes").
        """
        q = self._encoder.encode([query])
        scores, ids = self._index.search(q, len(self.notes))  # tiny corpus: score all
        ranked = sorted(
            (
                (round(float(s), 6), self.notes[i])
                for s, i in zip(scores[0], ids[0])
                if i >= 0 and (route is None or self.notes[i].covers_route(route))
            ),
            key=lambda t: (-t[0], t[1].note_id),  # deterministic tie-break on note_id
        )
        return [Hit(note=n, score=s, rank=r + 1) for r, (s, n) in enumerate(ranked[:k])]
