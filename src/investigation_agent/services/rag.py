"""Hybrid RAG over Qdrant: dense (bge-m3 on the H200) + sparse (BM25 in Qdrant), fused with RRF.

Dense retrieval finds paraphrases; BM25 finds exact terms (dataset names, metric names,
"Parte 1") that embeddings blur. Reciprocal Rank Fusion combines both rankings by position,
so their incomparable scores never need calibrating.
"""

import re
import unicodedata
import uuid
from pathlib import Path

import pymupdf
from qdrant_client import QdrantClient, models

from investigation_agent.config.h200 import H200Client
from investigation_agent.config.settings import Settings

# Fixed namespace for deterministic ids: the same chunk of the same source is the same point.
_ID_NAMESPACE = uuid.UUID("6f1c2a4e-3b8d-4c5a-9e7f-0a1b2c3d4e5f")
_MD_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.M)
_MD_FENCE = re.compile(r"^```.*?^```", re.M | re.S)


class RagService:
    def __init__(self, llm: H200Client, qdrant: QdrantClient, settings: Settings):
        self.llm = llm
        self.qdrant = qdrant
        self.settings = settings

    # ---------------------------------------------------------------- ingestion
    def ingest(self, path: str | Path) -> int:
        """Index a .pdf, .md or .txt file. Returns the number of chunks written.

        Re-ingesting a source replaces it: its old points are deleted first, so a shorter
        new version leaves no stale chunks behind."""
        path = Path(path)
        source = path.name
        chunks = [{"source": source, "section": section, "chunk_idx": i, "text": text}
                  for i, (section, text) in enumerate(self._chunk(self.sections(path)))]
        if not chunks:
            return 0
        dense = self.llm.embed([f"{c['section']}\n{c['text']}" for c in chunks])
        self.delete_source(source)
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(_ID_NAMESPACE, f"{source}#{c['chunk_idx']}")),
                vector={self.settings.qdrant_dense_vector: vector,
                        self.settings.qdrant_sparse_vector: self._bm25(c["text"])},
                payload=c)
            for c, vector in zip(chunks, dense)]
        for i in range(0, len(points), 128):
            self.qdrant.upsert(self.settings.qdrant_collection, points=points[i:i + 128], wait=True)
        return len(points)

    def delete_source(self, source: str) -> None:
        self.qdrant.delete(
            self.settings.qdrant_collection,
            points_selector=models.Filter(must=[
                models.FieldCondition(key="source", match=models.MatchValue(value=source))]),
            wait=True)

    def has_source(self, source: str) -> bool:
        return self.qdrant.count(
            self.settings.qdrant_collection, exact=True,
            count_filter=models.Filter(must=[
                models.FieldCondition(key="source", match=models.MatchValue(value=source))]),
        ).count > 0

    @staticmethod
    def sections(path: Path) -> list[tuple[str, str]]:
        """(section, text) pairs. Markdown splits on headings; a PDF on pages, since its
        headings carry no markup."""
        if path.suffix.lower() == ".pdf":
            with pymupdf.open(path) as doc:
                return [(f"p. {n}", unicodedata.normalize("NFKC", page.get_text()))
                        for n, page in enumerate(doc, start=1)]
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix.lower() != ".md":
            return [(path.stem, text)]
        # A "# comment" inside a fenced code block is not a heading.
        fences = [m.span() for m in _MD_FENCE.finditer(text)]
        sections, title, start = [], path.stem, 0
        for m in _MD_HEADING.finditer(text):
            if any(a <= m.start() < b for a, b in fences):
                continue
            sections.append((title, text[start:m.start()]))
            title, start = m.group(1), m.end()
        sections.append((title, text[start:]))
        return sections

    def _chunk(self, sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Pack paragraphs into chunks of at most `rag_chunk_size` characters, carrying the
        last `rag_chunk_overlap` characters into the next chunk. Chunks never cross sections."""
        size, overlap = self.settings.rag_chunk_size, self.settings.rag_chunk_overlap
        out: list[tuple[str, str]] = []
        for section, text in sections:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            # A paragraph longer than a chunk is cut into chunk-sized pieces.
            pieces = [p[i:i + size] for p in paragraphs for i in range(0, len(p), size)]
            current = ""
            for piece in pieces:
                if current and len(current) + len(piece) + 2 > size:
                    out.append((section, current))
                    current = current[-overlap:] if overlap else ""
                current = f"{current}\n\n{piece}" if current else piece
            if current:
                out.append((section, current))
        return out

    # ---------------------------------------------------------------- retrieval
    def retrieve(self, query: str, k: int | None = None,
                 where: dict[str, str | list[str]] | None = None) -> list[dict]:
        """Hybrid search: dense and BM25 each prefetch `rag_prefetch_k` candidates, RRF fuses
        them and the top `k` come back with their source, section and fused score.
        `where` matches payload fields: a value is an exact match, a list matches any of
        its items, e.g. {"source": ["tarea-a.pdf", "s3-agentes.md"]}."""
        k = k or self.settings.rag_top_k
        prefetch_k = max(self.settings.rag_prefetch_k, k)
        query_filter = None
        if where:
            query_filter = models.Filter(must=[
                models.FieldCondition(
                    key=field,
                    match=models.MatchAny(any=value) if isinstance(value, list)
                    else models.MatchValue(value=value))
                for field, value in where.items()])
        dense = self.llm.embed([query])[0]
        r = self.qdrant.query_points(
            self.settings.qdrant_collection,
            prefetch=[
                models.Prefetch(query=dense, using=self.settings.qdrant_dense_vector,
                                limit=prefetch_k, filter=query_filter),
                models.Prefetch(query=self._bm25(query), using=self.settings.qdrant_sparse_vector,
                                limit=prefetch_k, filter=query_filter),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
            with_payload=True)
        return [{**p.payload, "score": p.score} for p in r.points]

    def _bm25(self, text: str) -> models.Document:
        return models.Document(text=text, model=self.settings.qdrant_bm25_model,
                               options={"language": self.settings.qdrant_bm25_language})
