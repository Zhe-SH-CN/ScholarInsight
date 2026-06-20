"""本地论文检索工具，替代 Web SearchTool。

使用 sentence-transformers (bge-large-en-v1.5) + numpy 做 embedding 检索。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from cg.schemas.research import SourceCandidate
from cg.settings import Settings


QUERY_STOPWORDS = {
    "about",
    "across",
    "analysis",
    "and",
    "are",
    "application",
    "applications",
    "approach",
    "approaches",
    "based",
    "bottleneck",
    "bottlenecks",
    "by",
    "challenge",
    "challenges",
    "comparison",
    "comparisons",
    "design",
    "direction",
    "directions",
    "experiment",
    "experiments",
    "for",
    "framework",
    "frameworks",
    "from",
    "in",
    "method",
    "methods",
    "new",
    "of",
    "on",
    "or",
    "paper",
    "papers",
    "recent",
    "research",
    "results",
    "review",
    "study",
    "survey",
    "system",
    "systems",
    "the",
    "to",
    "using",
    "with",
}


def normalize_term(term: str) -> str:
    """Normalize query/paper terms enough for lexical reranking."""
    term = term.lower().strip("-_")
    aliases = {
        "grounded": "ground",
        "grounding": "ground",
        "graphrag": "graph_rag",
        "graphs": "graph",
        "kgs": "kg",
        "llms": "llm",
        "gnns": "gnn",
        "temporal": "dynamic",
        "evolving": "dynamic",
        "evolution": "dynamic",
        "updates": "update",
    }
    if term in aliases:
        return aliases[term]
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 4 and term.endswith("s"):
        return term[:-1]
    return term


def query_terms(query: str) -> list[str]:
    """Return meaningful query terms, preserving order."""
    seen: set[str] = set()
    terms: list[str] = []
    for raw in re.findall(r"[a-z0-9]{2,}", query.lower()):
        term = normalize_term(raw)
        if term in QUERY_STOPWORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def query_phrases(terms: list[str]) -> list[str]:
    phrases: list[str] = []
    for size in (3, 2):
        for index in range(0, max(0, len(terms) - size + 1)):
            phrase = " ".join(terms[index:index + size])
            if phrase not in phrases:
                phrases.append(phrase)
    return phrases[:8]


class LocalPaperIndex:
    """基于 embedding 的本地论文索引。"""

    def __init__(self, index_path: str, embeddings_path: str | None = None):
        self.papers: list[dict] = []
        self.searchable_papers: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self._model = None
        self._model_error: str | None = None
        self._lexical_cache: list[dict[str, object]] | None = None

        if Path(index_path).exists():
            with open(index_path, "r", encoding="utf-8") as f:
                self.papers = json.load(f)
        self.searchable_papers = [paper for paper in self.papers if self._is_valid_paper(paper)]

        # 加载 embeddings
        if embeddings_path is None:
            embeddings_path = str(Path(index_path).parent / "embeddings.npy")
        if Path(embeddings_path).exists():
            self.embeddings = np.load(embeddings_path)

    def _is_valid_paper(self, paper: dict) -> bool:
        path = str(paper.get("pdf_path") or "")
        if not path.lower().endswith(".pdf"):
            return False
        if Path(path).name.startswith("._"):
            return False
        return "error" not in paper

    def _get_model(self):
        """延迟加载 sentence-transformers 模型。"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        return self._model

    def _encode_query(self, query: str) -> np.ndarray | None:
        if self._model_error:
            return None
        try:
            model = self._get_model()
            encoded = model.encode([query], normalize_embeddings=True)
            return np.asarray(encoded, dtype=np.float32)[0]
        except Exception as exc:
            self._model_error = f"{type(exc).__name__}: {exc}"
            return None

    def search(self, query: str, max_results: int = 10) -> list[SourceCandidate]:
        """基于 embedding cosine similarity 检索，并用 lexical relevance 重排。"""
        if not self.searchable_papers:
            return []

        if self.embeddings is not None and len(self.embeddings) > 0:
            return self._embedding_search(query, max_results)

        # fallback: 简单文本匹配
        return self._text_search(query, max_results)

    def _embedding_search(self, query: str, max_results: int) -> list[SourceCandidate]:
        embedding_count = min(len(self.embeddings), len(self.searchable_papers))  # type: ignore[arg-type]
        papers = self.searchable_papers[:embedding_count]
        embeddings = self.embeddings[:embedding_count]  # type: ignore[index]
        terms = query_terms(query)
        phrases = query_phrases(terms)
        lexical_scores = np.asarray(
            [self._lexical_score_for_index(terms, phrases, idx) for idx in range(embedding_count)],
            dtype=np.float32,
        )

        query_emb = self._encode_query(query)
        if query_emb is None:
            query_emb = self._pseudo_query_embedding(lexical_scores, embeddings)

        if query_emb is not None:
            embedding_scores = np.dot(embeddings, query_emb.T).flatten()
        else:
            embedding_scores = np.zeros(embedding_count, dtype=np.float32)

        pool_size = min(embedding_count, max(max_results * 12, 120))
        embedding_top = np.argsort(embedding_scores)[::-1][:pool_size]
        lexical_top = np.argsort(lexical_scores)[::-1][:pool_size]
        pool_indices = sorted(set(int(idx) for idx in embedding_top) | set(int(idx) for idx in lexical_top))

        ranked: list[tuple[float, float, float, int]] = []
        for idx in pool_indices:
            embedding_score = float(embedding_scores[idx])
            lexical_score = float(lexical_scores[idx])
            combined_score = embedding_score * 0.60 + lexical_score * 0.40
            ranked.append((combined_score, embedding_score, lexical_score, idx))
        ranked.sort(key=lambda item: item[0], reverse=True)

        if terms:
            lexical_floor = self._lexical_floor(set(terms))
            on_topic = [item for item in ranked if item[2] >= lexical_floor]
            if len(on_topic) >= min(max_results, 3):
                ranked = on_topic

        results = []
        for combined_score, _embedding_score, _lexical_score, idx in ranked[:max_results]:
            results.append(self._candidate_from_paper(papers[idx], query, combined_score))
        return results

    def _pseudo_query_embedding(
        self,
        lexical_scores: np.ndarray,
        embeddings: np.ndarray,
    ) -> np.ndarray | None:
        seed_indices = np.argsort(lexical_scores)[::-1][:32]
        seed_indices = np.asarray([idx for idx in seed_indices if lexical_scores[idx] > 0], dtype=np.int64)
        if seed_indices.size == 0:
            return None
        weights = lexical_scores[seed_indices].astype(np.float32)
        query_vec = np.average(embeddings[seed_indices], axis=0, weights=weights)
        norm = float(np.linalg.norm(query_vec))
        if norm <= 0:
            return None
        return (query_vec / norm).astype(np.float32)

    def _lexical_floor(self, term_set: set[str]) -> float:
        if {"knowledge", "graph"}.issubset(term_set):
            return 0.10
        if {"knowledge", "ground"}.issubset(term_set):
            return 0.10
        if "rag" in term_set or {"retrieval", "augmented", "generation"}.issubset(term_set):
            return 0.10
        return 0.06

    def _text_search(self, query: str, max_results: int) -> list[SourceCandidate]:
        """Fallback: 基于文本匹配的检索。"""
        terms = query_terms(query)
        phrases = query_phrases(terms)
        scored = []
        for idx, paper in enumerate(self.searchable_papers):
            score = self._lexical_score_for_index(terms, phrases, idx)
            scored.append((score, paper))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, paper in scored[:max_results]:
            if score > 0:
                results.append(self._candidate_from_paper(paper, query, score))
        return results

    def _candidate_from_paper(self, paper: dict, query: str, score: float) -> SourceCandidate:
        return SourceCandidate(
            url=paper.get("pdf_path", ""),
            title=paper.get("title", ""),
            snippet=paper.get("abstract", "")[:300],
            content=paper.get("focused_text", ""),
            source_type="academic_paper",
            query=query,
            score=min(max(float(score), 0.0), 1.0),
            source_provider="local_papers",
        )

    def _ensure_lexical_cache(self) -> list[dict[str, object]]:
        if self._lexical_cache is not None:
            return self._lexical_cache
        cache: list[dict[str, object]] = []
        for paper in self.searchable_papers:
            title = paper.get("title", "").lower()
            abstract = paper.get("abstract", "").lower()
            focused = paper.get("focused_text", "").lower()[:3000]
            text = f"{title} {abstract} {focused}"
            cache.append(
                {
                    "text": text,
                    "terms": {normalize_term(term) for term in re.findall(r"[a-z0-9]{2,}", text)},
                    "title_terms": {normalize_term(term) for term in re.findall(r"[a-z0-9]{2,}", title)},
                }
            )
        self._lexical_cache = cache
        return cache

    def _lexical_score_for_index(self, terms: list[str], phrases: list[str], idx: int) -> float:
        return self._lexical_score_from_cache(terms, phrases, self._ensure_lexical_cache()[idx])

    def _lexical_score_from_cache(self, terms: list[str], phrases: list[str], cached: dict[str, object]) -> float:
        if not terms:
            return 0.0
        text = str(cached["text"])
        text_terms = cached["terms"]
        matched = sum(1 for term in terms if term in text_terms)
        title_terms = cached["title_terms"]
        title_matched = sum(1 for term in terms if term in title_terms)
        phrase_hits = sum(1 for phrase in phrases if phrase in text)

        term_score = matched / max(1, len(terms))
        title_score = title_matched / max(1, len(terms))
        phrase_score = phrase_hits / max(1, len(phrases))
        score = min(1.0, term_score * 0.70 + title_score * 0.20 + phrase_score * 0.10)

        term_set = set(terms)
        has_kg = self._has_kg_signal(text, text_terms)
        has_rag = self._has_rag_signal(text, text_terms)
        has_dynamic = self._has_dynamic_signal(text, text_terms)
        query_needs_kg = {"knowledge", "graph"}.issubset(term_set) or "kg" in term_set or "graph_rag" in term_set
        query_needs_rag = (
            "rag" in term_set
            or "graph_rag" in term_set
            or {"retrieval", "augmented", "generation"}.issubset(term_set)
        )
        query_needs_dynamic = bool(term_set & {"dynamic", "update", "continual", "time", "event", "stream"})

        if query_needs_kg and query_needs_rag and not (has_kg and has_rag):
            return 0.0
        if query_needs_kg and query_needs_dynamic and not has_dynamic:
            return 0.0

        if {"knowledge", "graph"}.issubset(term_set):
            if not has_kg:
                return 0.0
        if {"knowledge", "ground"}.issubset(term_set):
            has_grounded_phrase = "knowledge grounded" in text or "knowledge-grounded" in text
            has_grounding_phrase = "knowledge grounding" in text or "knowledge-grounding" in text
            has_rag_grounding = has_rag
            if not (has_grounded_phrase or has_grounding_phrase or has_rag_grounding):
                return 0.0
        if "rag" in term_set or {"retrieval", "augmented", "generation"}.issubset(term_set):
            if not has_rag:
                score *= 0.35
        return score

    def _has_kg_signal(self, text: str, text_terms: object) -> bool:
        terms = text_terms if isinstance(text_terms, set) else set()
        return (
            "kg" in terms
            or "tkg" in terms
            or "graph_rag" in terms
            or "graphrag" in terms
            or "knowledge graph" in text
            or "knowledge graphs" in text
            or "knowledge-graph" in text
            or "graph retrieval augmented generation" in text
            or "graph retrieval-augmented generation" in text
        )

    def _has_rag_signal(self, text: str, text_terms: object) -> bool:
        terms = text_terms if isinstance(text_terms, set) else set()
        return (
            "rag" in terms
            or "graph_rag" in terms
            or "graphrag" in terms
            or "retrieval augmented generation" in text
            or "retrieval-augmented generation" in text
            or "retrieval augmented language model" in text
            or "retrieval-augmented language model" in text
        )

    def _has_dynamic_signal(self, text: str, text_terms: object) -> bool:
        terms = text_terms if isinstance(text_terms, set) else set()
        return bool(
            terms & {"dynamic", "temporal", "tkg", "update", "continual", "evolving", "evolution", "time", "event"}
        ) or any(
            phrase in text
            for phrase in (
                "time-varying",
                "time evolving",
                "time-evolving",
                "temporal knowledge graph",
                "dynamic knowledge graph",
                "evolving knowledge graph",
            )
        )


class LocalPaperSearchTool:
    """与 SearchTool 接口兼容的本地论文检索。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        index_path = getattr(settings, "scholar_paper_index_path", "paper_index.json")
        p = Path(index_path)
        if not p.is_absolute():
            # 先尝试直接相对于 data_dir
            candidate = (settings.data_dir / p.name).resolve()
            if candidate.exists():
                p = candidate
            else:
                # fallback: data_dir 下的子路径
                p = (settings.data_dir / index_path).resolve()
        self.index = LocalPaperIndex(str(p))

    @property
    def provider_names(self) -> list[str]:
        return ["local_papers"]

    @property
    def active_provider_names(self) -> list[str]:
        return ["local_papers"]

    async def search(self, query: str, max_results: int = 10) -> list[SourceCandidate]:
        return self.index.search(query, max_results)
