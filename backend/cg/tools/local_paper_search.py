"""本地论文检索工具，替代 Web SearchTool。

使用 sentence-transformers (bge-large-en-v1.5) + numpy 做 embedding 检索。
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
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


def use_direct_huggingface_endpoint() -> None:
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"


@dataclass(frozen=True)
class RerankedScore:
    combined_score: float
    relevance_score: float
    relevance_label: str
    reranker_score: float | None
    rejection_reason: str


@dataclass(frozen=True)
class SourceSubtype:
    label: str
    reason: str
    rejection_reason: str = ""


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


def compact_topic_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"([a-zA-Z])-\s+([a-zA-Z])", r"\1\2", text)
    return re.sub(r"\s+", " ", text.lower()).strip()


BAD_TITLE_PREFIXES = (
    "findings of the association",
    "proceedings of the",
    "published as a conference paper",
    "under review as a conference paper",
)

TITLE_STOPWORDS = {
    "anonymous authors",
    "paper under double-blind review",
    "abstract",
}

TITLE_SKIP_PREFIXES = BAD_TITLE_PREFIXES + (
    "july ",
    "august ",
    "september ",
    "copyright",
    "©",
)

TRUNCATED_TITLE_ENDINGS = {
    "a",
    "and",
    "as",
    "at",
    "for",
    "from",
    "in",
    "into",
    "knowledge",
    "language",
    "large",
    "of",
    "on",
    "plain",
    "the",
    "to",
    "using",
    "via",
    "with",
}

TITLE_TERMS = {
    "analysis",
    "approach",
    "augmented",
    "benchmark",
    "data",
    "efficient",
    "framework",
    "generation",
    "graph",
    "graphs",
    "knowledge",
    "language",
    "learning",
    "llm",
    "model",
    "models",
    "neural",
    "rag",
    "reasoning",
    "retrieval",
    "system",
    "systems",
}


def canonical_paper_title(paper: dict) -> str:
    """Return a runtime-corrected title without mutating paper_index.json."""
    raw_title = _clean_title(str(paper.get("title") or ""))
    focused_title = _title_from_focused_text(str(paper.get("focused_text") or ""))
    filename_title = _title_from_filename(str(paper.get("pdf_path") or ""))

    if _title_needs_replacement(raw_title):
        return focused_title or filename_title or raw_title
    if focused_title and _focused_title_extends_raw(raw_title, focused_title):
        return focused_title
    return raw_title or focused_title or filename_title


def _clean_title(title: str) -> str:
    title = title.replace("\u00ad", "")
    title = re.sub(r"\s+", " ", title).strip(" -_\t\r\n")
    return title


def _title_needs_replacement(title: str) -> bool:
    if len(re.findall(r"[a-zA-Z]", title)) < 4:
        return True
    title_lower = title.lower().strip()
    if title_lower.startswith(BAD_TITLE_PREFIXES):
        return True
    if Path(title).name.startswith("._"):
        return True
    last_word = re.sub(r"[^a-z0-9]+", "", title_lower.split()[-1]) if title_lower.split() else ""
    return last_word in TRUNCATED_TITLE_ENDINGS


def _focused_title_extends_raw(raw_title: str, focused_title: str) -> bool:
    raw_norm = _normalize_title_for_compare(raw_title)
    focused_norm = _normalize_title_for_compare(focused_title)
    if not raw_norm or not focused_norm:
        return False
    return focused_norm.startswith(raw_norm) and len(focused_norm) >= len(raw_norm) + 8


def _normalize_title_for_compare(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _title_from_focused_text(text: str) -> str:
    lines = []
    for raw_line in text.splitlines()[:120]:
        line = _clean_title(raw_line)
        if not line:
            continue
        if _is_numeric_line(line) or _is_title_skip_line(line):
            continue
        if line.lower() in TITLE_STOPWORDS:
            break
        if lines and _looks_like_author_or_affiliation_line(line):
            break
        if _looks_like_title_line(line):
            lines.append(line)
            if len(lines) >= 4:
                break
            continue
        if lines:
            break
    title = _join_title_lines(lines)
    if _title_needs_replacement(title):
        return ""
    return title


def _is_numeric_line(line: str) -> bool:
    return bool(re.fullmatch(r"\d{1,4}", line.strip()))


def _is_title_skip_line(line: str) -> bool:
    lowered = line.lower()
    if lowered.startswith(TITLE_SKIP_PREFIXES):
        return True
    if "conference on neural information processing systems" in lowered:
        return True
    if "association for computational linguistics" in lowered:
        return True
    return False


def _looks_like_title_line(line: str) -> bool:
    lowered = line.lower()
    if len(re.findall(r"[a-zA-Z]", line)) < 4:
        return False
    if "@" in line or lowered.startswith(("http://", "https://", "www.")):
        return False
    if lowered in TITLE_STOPWORDS or _is_title_skip_line(line):
        return False
    if _looks_like_author_or_affiliation_line(line):
        return False
    return True


def _looks_like_author_or_affiliation_line(line: str) -> bool:
    lowered = line.lower()
    if "@" in line:
        return True
    affiliation_pattern = (
        r"\b(university|institute|department|laboratory|college|center|centre|"
        r"technologies|inc|ltd|china|singapore|switzerland)\b|school of|hong kong"
    )
    if re.search(affiliation_pattern, lowered):
        return True
    if re.search(r"\d|[*†‡♢]", line):
        tokens = re.findall(r"[A-Za-z][A-Za-z'.-]+", line)
        title_terms = {token.lower().strip(".") for token in tokens} & TITLE_TERMS
        return not title_terms
    tokens = re.findall(r"[A-Za-z][A-Za-z'.-]+", line)
    if 2 <= len(tokens) <= 5:
        title_terms = {token.lower().strip(".") for token in tokens} & TITLE_TERMS
        capitalized = sum(1 for token in tokens if token[:1].isupper())
        if capitalized == len(tokens) and not title_terms and ":" not in line:
            return True
    return False


def _join_title_lines(lines: list[str]) -> str:
    title = ""
    for line in lines:
        if not title:
            title = line
        elif title.endswith("-"):
            title = title[:-1] + line
        elif _continues_uppercase_word(title, line):
            title += line
        else:
            title += " " + line
    return _clean_title(title)


def _continues_uppercase_word(title: str, line: str) -> bool:
    previous = re.search(r"([A-Z]{2,5})$", title)
    return bool(previous and re.fullmatch(r"[A-Z]{4,}", line))


def _title_from_filename(path: str) -> str:
    if not path:
        return ""
    stem = Path(path).stem
    if stem.startswith("._"):
        stem = stem[2:]
    if re.fullmatch(r"\d{4}\.[A-Za-z-]+\.\d+", stem):
        return ""
    stem = re.sub(r"^\d+_", "", stem)
    stem = re.sub(r"^\d{4}\.[A-Za-z-]+\.\d+_", "", stem)
    title = stem.replace("_", " ")
    title = re.sub(r"\s+", " ", title).strip()
    if _title_needs_replacement(title):
        return ""
    return title


class LocalPaperIndex:
    """基于 embedding 的本地论文索引。"""

    def __init__(
        self,
        index_path: str,
        embeddings_path: str | None = None,
        settings: Settings | None = None,
    ):
        use_direct_huggingface_endpoint()
        self.settings = settings or Settings()
        self.papers: list[dict] = []
        self.searchable_papers: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self._model = None
        self._model_error: str | None = None
        self._reranker = None
        self._reranker_error: str | None = None
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
            use_direct_huggingface_endpoint()
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

    def _get_reranker(self):
        """Lazy-load a sequence-classification reranker via transformers.

        sentence-transformers CrossEncoder currently tries AutoProcessor for
        bge-reranker-base in this environment, so use transformers directly.
        """
        if self._reranker is None:
            use_direct_huggingface_endpoint()

            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            device_setting = self.settings.scholar_reranker_device.strip().lower()
            if device_setting == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = device_setting
            tokenizer = AutoTokenizer.from_pretrained(self.settings.scholar_reranker_model)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.settings.scholar_reranker_model
            ).to(device)
            model.eval()
            self._reranker = (tokenizer, model, device, torch)
        return self._reranker

    def _rerank_scores(self, query: str, papers: list[dict]) -> list[float] | None:
        if not self.settings.scholar_enable_reranker or self._reranker_error:
            return None
        try:
            tokenizer, model, device, torch = self._get_reranker()
            batch_size = max(1, self.settings.scholar_reranker_batch_size)
            scores: list[float] = []
            docs = [self._reranker_text(paper) for paper in papers]
            with torch.no_grad():
                for start in range(0, len(docs), batch_size):
                    batch_docs = docs[start:start + batch_size]
                    inputs = tokenizer(
                        [query] * len(batch_docs),
                        batch_docs,
                        padding=True,
                        truncation=True,
                        max_length=self.settings.scholar_reranker_max_length,
                        return_tensors="pt",
                    ).to(device)
                    logits = model(**inputs).logits.reshape(-1)
                    scores.extend(float(value) for value in logits.detach().cpu())
            return scores
        except Exception as exc:
            self._reranker_error = f"{type(exc).__name__}: {exc}"
            return None

    def search(
        self,
        query: str,
        max_results: int = 10,
        target_topic: str | None = None,
    ) -> list[SourceCandidate]:
        """基于 embedding cosine similarity 检索，并用 lexical relevance 重排。"""
        if not self.searchable_papers:
            return []

        if self.embeddings is not None and len(self.embeddings) > 0:
            return self._embedding_search(query, max_results, target_topic=target_topic)

        # fallback: 简单文本匹配
        return self._text_search(query, max_results, target_topic=target_topic)

    def _embedding_search(
        self,
        query: str,
        max_results: int,
        *,
        target_topic: str | None = None,
    ) -> list[SourceCandidate]:
        embedding_count = min(len(self.embeddings), len(self.searchable_papers))  # type: ignore[arg-type]
        papers = self.searchable_papers[:embedding_count]
        embeddings = self.embeddings[:embedding_count]  # type: ignore[index]
        ranking_query = self._query_with_topic(query, target_topic)
        terms = query_terms(ranking_query)
        phrases = query_phrases(terms)
        lexical_scores = np.asarray(
            [self._lexical_score_for_index(terms, phrases, idx) for idx in range(embedding_count)],
            dtype=np.float32,
        )

        query_emb = self._encode_query(ranking_query)
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

        ranked = self._rerank_ranked(
            query,
            ranked,
            papers,
            max_results,
            target_topic=target_topic,
        )

        results = []
        accepted_count = 0
        rejected_count = 0
        for combined_score, embedding_score, lexical_score, idx in ranked:
            if isinstance(combined_score, RerankedScore):
                relevance = {
                    "score": combined_score.relevance_score,
                    "label": combined_score.relevance_label,
                    "reason": combined_score.rejection_reason,
                    "reranker_score": combined_score.reranker_score,
                }
                combined_value = combined_score.combined_score
            else:
                relevance = self._relevance_for_scores(
                    embedding_score=embedding_score,
                    lexical_score=lexical_score,
                    combined_score=combined_score,
                    reranker_score=None,
                )
                combined_value = combined_score
            if relevance["label"] == "reject":
                if rejected_count >= max_results:
                    continue
                rejected_count += 1
            else:
                if accepted_count >= max_results:
                    continue
                accepted_count += 1
            results.append(
                self._candidate_from_paper(
                    papers[idx],
                    query,
                    combined_value,
                    target_topic=target_topic,
                    embedding_score=embedding_score,
                    lexical_score=lexical_score,
                    reranker_score=relevance["reranker_score"],
                    relevance_score=relevance["score"],
                    relevance_label=relevance["label"],
                    rejection_reason=relevance["reason"],
                )
            )
            if accepted_count >= max_results and rejected_count >= max_results:
                break
        return results

    def _rerank_ranked(
        self,
        query: str,
        ranked: list[tuple[float, float, float, int]],
        papers: list[dict],
        max_results: int,
        *,
        target_topic: str | None = None,
    ) -> list[tuple[object, float, float, int]]:
        if not ranked:
            return []
        pool_size = min(
            len(ranked),
            max(max_results * 4, self.settings.scholar_reranker_pool_size),
        )
        pool = ranked[:pool_size]
        ranking_query = self._query_with_topic(query, target_topic)
        reranker_scores = self._rerank_scores(ranking_query, [papers[idx] for *_scores, idx in pool])
        rescored: list[tuple[object, float, float, int]] = []
        for position, (combined_score, embedding_score, lexical_score, idx) in enumerate(pool):
            reranker_score = reranker_scores[position] if reranker_scores is not None else None
            relevance = self._relevance_for_scores(
                embedding_score=embedding_score,
                lexical_score=lexical_score,
                combined_score=combined_score,
                reranker_score=reranker_score,
            )
            topic_rejection = self._topic_rejection_reason(
                query,
                papers[idx],
                target_topic=target_topic,
            )
            if topic_rejection:
                relevance = {
                    "score": 0.0,
                    "label": "reject" if self.settings.scholar_source_gate_enabled else "low",
                    "reason": topic_rejection,
                    "reranker_score": reranker_score,
                }
            rescored.append((
                RerankedScore(
                    combined_score=combined_score,
                    relevance_score=relevance["score"],
                    relevance_label=relevance["label"],
                    reranker_score=reranker_score,
                    rejection_reason=relevance["reason"],
                ),
                embedding_score,
                lexical_score,
                idx,
            ))
        rescored.sort(
            key=lambda item: (
                item[0].relevance_score if isinstance(item[0], RerankedScore) else float(item[0]),
                item[0].combined_score if isinstance(item[0], RerankedScore) else float(item[0]),
            ),
            reverse=True,
        )
        accepted = [
            item
            for item in rescored
            if isinstance(item[0], RerankedScore) and item[0].relevance_label != "reject"
        ]
        rejected = [
            item
            for item in rescored
            if isinstance(item[0], RerankedScore) and item[0].relevance_label == "reject"
        ]
        return accepted + rejected

    def _relevance_for_scores(
        self,
        *,
        embedding_score: float,
        lexical_score: float,
        combined_score: float,
        reranker_score: float | None,
    ) -> dict[str, object]:
        embedding_norm = min(1.0, max(0.0, (embedding_score + 1.0) / 2.0))
        if reranker_score is None:
            score = min(1.0, max(0.0, lexical_score * 0.62 + embedding_norm * 0.38))
        else:
            reranker_norm = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, reranker_score))))
            score = min(
                1.0,
                max(0.0, reranker_norm * 0.58 + lexical_score * 0.27 + embedding_norm * 0.15),
            )

        min_relevance = self.settings.scholar_min_source_relevance
        if score >= 0.62:
            label = "high"
        elif score >= 0.35:
            label = "medium"
        elif score >= min_relevance:
            label = "low"
        else:
            label = "reject" if self.settings.scholar_source_gate_enabled else "low"

        reason = ""
        if label == "reject":
            reason = (
                "low combined source relevance"
                f" (embedding={embedding_score:.3f}, lexical={lexical_score:.3f}"
                + (f", reranker={reranker_score:.3f}" if reranker_score is not None else "")
                + ")"
            )
        return {
            "score": round(float(score), 4),
            "label": label,
            "reason": reason,
            "reranker_score": reranker_score,
        }

    def _topic_rejection_reason(
        self,
        query: str,
        paper: dict,
        *,
        target_topic: str | None = None,
    ) -> str:
        title = canonical_paper_title(paper)
        title_lower = title.lower()
        if len(re.findall(r"[a-zA-Z]", title)) < 4:
            return "non-informative paper title metadata"
        if title_lower.startswith((
            "proceedings of the",
            "findings of the association",
            "published as a conference paper",
        )):
            return "conference proceedings page rather than a paper title"

        gate_query = self._query_with_topic(query, target_topic)
        terms = set(query_terms(gate_query))
        primary = self._primary_text(paper).lower()
        primary_terms = {normalize_term(term) for term in re.findall(r"[a-z0-9]{2,}", primary)}
        primary_has_kg = self._has_kg_signal(primary, primary_terms)
        primary_has_exact_kg = self._has_exact_kg_signal(primary, primary_terms)
        primary_has_rag = self._has_rag_signal(primary, primary_terms)
        primary_has_dynamic = self._has_dynamic_signal(primary, primary_terms)
        query_needs_kg = {"knowledge", "graph"}.issubset(terms) or "kg" in terms or "graph_rag" in terms
        query_needs_rag = (
            "rag" in terms
            or "graph_rag" in terms
            or {"retrieval", "augmented", "generation"}.issubset(terms)
        )
        dynamic_query_terms = {"dynamic", "temporal", "update", "continual", "evolving", "event", "stream"}
        query_lower = gate_query.lower()
        query_needs_dynamic_kg = query_needs_kg and (
            bool(terms & dynamic_query_terms)
            or "time-evolving" in query_lower
            or "time evolving" in query_lower
        )
        family = self._topic_family(gate_query)
        if family in {
            "causal_reasoning_llm",
            "counterfactual_inference",
            "multi_hop_graph_reasoning",
        }:
            subtype = self._source_subtype_for_topic(query, paper, target_topic=target_topic)
            return subtype.rejection_reason

        if query_needs_dynamic_kg:
            dynamic_title_ok = any(
                phrase in title_lower
                for phrase in (
                    "dynamic knowledge",
                    "dynamic graph",
                    "temporal knowledge",
                    "temporal graph",
                    "time-evolving",
                    "time evolving",
                    "evolving knowledge",
                    "knowledge update",
                    "episodic memory",
                    "dpcl-diff",
                    "tgb",
                )
            )
            if not dynamic_title_ok:
                return "dynamic KG query requires title-level dynamic/temporal graph signal"
            primary_has_temporal_graph = any(
                phrase in primary
                for phrase in (
                    "temporal graph",
                    "dynamic graph",
                    "temporal knowledge graph",
                    "dynamic knowledge graph",
                    "time-evolving graph",
                    "time evolving graph",
                )
            ) or "tgb" in primary_terms
            if not ((primary_has_exact_kg and primary_has_dynamic) or primary_has_temporal_graph):
                return "dynamic KG query requires title/abstract temporal-graph or dynamic-KG signal"
            off_topic_dynamic = (
                "protein directed evolution",
                "knowledge distillation",
                "takeaway recommendation",
                "mobius group",
                "multi-modal",
            )
            if any(phrase in primary for phrase in off_topic_dynamic) and "knowledge graph" not in title_lower:
                return "title/abstract indicates an off-topic dynamic or knowledge transfer paper"

        if query_needs_kg and query_needs_rag:
            rag_title_ok = (
                "rag" in title_lower
                or "retrieval-augmented" in title_lower
                or "retrieval augmented" in title_lower
            )
            kg_title_ok = (
                "graph" in title_lower
                or "knowledge-graph" in title_lower
                or "knowledge graph" in title_lower
                or "graphrag" in title_lower
            )
            if not (rag_title_ok and kg_title_ok):
                return "RAG+KG query requires title-level RAG and graph signal"
            if not primary_has_kg:
                return "RAG+KG query requires title/abstract knowledge graph signal"
            subtype = self._source_subtype_for_topic(query, paper, target_topic=target_topic)
            if subtype.rejection_reason:
                return subtype.rejection_reason
            if not primary_has_rag:
                return "RAG+KG query requires title/abstract retrieval-augmented generation signal"
        subtype = self._source_subtype_for_topic(query, paper, target_topic=target_topic)
        if subtype.rejection_reason:
            return subtype.rejection_reason
        return ""

    def _source_subtype_for_topic(
        self,
        query: str,
        paper: dict,
        *,
        target_topic: str | None = None,
    ) -> SourceSubtype:
        gate_query = self._query_with_topic(query, target_topic)
        family = self._topic_family(gate_query)
        if family == "causal_reasoning_llm":
            return self._causal_reasoning_llm_subtype(paper)
        if family == "counterfactual_inference":
            return self._counterfactual_inference_subtype(paper)
        if family == "multi_hop_graph_reasoning":
            return self._multi_hop_graph_subtype(paper)

        terms = set(query_terms(gate_query))
        query_needs_kg = {"knowledge", "graph"}.issubset(terms) or "kg" in terms or "graph_rag" in terms
        query_needs_rag = (
            "rag" in terms
            or "graph_rag" in terms
            or {"retrieval", "augmented", "generation"}.issubset(terms)
        )
        if not (query_needs_kg and query_needs_rag):
            return SourceSubtype("unclassified", "")

        title = canonical_paper_title(paper)
        title_lower = title.lower()
        abstract_lower = str(paper.get("abstract") or "").lower()
        primary = f"{title_lower}\n{abstract_lower}"
        primary_compact = re.sub(r"\s+", " ", primary)
        title_compact = re.sub(r"\s+", " ", title_lower)
        strong_core_markers = (
            "graphrag",
            "graph-rag",
            "graph rag",
            "kg-rag",
            "kg rag",
            "knowledge graph enhanced rag",
            "knowledge graph-enhanced rag",
            "knowledge-graph enhanced rag",
            "knowledge-graph-enhanced rag",
            "knowledge graph based retrieval-augmented generation",
            "knowledge-graph based retrieval-augmented generation",
            "knowledge graph-based retrieval-augmented generation",
            "knowledge graph retrieval-augmented generation",
            "graph retrieval-augmented generation",
            "graph retrieval augmented generation",
            "retrieval-augmented generation based on knowledge graph",
            "retrieval-augmented generation based on knowledge graphs",
            "retrieval augmented generation based on knowledge graph",
            "retrieval augmented generation based on knowledge graphs",
            "retrieval-augmented generation on knowledge graph",
            "retrieval-augmented generation on knowledge graphs",
            "retrieval augmented generation on knowledge graph",
            "retrieval augmented generation on knowledge graphs",
            "retrieval-augmented generation with knowledge graph",
            "retrieval-augmented generation with knowledge graphs",
            "retrieval augmented generation with knowledge graph",
            "retrieval augmented generation with knowledge graphs",
        )
        title_graph_signal = (
            "knowledge-graph" in title_compact
            or "knowledge graph" in title_compact
            or "graphrag" in title_compact
            or "graph retrieval-augmented" in title_compact
            or "graph retrieval augmented" in title_compact
        )
        title_has_rag_acronym = "rag" in title_compact
        has_strong_core_marker = (
            any(marker in primary_compact for marker in strong_core_markers)
            or (title_has_rag_acronym and title_graph_signal)
        )
        title_has_strong_core_marker = (
            any(marker in title_compact for marker in strong_core_markers)
            or (title_has_rag_acronym and title_graph_signal)
        )

        kg_construction_markers = (
            "knowledge graph construction",
            "knowledge graph extraction",
            "knowledge graph synthesis",
            "knowledge graph generation",
            "knowledge graphs from plain text",
            "extracting knowledge graphs",
            "constructing knowledge graphs",
            "kg construction",
            "kgc",
            "mkgc",
        )
        title_has_kg_construction = any(
            marker in title_compact for marker in kg_construction_markers
        )
        primary_has_kg_construction = any(
            marker in primary_compact for marker in kg_construction_markers
        )
        if (
            (title_has_kg_construction and not title_has_strong_core_marker)
            or (primary_has_kg_construction and not has_strong_core_marker)
        ):
            return SourceSubtype(
                "kg_construction",
                "paper is primarily about constructing or extracting knowledge graphs, not KG-enhanced RAG",
                "RAG+KG query excludes pure knowledge-graph construction/extraction papers",
            )

        kgqa_or_reasoning_markers = (
            "knowledge graph question answering",
            "graph question answering",
            "kgqa",
            "question answering",
            "think-on-graph",
            "think on graph",
            "reasoning on knowledge graph",
            "reasoning over knowledge graph",
            "reasoning of large language model on knowledge graph",
            "faithful reasoning",
            "knowledge graph based prompting",
            "knowledge graph-based prompting",
            "knowledge graph prompting",
            "chat with their graph",
        )
        if any(marker in primary_compact for marker in kgqa_or_reasoning_markers):
            return SourceSubtype(
                "kgqa_or_graph_reasoning",
                "paper centers KGQA, graph reasoning, or KG prompting rather than a KG-RAG method",
            )

        application_markers = (
            "recommendation",
            "recommender",
            "medical",
            "clinical",
            "healthcare",
            "robot",
            "robotic",
            "planning",
            "low-resourced",
            "low resourced",
            "multilingual",
        )
        if any(marker in primary_compact for marker in application_markers):
            return SourceSubtype(
                "application_case",
                "paper studies KG-RAG in a specific application domain; use as supporting evidence",
            )

        benchmark_markers = (
            "benchmark",
            "dataset",
            "comprehensive analysis",
            "when to use graphs",
        )
        if any(marker in title_compact for marker in benchmark_markers):
            return SourceSubtype(
                "benchmark_or_analysis",
                "paper primarily evaluates or analyzes KG-RAG behavior",
            )

        if has_strong_core_marker:
            return SourceSubtype(
                "core_kg_rag_method",
                "title/abstract explicitly frames a KG/GraphRAG retrieval-augmented generation method",
            )

        return SourceSubtype(
            "rag_kg_adjacent",
            "paper has RAG and knowledge graph signals but lacks a strong KG-RAG method marker",
        )

    def _topic_family(self, query: str) -> str:
        lower = query.lower()
        terms = set(query_terms(query))
        has_llm = (
            "llm" in terms
            or "large language model" in lower
            or "large language models" in lower
            or "language model" in lower
            or "language models" in lower
        )
        if "counterfactual inference" in lower:
            return "counterfactual_inference"
        if has_llm and ("causal reasoning" in lower or {"causal", "reasoning"}.issubset(terms)):
            return "causal_reasoning_llm"
        if {"counterfactual", "inference"}.issubset(terms):
            return "counterfactual_inference"
        if (
            ("multi-hop" in lower or "multi hop" in lower or "multihop" in lower or {"multi", "hop"}.issubset(terms))
            and ("graph" in terms or "knowledge graph" in lower or "knowledge graphs" in lower)
        ):
            return "multi_hop_graph_reasoning"
        return ""

    def _paper_primary_compact(self, paper: dict) -> tuple[str, str]:
        title = canonical_paper_title(paper)
        abstract = str(paper.get("abstract") or "")[:2400]
        focused = str(paper.get("focused_text") or "")[:1800]
        primary = f"{title}\n{abstract}\n{focused}"
        return compact_topic_text(title), compact_topic_text(primary)

    def _causal_reasoning_llm_subtype(self, paper: dict) -> SourceSubtype:
        title, primary = self._paper_primary_compact(paper)
        retrieval_or_qa_drift = any(
            marker in title or marker in primary
            for marker in (
                "rag",
                "retrieval-augmented generation",
                "retrieval augmented generation",
                "graph-based rag",
                "graph based rag",
                "graphrag",
                "knowledge graph",
                "tableqa",
                "question answering",
            )
        )
        explicit_llm_causal_eval = any(
            marker in primary
            for marker in (
                "causal reasoning capability",
                "causal reasoning capabilities",
                "causal reasoning in large language models",
                "causal reasoning of large language models",
                "causal abilities in large language models",
                "large language models infer causation",
                "benchmarking llms against statistical pitfalls in causal inference",
                "evaluate large language models on causal",
                "evaluates large language models on causal",
                "evaluating explicit causal reasoning in large language models",
                "counterfactual reasoning in large language models",
                "llms for counterfactual reasoning",
                "language models for counterfactual reasoning",
                "causal reasoning tasks",
                "llm causal reasoning",
                "llms' causal reasoning",
                "llms causal reasoning",
            )
        )
        if retrieval_or_qa_drift and not explicit_llm_causal_eval:
            return SourceSubtype(
                "rag_or_retrieval_adjacent",
                "paper uses causal language inside RAG/retrieval/QA rather than studying LLM causal reasoning ability",
                "Causal reasoning with LLMs query excludes RAG/retrieval/QA papers unless they evaluate LLM causal reasoning",
            )
        causal_application_drift = any(
            marker in primary
            for marker in (
                "causal discovery",
                "uncovering cause-and-effect mechanisms",
                "multimodal causality",
                "attention causality",
                "preference learning",
                "ai alignment",
                "reward modelling",
                "reward modeling",
                "causality-driven robust optimization",
                "boosting resilience",
                "modality prior-induced hallucinations",
                "audio llm",
                "audio llms",
                "reasoning process rewards",
                "process rewards",
            )
        )
        if causal_application_drift and not explicit_llm_causal_eval:
            return SourceSubtype(
                "causal_application_adjacent",
                "paper applies causal framing to discovery, alignment, robustness, modality, or process-reward settings rather than evaluating LLM causal reasoning ability",
                "Causal reasoning with LLMs query excludes causal application papers unless they directly evaluate LLM causal/counterfactual reasoning",
            )
        generic_uncertainty_reasoning = any(
            marker in primary
            for marker in (
                "uncertain text",
                "probabilistic reasoning",
                "bayesian linguistic inference",
                "probabilistic logical programming",
            )
        )
        if generic_uncertainty_reasoning and "causal" not in title and "counterfactual" not in title:
            return SourceSubtype(
                "llm_reasoning_adjacent",
                "paper primarily studies probabilistic or uncertainty reasoning rather than LLM causal reasoning",
                "Causal reasoning with LLMs query requires causal/counterfactual reasoning as the primary task",
            )
        has_llm = any(
            marker in primary
            for marker in (
                "large language model",
                "large language models",
                "language model",
                "language models",
                "llm",
                "llms",
                "chain-of-thought",
                "chain of thought",
            )
        )
        has_causal_reasoning = any(
            marker in primary
            for marker in (
                "causal reasoning",
                "counterfactual reasoning",
                "causal inference",
                "causal abilities",
                "causal ability",
                "causal question",
                "causal benchmark",
                "statistical pitfalls in causal inference",
            )
        )
        autoregressive_only = (
            "causal language modeling" in primary
            or "causal llm inference" in primary
            or "causal attention" in primary
        ) and not has_causal_reasoning
        if not has_llm or not has_causal_reasoning or autoregressive_only:
            return SourceSubtype(
                "llm_reasoning_adjacent",
                "paper lacks explicit causal/counterfactual reasoning evidence for LLMs",
                "Causal reasoning with LLMs query requires causal/counterfactual reasoning and LLM signal",
            )
        if "benchmark" in title or "evaluat" in primary or "pitfall" in primary:
            return SourceSubtype(
                "causal_reasoning_benchmark",
                "paper evaluates causal/counterfactual reasoning ability in LLMs",
            )
        if "counterfactual" in primary:
            return SourceSubtype(
                "llm_counterfactual_reasoning",
                "paper studies counterfactual reasoning with language models",
            )
        return SourceSubtype(
            "core_llm_causal_reasoning",
            "paper directly studies causal reasoning with language models",
        )

    def _counterfactual_inference_subtype(self, paper: dict) -> SourceSubtype:
        title, primary = self._paper_primary_compact(paper)
        has_counterfactual = "counterfactual" in primary
        has_treatment = any(
            marker in primary
            for marker in (
                "treatment effect",
                "treatment effects",
                "cate",
                "heterogeneous treatment",
                "individualized treatment",
                "potential outcome",
                "potential outcomes",
            )
        )
        has_causal_inference = any(
            marker in primary
            for marker in (
                "causal inference",
                "causal effect",
                "causal effects",
                "structural causal model",
                "structural causal models",
                "unconfoundedness",
            )
        )
        title_terms = re.findall(r"[a-z]{3,}", title)
        title_has_counterfactual_signal = any(
            marker in title
            for marker in (
                "counterfactual",
                "treatment effect",
                "causal effect",
                "potential outcome",
                "structural causal",
            )
        )
        has_application_drift = any(
            marker in primary
            for marker in (
                "recommender",
                "news recommendation",
                "item recommendation",
                "implicit feedback",
                "collaborative filtering",
                "user-item",
                "user item",
                "click-through",
                "click through",
                "popularity-aware",
                "popularity aware",
                "popularity bias",
                "debiased modeling",
            )
        ) or "recommendation" in title
        has_policy_eval_drift = any(
            marker in primary
            for marker in (
                "off-policy",
                "off policy",
                "policy evaluation",
                "policy mean embedding",
                "policy mean embeddings",
            )
        )
        has_vision_drift = any(
            marker in primary
            for marker in (
                "vision transformer",
                "vision transformers",
                "image classification",
                "long-tailed",
                "long tailed",
                "object detection",
                "semantic segmentation",
            )
        )
        has_causal_discovery_drift = any(
            marker in primary
            for marker in (
                "causal discovery",
                "causal structure",
                "causal graph discovery",
                "interventional distribution",
                "interventional distributions",
            )
        )
        if has_application_drift:
            return SourceSubtype(
                "application_or_recommender_adjacent",
                "paper applies counterfactual or causal language to recommendation/debiasing rather than counterfactual inference methods",
                "Counterfactual inference query excludes recommender/debiasing application papers",
            )
        if has_policy_eval_drift and not has_treatment and "counterfactual fairness" not in primary:
            return SourceSubtype(
                "policy_evaluation_adjacent",
                "paper is primarily off-policy or policy-evaluation work rather than counterfactual inference",
                "Counterfactual inference query excludes off-policy evaluation papers",
            )
        if has_vision_drift and not (has_counterfactual or has_treatment):
            return SourceSubtype(
                "vision_application_adjacent",
                "paper is a vision-domain causal intervention application rather than counterfactual inference",
                "Counterfactual inference query excludes generic vision causal-intervention papers",
            )
        if has_causal_discovery_drift and not (has_counterfactual or has_treatment):
            return SourceSubtype(
                "causal_discovery_adjacent",
                "paper studies causal discovery or interventional distributions without explicit counterfactual/treatment-effect inference",
                "Counterfactual inference query excludes causal-discovery-only papers",
            )
        if (
            len(title_terms) <= 2
            and not title_has_counterfactual_signal
            and not (has_counterfactual or has_treatment or has_causal_inference)
        ):
            return SourceSubtype(
                "probabilistic_or_ml_adjacent",
                "paper title metadata is too weak for topic-specific counterfactual screening",
                "non-informative paper title metadata",
            )
        if not (has_counterfactual or has_treatment or has_causal_inference):
            return SourceSubtype(
                "probabilistic_or_ml_adjacent",
                "paper lacks counterfactual, treatment-effect, or causal-inference signal",
                "Counterfactual inference query requires counterfactual/treatment-effect/causal-inference signal",
            )
        if has_counterfactual and ("explanation" in primary or "fairness" in primary):
            return SourceSubtype(
                "counterfactual_explanation_or_fairness",
                "paper studies counterfactual explanations or counterfactual fairness",
            )
        if has_treatment:
            return SourceSubtype(
                "treatment_effect_estimation",
                "paper focuses treatment-effect or potential-outcome estimation",
            )
        if has_counterfactual:
            return SourceSubtype(
                "core_counterfactual_inference",
                "paper explicitly studies counterfactual inference or estimation",
            )
        return SourceSubtype(
            "causal_inference_adjacent",
            "paper is causal-inference adjacent but not explicitly counterfactual",
        )

    def _multi_hop_graph_subtype(self, paper: dict) -> SourceSubtype:
        title, primary = self._paper_primary_compact(paper)
        has_graph = (
            "graph" in primary
            or "graphs" in primary
            or "knowledge graph" in primary
            or "knowledge graphs" in primary
            or " kg " in f" {primary} "
        )
        has_multihop = (
            "multi-hop" in primary
            or "multi hop" in primary
            or "multihop" in primary
            or "long-range" in primary
            or "long range" in primary
        )
        has_reasoning = any(
            marker in primary
            for marker in (
                "reasoning",
                "question answering",
                "semantic parsing",
                "path reasoning",
                "knowledge graph reasoning",
                "graph question answering",
                "kgqa",
            )
        )
        generic_graph_ml = any(
            marker in primary
            for marker in (
                "node classification",
                "representation learning",
                "message passing",
                "graph neural network",
                "graph neural networks",
                "graph out-of-distribution",
                "subgraph representation",
            )
        )
        if has_graph and generic_graph_ml and not (has_multihop or has_reasoning):
            return SourceSubtype(
                "graph_ml_adjacent",
                "paper is generic graph representation/propagation work rather than multi-hop graph reasoning",
                "Multi-hop graph reasoning query excludes generic graph representation learning",
            )
        if not has_graph or not (has_multihop or has_reasoning):
            return SourceSubtype(
                "graph_or_reasoning_adjacent",
                "paper lacks explicit graph-based multi-hop reasoning signal",
                "Multi-hop graph reasoning query requires graph plus multi-hop/reasoning/QA signal",
            )
        if generic_graph_ml and not has_reasoning:
            return SourceSubtype(
                "graph_ml_adjacent",
                "paper is generic graph representation/propagation work rather than multi-hop graph reasoning",
                "Multi-hop graph reasoning query excludes generic graph representation learning",
            )
        if "benchmark" in title or "dataset" in title:
            return SourceSubtype(
                "graph_reasoning_benchmark",
                "paper benchmarks multi-hop or graph question answering",
            )
        if "question answering" in primary or "semantic parsing" in primary or "kgqa" in primary:
            return SourceSubtype(
                "kgqa_or_graph_reasoning",
                "paper centers graph/KG question answering or semantic parsing",
            )
        title_has_rag = (
            "rag" in title
            or "retrieval-augmented" in title
            or "retrieval augmented" in title
        )
        primary_has_rag_framework = any(
            marker in primary
            for marker in (
                "retrieval-augmented generation",
                "retrieval augmented generation",
                "graph retrieval augmented generation",
                "graph retrieval-augmented generation",
                "graphrag",
                "graph-rag",
                "graph rag",
            )
        )
        if title_has_rag or primary_has_rag_framework:
            return SourceSubtype(
                "graph_retrieval_rag_adjacent",
                "paper connects graph reasoning with retrieval-augmented generation",
            )
        return SourceSubtype(
            "core_multi_hop_graph_reasoning",
            "paper directly studies multi-hop reasoning on graph-structured knowledge",
        )

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

    def _text_search(
        self,
        query: str,
        max_results: int,
        *,
        target_topic: str | None = None,
    ) -> list[SourceCandidate]:
        """Fallback: 基于文本匹配的检索。"""
        terms = query_terms(self._query_with_topic(query, target_topic))
        phrases = query_phrases(terms)
        scored = []
        for idx, paper in enumerate(self.searchable_papers):
            score = self._lexical_score_for_index(terms, phrases, idx)
            scored.append((score, paper))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, paper in scored[:max_results]:
            if score > 0:
                results.append(self._candidate_from_paper(paper, query, score, target_topic=target_topic))
        return results

    def _query_with_topic(self, query: str, target_topic: str | None) -> str:
        topic = (target_topic or "").strip()
        if not topic:
            return query
        if topic.lower() in query.lower():
            return query
        return f"{topic} {query}"

    def _candidate_from_paper(
        self,
        paper: dict,
        query: str,
        score: float,
        *,
        target_topic: str | None = None,
        embedding_score: float | None = None,
        lexical_score: float | None = None,
        reranker_score: float | None = None,
        relevance_score: float | None = None,
        relevance_label: str = "unscored",
        rejection_reason: str = "",
    ) -> SourceCandidate:
        title = canonical_paper_title(paper)
        subtype = self._source_subtype_for_topic(query, paper, target_topic=target_topic)
        return SourceCandidate(
            url=paper.get("pdf_path", ""),
            title=title,
            snippet=paper.get("abstract", "")[:300],
            content=paper.get("focused_text", ""),
            source_type="academic_paper",
            query=query,
            score=min(max(float(score), 0.0), 1.0),
            embedding_score=embedding_score,
            lexical_score=lexical_score,
            reranker_score=reranker_score,
            relevance_score=relevance_score if relevance_score is not None else min(max(float(score), 0.0), 1.0),
            relevance_label=relevance_label,
            rejection_reason=rejection_reason,
            source_subtype=subtype.label,
            source_subtype_reason=subtype.reason,
            source_provider="local_papers",
        )

    def _reranker_text(self, paper: dict) -> str:
        title = canonical_paper_title(paper)
        abstract = str(paper.get("abstract") or "")
        focused = str(paper.get("focused_text") or "")[:1800]
        return "\n".join(part for part in (title, abstract, focused) if part.strip())

    def _primary_text(self, paper: dict) -> str:
        title = canonical_paper_title(paper)
        abstract = str(paper.get("abstract") or "")
        return "\n".join(part for part in (title, abstract) if part.strip())

    def _ensure_lexical_cache(self) -> list[dict[str, object]]:
        if self._lexical_cache is not None:
            return self._lexical_cache
        cache: list[dict[str, object]] = []
        for paper in self.searchable_papers:
            title = canonical_paper_title(paper).lower()
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
        query_needs_dynamic = bool(
            term_set & {"dynamic", "temporal", "update", "continual", "evolving", "event", "stream"}
        )

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

    def _has_exact_kg_signal(self, text: str, text_terms: object) -> bool:
        terms = text_terms if isinstance(text_terms, set) else set()
        return (
            "kg" in terms
            or "tkg" in terms
            or "knowledge graph" in text
            or "knowledge graphs" in text
            or "knowledge-graph" in text
            or "temporal knowledge graph" in text
            or "dynamic knowledge graph" in text
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
        self.index = LocalPaperIndex(str(p), settings=settings)

    @property
    def provider_names(self) -> list[str]:
        return ["local_papers"]

    @property
    def active_provider_names(self) -> list[str]:
        return ["local_papers"]

    async def search(self, query: str, max_results: int = 10) -> list[SourceCandidate]:
        return self.index.search(query, max_results)

    async def search_for_topic(
        self,
        query: str,
        target_topic: str,
        max_results: int = 10,
    ) -> list[SourceCandidate]:
        return self.index.search(query, max_results, target_topic=target_topic)
