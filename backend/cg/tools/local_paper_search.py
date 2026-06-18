"""本地论文检索工具，替代 CompeteInsight 的 Web SearchTool。

使用 sentence-transformers (bge-large-en-v1.5) + numpy 做 embedding 检索。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cg.schemas.research import SourceCandidate
from cg.settings import Settings


class LocalPaperIndex:
    """基于 embedding 的本地论文索引。"""

    def __init__(self, index_path: str, embeddings_path: str | None = None):
        self.papers: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self._model = None

        if Path(index_path).exists():
            with open(index_path, "r") as f:
                self.papers = json.load(f)

        # 加载 embeddings
        if embeddings_path is None:
            embeddings_path = str(Path(index_path).parent / "embeddings.npy")
        if Path(embeddings_path).exists():
            self.embeddings = np.load(embeddings_path)

    def _get_model(self):
        """延迟加载 sentence-transformers 模型。"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        return self._model

    def search(self, query: str, max_results: int = 10) -> list[SourceCandidate]:
        """基于 embedding cosine similarity 检索。"""
        if not self.papers:
            return []

        # 如果有 embedding，用 cosine similarity
        if self.embeddings is not None and len(self.embeddings) == len(self.papers):
            model = self._get_model()
            query_emb = model.encode([query], normalize_embeddings=True)
            # cosine similarity via dot product (embeddings already normalized)
            scores = np.dot(self.embeddings, query_emb.T).flatten()
            top_indices = np.argsort(scores)[::-1][:max_results]

            results = []
            for idx in top_indices:
                paper = self.papers[idx]
                score = float(scores[idx])
                results.append(SourceCandidate(
                    url=paper.get("pdf_path", ""),
                    title=paper.get("title", ""),
                    snippet=paper.get("abstract", "")[:300],
                    content=paper.get("focused_text", ""),
                    source_type="academic_paper",
                    query=query,
                    score=min(max(score, 0.0), 1.0),
                ))
            return results

        # fallback: 简单文本匹配
        return self._text_search(query, max_results)

    def _text_search(self, query: str, max_results: int) -> list[SourceCandidate]:
        """Fallback: 基于文本匹配的检索。"""
        query_lower = query.lower()
        scored = []
        for paper in self.papers:
            title = paper.get("title", "").lower()
            abstract = paper.get("abstract", "").lower()
            score = 0.0
            for w in query_lower.split():
                if w in title:
                    score += 0.6
                if w in abstract:
                    score += 0.4
            scored.append((score, paper))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, paper in scored[:max_results]:
            if score > 0:
                results.append(SourceCandidate(
                    url=paper.get("pdf_path", ""),
                    title=paper.get("title", ""),
                    snippet=paper.get("abstract", "")[:300],
                    content=paper.get("focused_text", ""),
                    source_type="academic_paper",
                    query=query,
                    score=min(score, 1.0),
                ))
        return results


class LocalPaperSearchTool:
    """与 CompeteInsight SearchTool 接口兼容的本地论文检索。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        index_path = getattr(settings, "scholar_paper_index_path", "data/paper_index.json")
        # 如果是相对路径，相对于 data_dir
        p = Path(index_path)
        if not p.is_absolute():
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
