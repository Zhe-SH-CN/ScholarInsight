"""Source discovery continuation behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cg.agents.research_agents import (
    SUPPLEMENTAL_APPLICATION_REJECTION_REASON,
    SourceResearchAgent,
    should_reject_supplemental_application_source,
)
from cg.agents.runtime import AgentContext
from cg.llm import LLMClient
from cg.schemas.research import ResearchPlan, ResearchRequest, SourceCandidate, SourceTask
from cg.settings import Settings


class FakeSearch:
    def __init__(self, responses: dict[str, list[SourceCandidate]]):
        self.responses = responses
        self.queries: list[str] = []

    async def search(self, query: str, max_results: int = 3) -> list[SourceCandidate]:
        self.queries.append(query)
        return [
            item.model_copy()
            for item in self.responses.get(query, [])[:max_results]
        ]


@pytest.mark.asyncio
async def test_discover_runs_gap_search_when_first_pass_has_coverage_gaps(tmp_path: Path) -> None:
    request = ResearchRequest(
        project_name="Pattern gap search",
        target_topic="Knowledge Graph Reasoning",
        seed_papers=["GraphRAG"],
        analysis_dimensions=["gap_driven_reframing", "cross_domain_synthesis"],
        max_sources=4,
        max_sources_per_query=2,
        max_search_rounds=2,
    )
    plan = ResearchPlan(
        research_goal=request.research_goal,
        papers=request.seed_papers,
        dimensions=request.analysis_dimensions,
        queries=[],
        source_tasks=[
            SourceTask(
                task_id="task_01",
                entity="Knowledge Graph Reasoning",
                dimension="gap_driven_reframing",
                intent="official",
                query="Knowledge Graph Reasoning limitations bottlenecks survey",
                expected_source_types=["academic_paper"],
            ),
        ],
    )
    search = FakeSearch(
        {
            "Knowledge Graph Reasoning limitations bottlenecks survey": [
                SourceCandidate(
                    url="/papers/graphrag.pdf",
                    title="Knowledge Graph Reasoning limitations survey",
                    snippet="Knowledge Graph Reasoning bottlenecks and problem reframing.",
                    source_type="academic_paper",
                    score=0.7,
                )
            ],
        }
    )

    candidates = await make_agent(tmp_path).discover(request, plan, search)  # type: ignore[arg-type]

    assert search.queries[0] == "Knowledge Graph Reasoning limitations bottlenecks survey"
    assert "Knowledge Graph Reasoning cross-domain synthesis knowledge graph multimodal integration" in search.queries
    assert {candidate.url for candidate in candidates} == {"/papers/graphrag.pdf"}
    assert all(candidate.query for candidate in candidates)


@pytest.mark.asyncio
async def test_discover_uses_second_pass_to_break_domain_concentration(tmp_path: Path) -> None:
    request = ResearchRequest(
        project_name="Filled source budget",
        target_topic="Retrieval-Augmented Generation",
        seed_papers=["Self-RAG"],
        analysis_dimensions=["gap_driven_reframing", "data_evaluation_engineering"],
        max_sources=3,
        max_sources_per_query=3,
        max_search_rounds=2,
    )
    plan = ResearchPlan(
        research_goal=request.research_goal,
        papers=request.seed_papers,
        dimensions=request.analysis_dimensions,
        queries=[],
        source_tasks=[
            SourceTask(
                task_id="task_01",
                entity="Retrieval-Augmented Generation",
                dimension="gap_driven_reframing",
                intent="official",
                query="Retrieval-Augmented Generation limitations bottlenecks survey",
                expected_source_types=["academic_paper"],
            )
        ],
    )
    search = FakeSearch(
        {
            "Retrieval-Augmented Generation limitations bottlenecks survey": [
                SourceCandidate(
                    url="/papers/rag-1.pdf",
                    title="Retrieval-Augmented Generation limitations",
                    snippet="RAG bottlenecks and problem reframing.",
                    source_type="academic_paper",
                    score=0.83,
                ),
                SourceCandidate(
                    url="/papers/rag-2.pdf",
                    title="Retrieval-Augmented Generation evaluation benchmark",
                    snippet="RAG benchmark dataset evaluation metrics.",
                    source_type="academic_paper",
                    score=0.8,
                ),
                SourceCandidate(
                    url="/papers/rag-3.pdf",
                    title="Retrieval-Augmented Generation recent advances",
                    snippet="Recent advances in RAG systems.",
                    source_type="academic_paper",
                    score=0.78,
                ),
            ],
        }
    )

    candidates = await make_agent(tmp_path).discover(request, plan, search)  # type: ignore[arg-type]

    assert "Retrieval-Augmented Generation benchmark dataset evaluation metric annotation" not in search.queries
    assert len(candidates) == 3
    assert {candidate.url for candidate in candidates} == {
        "/papers/rag-1.pdf",
        "/papers/rag-2.pdf",
        "/papers/rag-3.pdf",
    }


def test_supplemental_application_source_filter_only_for_generic_topics() -> None:
    candidate = SourceCandidate(
        url="/papers/medical-kg-rag.pdf",
        title="Medical KG-RAG",
        source_subtype="application_case",
    )

    assert should_reject_supplemental_application_source(
        candidate,
        target_topic="RAG with Knowledge Graphs",
        loop_round=2,
    )
    assert not should_reject_supplemental_application_source(
        candidate,
        target_topic="Medical RAG with Knowledge Graphs",
        loop_round=2,
    )
    assert not should_reject_supplemental_application_source(
        candidate,
        target_topic="RAG with Knowledge Graphs",
        loop_round=1,
    )
    assert SUPPLEMENTAL_APPLICATION_REJECTION_REASON


def make_agent(tmp_path: Path) -> SourceResearchAgent:
    settings = Settings(cg_data_dir=str(tmp_path))
    ctx = AgentContext(
        run_id="run_test",
        run_dir=tmp_path,
        settings=settings,
        llm=LLMClient(settings),
        trace=record_trace,
    )
    return SourceResearchAgent(ctx)


async def record_trace(
    node: str,
    phase: str,
    status: str,
    message: str,
    payload: dict[str, Any] | None = None,
) -> None:
    _ = (node, phase, status, message, payload)
