"""Gap assessment should use real evidence papers and verified claims."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from cg.agents.research_agents import AnalysisAndReviewAgent
from cg.agents.research_agents import normalize_dimension_key
from cg.agents.runtime import AgentContext
from cg.llm import LLMClient
from cg.schemas.research import Claim, Evidence, ResearchRequest
from cg.settings import Settings


@pytest.mark.asyncio
async def test_gap_assessment_does_not_treat_target_topic_as_paper(tmp_path: Path) -> None:
    request = ResearchRequest(
        project_name="Gap regression",
        target_topic="RAG with Knowledge Graphs",
        analysis_dimensions=["modular_pipeline_composition", "cross_domain_synthesis"],
    )
    evidence = [
        _evidence("ev_a", "GraphRAG Paper A", "modular_pipeline_composition"),
        _evidence("ev_b", "GraphRAG Paper B", "modular_pipeline_composition"),
    ]
    claims = [
        _claim(
            "claim_a",
            "modular_pipeline_composition",
            ["ev_a", "ev_b"],
            verification_status="verified",
        )
    ]

    feedback = await _agent(tmp_path).assess_gaps(
        request,
        evidence,
        claims,
        loop_round=1,
        coverage_score=0.5,
    )

    assert feedback.needs_more_research
    assert [gap.dimension for gap in feedback.gaps] == ["cross_domain_synthesis"]


@pytest.mark.asyncio
async def test_gap_assessment_flags_dimensions_without_verified_claims(tmp_path: Path) -> None:
    request = ResearchRequest(
        project_name="Verified gap",
        target_topic="Scientific Reasoning with LLMs",
        analysis_dimensions=["formal_experimental_tightening"],
    )
    evidence = [
        _evidence("ev_a", "Reasoning Paper A", "formal_experimental_tightening"),
        _evidence("ev_b", "Reasoning Paper B", "formal_experimental_tightening"),
    ]
    claims = [
        _claim(
            "claim_a",
            "formal_experimental_tightening",
            ["ev_a", "ev_b"],
            verification_status="challenged",
        )
    ]

    feedback = await _agent(tmp_path).assess_gaps(
        request,
        evidence,
        claims,
        loop_round=1,
        coverage_score=0.75,
    )

    assert feedback.needs_more_research
    assert len(feedback.gaps) == 1
    assert feedback.gaps[0].dimension == "formal_experimental_tightening"
    assert feedback.gaps[0].priority == "high"
    assert "verified claim" in feedback.gaps[0].reason


def test_normalize_dimension_key_accepts_labels_and_gap_query_hints() -> None:
    assert normalize_dimension_key("表征转换") == "representation_shift"
    assert normalize_dimension_key("inference_time_control") == "inference_time_control"
    assert (
        normalize_dimension_key(
            "N/A",
            "Mathematical Reasoning natural language to symbolic equations",
        )
        == "representation_shift"
    )
    assert (
        normalize_dimension_key(
            "N/A",
            "Mathematical Reasoning modular pipeline planner solver verifier",
        )
        == "modular_pipeline_composition"
    )


def _agent(tmp_path: Path) -> AnalysisAndReviewAgent:
    settings = Settings(
        cg_data_dir=str(tmp_path),
        cg_llm_provider="deepseek",
        deepseek_api_key="",
        mimo_api_key="",
        gemini_api_key="",
        qwen_api_key="",
        ark_api_key="",
    )
    ctx = AgentContext(
        run_id="run_test",
        run_dir=tmp_path,
        settings=settings,
        llm=LLMClient(settings),
        trace=record_trace,
    )
    return AnalysisAndReviewAgent(ctx)


def _evidence(evidence_id: str, paper: str, dimension: str) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        run_id="run_test",
        url=f"/papers/{evidence_id}.pdf",
        title=paper,
        content="Evidence content",
        fetched_at=datetime.now(timezone.utc),
        source_type="academic_paper",
        dimension=dimension,
        dimension_label=dimension,
        paper=paper,
        fact=f"{paper} contains a relevant mechanism.",
        quote="Relevant mechanism.",
        source_title=paper,
        source_url=f"/papers/{evidence_id}.pdf",
        confidence=0.8,
    )


def _claim(
    claim_id: str,
    dimension: str,
    evidence_ids: list[str],
    *,
    verification_status: str,
) -> Claim:
    return Claim(
        claim_id=claim_id,
        run_id="run_test",
        dimension=dimension,
        dimension_label=dimension,
        claim="Evidence supports this dimension.",
        supporting_evidence_ids=evidence_ids,
        confidence=0.8,
        risk_level="low" if verification_status == "verified" else "medium",
        reasoning_summary="Synthesized from evidence.",
        verification_status=verification_status,  # type: ignore[arg-type]
    )


async def record_trace(
    node: str,
    phase: str,
    status: str,
    message: str,
    payload: dict[str, Any] | None = None,
) -> None:
    _ = (node, phase, status, message, payload)
