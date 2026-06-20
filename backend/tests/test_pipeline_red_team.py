"""Red-team model selection behavior."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from cg.orchestrator import pipeline as pipeline_module
from cg.orchestrator.pipeline import ResearchPipeline, build_paper_pattern_matrix, select_matrix_papers
from cg.schemas.research import Claim, Evidence, ResearchRequest
from cg.settings import Settings


def test_red_team_can_use_gpt55_model_with_gemini_credentials(tmp_path) -> None:
    settings = Settings(
        cg_data_dir=str(tmp_path),
        gemini_api_key="test-key",
        gemini_base_url="https://example.invalid/v1",
        red_team_llm_provider="gpt-5.5",
    )

    pipeline = ResearchPipeline(settings)

    assert pipeline.red_team_llm.provider == "gemini"
    assert pipeline.red_team_llm.model == "gpt-5.5"


def test_red_team_defaults_to_configured_gemini_model(tmp_path) -> None:
    settings = Settings(
        cg_data_dir=str(tmp_path),
        gemini_api_key="test-key",
        gemini_base_url="https://example.invalid/v1",
        gemini_model="gemini-pro-agent",
    )

    pipeline = ResearchPipeline(settings)

    assert pipeline.red_team_llm.provider == "gemini"
    assert pipeline.red_team_llm.model == "gemini-pro-agent"


def test_interim_matrix_uses_real_evidence_papers() -> None:
    request = ResearchRequest(
        project_name="Coverage regression",
        target_topic="RAG with Knowledge Graphs",
        analysis_dimensions=["modular_pipeline_composition"],
    )
    evidence = [
        _evidence("ev_1", "GraphRAG Paper A", "modular_pipeline_composition"),
        _evidence("ev_2", "GraphRAG Paper B", "modular_pipeline_composition"),
    ]

    papers = select_matrix_papers(request, evidence)
    matrix = build_paper_pattern_matrix(evidence, papers, request.analysis_dimensions)

    assert papers == ["GraphRAG Paper A", "GraphRAG Paper B"]
    assert matrix.coverage_by_dimension["modular_pipeline_composition"] > 0


class FailingReviewAgent:
    def __init__(self, ctx):
        self.ctx = ctx

    async def review(self, claims, evidence):
        raise RuntimeError("simulated red-team failure")


@pytest.mark.asyncio
async def test_red_team_fallback_records_trace_and_writes_claims(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pipeline_module, "AnalysisAndReviewAgent", FailingReviewAgent)
    pipeline = ResearchPipeline(
        Settings(
            cg_data_dir=str(tmp_path),
            scholar_paper_index_path=str(tmp_path / "missing_paper_index.json"),
            deepseek_api_key="",
            gemini_api_key="",
            mimo_api_key="",
        )
    )
    request = ResearchRequest(
        project_name="Red-team fallback regression",
        target_topic="Causal Reasoning with LLMs",
        analysis_dimensions=["llm_causal_benchmarking"],
    )
    status = await pipeline.prepare_run(request, owner="test")
    evidence = [_evidence("ev_redteam", "Causal Benchmark Paper", "llm_causal_benchmarking")]
    claims = [
        Claim(
            claim_id="claim_redteam",
            run_id=status.run_id,
            dimension="llm_causal_benchmarking",
            dimension_label="LLM causal benchmarking",
            claim="Causal Benchmark Paper provides a bounded observation.",
            supporting_evidence_ids=["ev_redteam"],
            confidence=0.8,
            risk_level="low",
            reasoning_summary="Supported by one evidence item.",
            verification_status="verified",
            claim_type="single_paper_observation",
            source_paper_count=1,
            evidence_support_level="single_paper",
        )
    ]

    reviewed = await pipeline.red_team(status.run_id, claims, evidence)

    run_dir = pipeline.runs.run_dir(status.run_id)
    assert reviewed
    assert (run_dir / "exports" / "claim_backlog.json").exists()
    assert (run_dir / "claims" / "claim_redteam.json").exists()
    trace_statuses = {
        json.loads(line)["status"]
        for line in (run_dir / "trace" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert "red_team_review_started" in trace_statuses
    assert "red_team_review_failed" in trace_statuses
    assert "red_team_review_fallback" in trace_statuses
    assert "red_team_review_completed" in trace_statuses


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
