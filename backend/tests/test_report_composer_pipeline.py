"""Report composer pipeline fallbacks and observability."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cg.orchestrator import pipeline as pipeline_module
from cg.schemas.research import (
    Claim,
    Evidence,
    MatrixCell,
    ObservabilitySnapshot,
    PaperPatternMatrix,
    QualityGate,
    ResearchRequest,
    RunMetrics,
)
from cg.settings import Settings


class FailingReportComposer:
    def __init__(self, ctx):
        self.ctx = ctx

    async def write(self, *args, **kwargs) -> str:
        raise RuntimeError("simulated report LLM failure")

    async def write_executive_summary(self, *args, **kwargs) -> str:
        return "# Summary\n\nLLM summary succeeded.\n"

    async def write_methodology(self, *args, **kwargs) -> str:
        return "# Methodology\n\nLLM methodology succeeded.\n"


class IncrementalReportComposer:
    def __init__(self, ctx):
        self.ctx = ctx

    async def write(self, *args, **kwargs) -> str:
        return "# Report\n\nLLM report succeeded.\n"

    async def write_executive_summary(self, *args, **kwargs) -> str:
        return "# Summary\n\nLLM summary succeeded.\n"

    async def write_methodology(self, *args, **kwargs) -> str:
        assert (self.ctx.run_dir / "reports" / "report.md").exists()
        assert (self.ctx.run_dir / "reports" / "executive_summary.md").exists()
        return "# Methodology\n\nLLM methodology succeeded.\n"


@pytest.mark.asyncio
async def test_write_report_falls_back_and_records_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_module, "ReportComposerAgent", FailingReportComposer)
    pipeline = pipeline_module.ResearchPipeline(
        Settings(
            cg_data_dir=str(tmp_path),
            scholar_paper_index_path=str(tmp_path / "missing_paper_index.json"),
            deepseek_api_key="",
            gemini_api_key="",
            mimo_api_key="",
        )
    )
    request = ResearchRequest(
        project_name="Report fallback regression",
        target_topic="Counterfactual Inference",
        analysis_dimensions=["core_counterfactual_inference"],
    )
    status = await pipeline.prepare_run(request, owner="test")
    evidence = [_evidence()]
    claims = [_claim()]
    metrics = RunMetrics(
        sources_fetched=1,
        evidence_count=1,
        claim_count=1,
        verified_claim_count=1,
    )
    matrix = _matrix()
    observability = _observability()

    fallbacks = await pipeline.write_report(
        status.run_id,
        request,
        evidence,
        claims,
        metrics,
        matrix,
        [],
        observability,
    )

    run_dir = pipeline.runs.run_dir(status.run_id)
    assert fallbacks[0]["section"] == "report"
    assert "simulated report LLM failure" in fallbacks[0]["error"]
    assert (run_dir / "reports" / "report.md").read_text(encoding="utf-8").startswith(
        "# Report fallback regression"
    )
    assert "LLM summary succeeded" in (run_dir / "reports" / "executive_summary.md").read_text(
        encoding="utf-8"
    )
    assert "LLM methodology succeeded" in (run_dir / "reports" / "methodology.md").read_text(
        encoding="utf-8"
    )
    fallback_rows = json.loads((run_dir / "reports" / "report_fallbacks.json").read_text(encoding="utf-8"))
    assert fallback_rows == fallbacks

    trace_statuses = {
        json.loads(line)["status"]
        for line in (run_dir / "trace" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert "report_step_failed" in trace_statuses
    assert "report_step_fallback" in trace_statuses
    assert "reports_written" in trace_statuses


@pytest.mark.asyncio
async def test_write_report_persists_sections_incrementally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_module, "ReportComposerAgent", IncrementalReportComposer)
    pipeline = pipeline_module.ResearchPipeline(
        Settings(
            cg_data_dir=str(tmp_path),
            scholar_paper_index_path=str(tmp_path / "missing_paper_index.json"),
            deepseek_api_key="",
            gemini_api_key="",
            mimo_api_key="",
        )
    )
    request = ResearchRequest(
        project_name="Report incremental persistence regression",
        target_topic="Counterfactual Inference",
        analysis_dimensions=["core_counterfactual_inference"],
    )
    status = await pipeline.prepare_run(request, owner="test")
    evidence = [_evidence()]
    claims = [_claim()]
    metrics = RunMetrics(
        sources_fetched=1,
        evidence_count=1,
        claim_count=1,
        verified_claim_count=1,
    )

    fallbacks = await pipeline.write_report(
        status.run_id,
        request,
        evidence,
        claims,
        metrics,
        _matrix(),
        [],
        _observability(),
    )

    run_dir = pipeline.runs.run_dir(status.run_id)
    assert fallbacks == []
    assert "LLM report succeeded" in (run_dir / "reports" / "report.md").read_text(
        encoding="utf-8"
    )
    assert "LLM summary succeeded" in (run_dir / "reports" / "executive_summary.md").read_text(
        encoding="utf-8"
    )
    assert "LLM methodology succeeded" in (run_dir / "reports" / "methodology.md").read_text(
        encoding="utf-8"
    )


def _evidence() -> Evidence:
    now = datetime.now(timezone.utc)
    return Evidence(
        evidence_id="ev_report",
        run_id="run_test",
        url="/papers/counterfactual.pdf",
        title="Counterfactual inference paper",
        content="Counterfactual inference evidence.",
        fetched_at=now,
        source_type="academic_paper",
        dimension="core_counterfactual_inference",
        dimension_label="Core Counterfactual Inference",
        paper="Counterfactual Paper",
        fact="The paper studies a counterfactual inference mechanism.",
        quote="counterfactual inference mechanism",
        source_title="Counterfactual Paper",
        source_url="/papers/counterfactual.pdf",
        confidence=0.8,
    )


def _claim() -> Claim:
    return Claim(
        claim_id="claim_report",
        run_id="run_test",
        dimension="core_counterfactual_inference",
        dimension_label="Core Counterfactual Inference",
        claim="Counterfactual Paper provides a bounded single-paper observation.",
        supporting_evidence_ids=["ev_report"],
        confidence=0.8,
        risk_level="low",
        reasoning_summary="Supported by one paper.",
        verification_status="verified",
        claim_type="single_paper_observation",
        source_paper_count=1,
        evidence_support_level="single_paper",
    )


def _matrix() -> PaperPatternMatrix:
    return PaperPatternMatrix(
        generated_at=datetime.now(timezone.utc),
        papers=["Counterfactual Paper"],
        dimensions=["core_counterfactual_inference"],
        dimension_labels={"core_counterfactual_inference": "Core Counterfactual Inference"},
        cells=[
            MatrixCell(
                paper="Counterfactual Paper",
                dimension="core_counterfactual_inference",
                dimension_label="Core Counterfactual Inference",
                summary="The paper provides bounded evidence.",
                evidence_count=1,
                confidence=0.8,
                source_types=["academic_paper"],
                evidence_ids=["ev_report"],
                status="partial",
            )
        ],
        coverage_by_paper={"Counterfactual Paper": 1.0},
        coverage_by_dimension={"core_counterfactual_inference": 1.0},
    )


def _observability() -> ObservabilitySnapshot:
    return ObservabilitySnapshot(
        generated_at=datetime.now(timezone.utc),
        source_mix={"academic_paper": 1},
        dimension_coverage={"core_counterfactual_inference": 1.0},
        paper_coverage={"Counterfactual Paper": 1.0},
        claim_pass_rate=1.0,
        red_team_challenge_rate=0.0,
        evidence_coverage_score=1.0,
        report_confidence=0.7,
        quality_gates=[
            QualityGate(
                gate_id="gate_report_readiness",
                name="Report readiness",
                status="pass",
                score=1.0,
                message="Ready enough for fallback regression.",
            )
        ],
    )
