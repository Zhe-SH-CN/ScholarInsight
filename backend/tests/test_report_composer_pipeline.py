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


class FailingReportAndSummaryComposer:
    def __init__(self, ctx):
        self.ctx = ctx

    async def write(self, *args, **kwargs) -> str:
        raise RuntimeError("simulated report failure")

    async def write_executive_summary(self, *args, **kwargs) -> str:
        raise RuntimeError("simulated summary failure")

    async def write_methodology(self, *args, **kwargs) -> str:
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


@pytest.mark.asyncio
async def test_write_report_fallback_splits_report_ready_and_audit_only_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_module, "ReportComposerAgent", FailingReportAndSummaryComposer)
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
        project_name="Report-ready rendering regression",
        target_topic="Counterfactual Inference",
        analysis_dimensions=["core_counterfactual_inference"],
    )
    status = await pipeline.prepare_run(request, owner="test")
    evidence = [
        _evidence_for_report_ready("ev_ready_1", "Paper A"),
        _evidence_for_report_ready("ev_ready_2", "Paper B"),
        _evidence_for_report_ready("ev_ready_3", "Paper C"),
        _evidence_for_report_ready("ev_ready_4", "Paper D"),
    ]
    claims = [_report_ready_claim(), _sample_limited_claim()]
    metrics = RunMetrics(
        sources_fetched=4,
        evidence_count=4,
        claim_count=2,
        verified_claim_count=2,
    )

    fallbacks = await pipeline.write_report(
        status.run_id,
        request,
        evidence,
        claims,
        metrics,
        _matrix_for_report_ready(),
        [],
        _observability(),
    )

    run_dir = pipeline.runs.run_dir(status.run_id)
    fallback_sections = {item["section"] for item in fallbacks}
    assert fallback_sections == {"report", "executive_summary"}

    summary = (run_dir / "reports" / "executive_summary.md").read_text(encoding="utf-8")
    assert "## Report-ready Findings" in summary
    assert "identifiability assumptions across counterfactual inference protocols" in summary
    assert "当前样本内" not in summary

    report = (run_dir / "reports" / "report.md").read_text(encoding="utf-8")
    assert "#### 可进入报告主体的综合结论" in report
    assert "identifiability assumptions across counterfactual inference protocols" in report
    assert "#### 待验证观察与补证线索" in report
    assert "暂不进入主结论：sample_limited_observation" in report
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


def _evidence_for_report_ready(evidence_id: str, paper: str) -> Evidence:
    now = datetime.now(timezone.utc)
    return Evidence(
        evidence_id=evidence_id,
        run_id="run_test",
        url=f"/papers/{paper}.pdf",
        title=paper,
        content="Counterfactual inference protocols discuss identifiability assumptions.",
        fetched_at=now,
        source_type="academic_paper",
        dimension="core_counterfactual_inference",
        dimension_label="Core Counterfactual Inference",
        paper=paper,
        fact="The paper discusses identifiability assumptions in counterfactual inference.",
        quote="identifiability assumptions in counterfactual inference",
        source_title=paper,
        source_url=f"/papers/{paper}.pdf",
        source_subtype="core_counterfactual_inference",
        confidence=0.82,
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


def _report_ready_claim() -> Claim:
    return Claim(
        claim_id="claim_report_ready",
        run_id="run_test",
        dimension="core_counterfactual_inference",
        dimension_label="Core Counterfactual Inference",
        claim=(
            "Multiple counterfactual inference papers organize evidence around "
            "identifiability assumptions across counterfactual inference protocols."
        ),
        supporting_evidence_ids=["ev_ready_1", "ev_ready_2", "ev_ready_3", "ev_ready_4"],
        confidence=0.88,
        risk_level="low",
        reasoning_summary="Supported by four independent core counterfactual inference papers.",
        verification_status="verified",
        claim_type="comparative",
        source_paper_count=4,
        evidence_support_level="strong",
        supporting_source_subtypes=["core_counterfactual_inference"],
        supporting_source_subtype_counts={"core_counterfactual_inference": 4},
        supporting_source_subtype_paper_counts={"core_counterfactual_inference": 4},
    )


def _sample_limited_claim() -> Claim:
    return Claim(
        claim_id="claim_sample_limited",
        run_id="run_test",
        dimension="core_counterfactual_inference",
        dimension_label="Core Counterfactual Inference",
        claim=(
            "当前样本内，多篇论文在 mechanism discussion 上出现相似观察，"
            "不单独构成领域趋势。"
        ),
        supporting_evidence_ids=["ev_ready_1", "ev_ready_2", "ev_ready_3", "ev_ready_4"],
        confidence=0.84,
        risk_level="low",
        reasoning_summary="Audit-only synthesis that must not become a main finding.",
        verification_status="verified",
        claim_type="comparative",
        source_paper_count=4,
        evidence_support_level="strong",
        supporting_source_subtypes=["core_counterfactual_inference"],
        supporting_source_subtype_counts={"core_counterfactual_inference": 4},
        supporting_source_subtype_paper_counts={"core_counterfactual_inference": 4},
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


def _matrix_for_report_ready() -> PaperPatternMatrix:
    return PaperPatternMatrix(
        generated_at=datetime.now(timezone.utc),
        papers=["Paper A", "Paper B", "Paper C", "Paper D"],
        dimensions=["core_counterfactual_inference"],
        dimension_labels={"core_counterfactual_inference": "Core Counterfactual Inference"},
        cells=[
            MatrixCell(
                paper=paper,
                dimension="core_counterfactual_inference",
                dimension_label="Core Counterfactual Inference",
                summary="The paper contributes evidence about counterfactual inference assumptions.",
                evidence_count=1,
                confidence=0.82,
                source_types=["academic_paper"],
                evidence_ids=[evidence_id],
                status="strong",
            )
            for paper, evidence_id in [
                ("Paper A", "ev_ready_1"),
                ("Paper B", "ev_ready_2"),
                ("Paper C", "ev_ready_3"),
                ("Paper D", "ev_ready_4"),
            ]
        ],
        coverage_by_paper={"Paper A": 1.0, "Paper B": 1.0, "Paper C": 1.0, "Paper D": 1.0},
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
