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


def test_report_ready_summary_keeps_cross_role_contrast_readable() -> None:
    request = ResearchRequest(
        project_name="Cross-role summary regression",
        target_topic="Scientific Reasoning with LLMs",
        analysis_dimensions=["lab_workflow_reasoning"],
    )
    claim = Claim(
        claim_id="claim_cross_role",
        run_id="run_test",
        dimension="lab_workflow_reasoning",
        dimension_label="实验流程推理",
        claim=(
            "在实验流程推理维度，4 篇论文共同围绕“scientific discovery workflow protocol”"
            "形成来源角色分工对照：scientific reasoning benchmark 来源（2 篇）侧重"
            "任务定义、数据集构造和评测协议；scientific discovery agent/workflow 来源（2 篇）"
            "侧重 agent workflow、工具调用和研究流程编排。该结论限定于假设生成、实验设计、"
            "数据驱动发现、agent 协作或研究流程编排这一证据轴，可作为报告主体中的范围限定综合结论。"
        ),
        supporting_evidence_ids=["ev_a", "ev_b", "ev_c", "ev_d"],
        confidence=0.88,
        risk_level="low",
        reasoning_summary="Supported by two benchmark papers and two discovery-agent papers.",
        verification_status="verified",
        claim_type="cross_role_contrast",
        source_paper_count=4,
        evidence_support_level="strong",
        supporting_source_subtypes=[
            "scientific_reasoning_benchmark",
            "scientific_discovery_agent",
        ],
        supporting_source_subtype_counts={
            "scientific_reasoning_benchmark": 2,
            "scientific_discovery_agent": 2,
        },
        supporting_source_subtype_paper_counts={
            "scientific_reasoning_benchmark": 2,
            "scientific_discovery_agent": 2,
        },
    )

    summary = pipeline_module.build_executive_summary(
        request,
        RunMetrics(sources_fetched=4, evidence_count=4, claim_count=1, verified_claim_count=1),
        _observability(),
        [],
        [claim],
    )

    assert "## Report-ready Findings" in summary
    assert "scientific discovery agent/workflow 来源（2 篇）侧重 agent workflow、工具调用和研究流程编排" in summary
    assert "age..." not in summary
    assert "agent..." not in summary


def test_analysis_report_includes_grounded_hypotheses_and_backing() -> None:
    request = ResearchRequest(
        project_name="Grounded hypothesis regression",
        target_topic="Counterfactual Inference",
        analysis_dimensions=["core_counterfactual_inference"],
    )
    evidence = [
        _evidence_for_report_ready("ev_ready_1", "Paper A"),
        _evidence_for_report_ready("ev_ready_2", "Paper B"),
        _evidence_for_report_ready("ev_ready_3", "Paper C"),
        _evidence_for_report_ready("ev_ready_4", "Paper D"),
    ]
    claim = _report_ready_claim()

    report = pipeline_module.build_analysis_report(
        request,
        evidence,
        [claim],
        RunMetrics(sources_fetched=4, evidence_count=4, claim_count=1, verified_claim_count=1),
        _matrix_for_report_ready(),
        [],
        _observability(),
    )

    assert "## 可检验研究假设与学术背书" in report
    assert "## 证据背书机会表" in report
    assert "| 机会 | 证据轴 | 学术背书 | 下一步验证 |" in report
    assert "### H1. 将核心反事实推断中的多论文共识转化为可复现实验协议" in report
    assert "**证据背书**：这一综合判断由 4 篇独立论文" in report
    assert "关注 report-ready evidence axis" in report
    assert "代表性支撑证据" in report
    assert "代表性证据显示" not in report
    assert "本维度当前证据仅支撑待验证观察" not in report
    assert "该 claim" not in report
    assert "不是直接写成领域趋势" in report


def test_analysis_report_does_not_promote_audit_only_representative_evidence() -> None:
    request = ResearchRequest(
        project_name="Audit-only representative evidence regression",
        target_topic="Counterfactual Inference",
        analysis_dimensions=["core_counterfactual_inference"],
    )
    report = pipeline_module.build_analysis_report(
        request,
        [_evidence()],
        [_claim()],
        RunMetrics(sources_fetched=1, evidence_count=1, claim_count=1, verified_claim_count=1),
        _matrix(),
        [],
        _observability(),
    )

    assert "当前尚无 report-ready 推理模式关注点" in report
    assert "本维度当前证据仅支撑待验证观察" in report
    assert "代表性证据显示" not in report


def test_report_ready_representative_evidence_uses_clean_snippets() -> None:
    request = ResearchRequest(
        project_name="Representative snippet cleanup regression",
        target_topic="Counterfactual Inference",
        analysis_dimensions=["core_counterfactual_inference"],
    )
    evidence = [
        _evidence_for_report_ready(f"ev_ready_{index}", paper).model_copy(
            update={
                "fact": (
                    f"{paper} 在核心反事实推断相关论文文本中出现了可核验表述："
                    "identifiability assumptions are evaluated through a shared protocol"
                ),
                "quote": "identifiability assumptions are evaluated through a shared protocol",
            }
        )
        for index, paper in enumerate(["Paper A", "Paper B", "Paper C", "Paper D"], start=1)
    ]
    report = pipeline_module.build_analysis_report(
        request,
        evidence,
        [_report_ready_claim()],
        RunMetrics(sources_fetched=4, evidence_count=4, claim_count=1, verified_claim_count=1),
        _matrix_for_report_ready(),
        [],
        _observability(),
    )

    body = report.split("## Evidence 附录", 1)[0]
    assert "代表性支撑证据" in body
    assert "identifiability assumptions are evaluated through a shared protocol" in body
    assert "相关论文文本中出现了可核验表述" not in body


def test_recommendations_use_grounded_opportunity_when_coverage_is_full() -> None:
    request = ResearchRequest(
        project_name="Grounded recommendation regression",
        target_topic="Counterfactual Inference",
        analysis_dimensions=["core_counterfactual_inference"],
    )
    evidence = [
        _evidence_for_report_ready("ev_ready_1", "Paper A"),
        _evidence_for_report_ready("ev_ready_2", "Paper B"),
        _evidence_for_report_ready("ev_ready_3", "Paper C"),
        _evidence_for_report_ready("ev_ready_4", "Paper D"),
    ]
    recommendations = pipeline_module.build_recommendations(
        "run_test",
        request,
        [_report_ready_claim()],
        evidence,
        _matrix_for_report_ready(),
    )

    assert len(recommendations) == 1
    assert recommendations[0].title == "将核心反事实推断中的多论文共识转化为可复现实验协议"
    assert "可复现实验协议" in recommendations[0].recommendation
    assert recommendations[0].rationale.startswith("这一综合判断由 4 篇独立论文")


def test_report_matrix_prioritizes_report_ready_supporting_papers() -> None:
    markdown = pipeline_module.build_matrix_markdown(
        _large_matrix_for_report_ready(),
        {},
        focus_evidence_ids={"ev_ready_1", "ev_ready_2", "ev_ready_3", "ev_ready_4"},
        max_papers=4,
    )

    header = markdown.splitlines()[0]
    assert "Paper A" in header
    assert "Paper B" in header
    assert "Paper C" in header
    assert "Paper D" in header
    assert "Paper E" not in header
    assert "完整 matrix" in markdown


def test_dimension_fallback_does_not_expand_unknown_cells() -> None:
    lines = pipeline_module.build_dimension_fallback_points(
        [
            MatrixCell(
                paper="Paper With No Evidence",
                dimension="core_counterfactual_inference",
                dimension_label="Core Counterfactual Inference",
                summary="尚未在本次公开资料中找到 Paper With No Evidence 的核心反事实推断强证据。",
                evidence_count=0,
                confidence=0.0,
                source_types=[],
                evidence_ids=[],
                status="unknown",
            )
        ]
    )

    rendered = "\n".join(lines)
    assert "暂无可引用证据" in rendered
    assert "Paper With No Evidence 的核心反事实推断强证据" not in rendered


def test_dimension_fallback_does_not_expand_weak_single_evidence_cells() -> None:
    lines = pipeline_module.build_dimension_fallback_points(
        [
            MatrixCell(
                paper="Weak Evidence Paper",
                dimension="core_counterfactual_inference",
                dimension_label="Core Counterfactual Inference",
                summary="Weak Evidence Paper has one isolated observation.",
                evidence_count=1,
                confidence=0.55,
                source_types=["academic_paper"],
                evidence_ids=["ev_weak"],
                status="weak",
            )
        ]
    )

    rendered = "\n".join(lines)
    assert "只有弱单证据" in rendered
    assert "Weak Evidence Paper has one isolated observation" not in rendered


def test_report_matrix_does_not_expand_weak_single_evidence_cells() -> None:
    markdown = pipeline_module.build_matrix_markdown(
        PaperPatternMatrix(
            generated_at=datetime.now(timezone.utc),
            papers=["Weak Evidence Paper"],
            dimensions=["core_counterfactual_inference"],
            dimension_labels={"core_counterfactual_inference": "Core Counterfactual Inference"},
            cells=[
                MatrixCell(
                    paper="Weak Evidence Paper",
                    dimension="core_counterfactual_inference",
                    dimension_label="Core Counterfactual Inference",
                    summary="Weak Evidence Paper has one isolated observation.",
                    evidence_count=1,
                    confidence=0.55,
                    source_types=["academic_paper"],
                    evidence_ids=["ev_weak"],
                    status="weak",
                )
            ],
            coverage_by_paper={"Weak Evidence Paper": 1.0},
            coverage_by_dimension={"core_counterfactual_inference": 1.0},
        )
    )

    assert "仅作审计线索" in markdown
    assert "Weak Evidence Paper has one isolated observation" not in markdown


def test_dimension_overview_uses_report_ready_axis_when_available() -> None:
    lines = pipeline_module.build_dimension_overview(
        "core_counterfactual_inference",
        _matrix_for_report_ready().cells,
        [_report_ready_claim()],
    )

    rendered = "\n".join(lines)
    assert "可进入主体的证据轴" in rendered
    assert "report-ready support" in rendered
    assert "当前最可引用" not in rendered


def test_dimension_overview_does_not_promote_weak_cells() -> None:
    lines = pipeline_module.build_dimension_overview(
        "core_counterfactual_inference",
        [
            MatrixCell(
                paper="Weak Evidence Paper",
                dimension="core_counterfactual_inference",
                dimension_label="Core Counterfactual Inference",
                summary="Weak Evidence Paper has one isolated observation.",
                evidence_count=1,
                confidence=0.55,
                source_types=["academic_paper"],
                evidence_ids=["ev_weak"],
                status="weak",
            )
        ],
        [],
    )

    rendered = "\n".join(lines)
    assert "weak/single evidence" in rendered
    assert "Weak Evidence Paper has one isolated observation" not in rendered


def test_analysis_report_does_not_repeat_weak_only_dimension_warning() -> None:
    request = ResearchRequest(
        project_name="Weak-only dimension warning regression",
        target_topic="Counterfactual Inference",
        analysis_dimensions=["core_counterfactual_inference"],
    )
    report = pipeline_module.build_analysis_report(
        request,
        [],
        [],
        RunMetrics(sources_fetched=1, evidence_count=1, claim_count=0, verified_claim_count=0),
        PaperPatternMatrix(
            generated_at=datetime.now(timezone.utc),
            papers=["Weak Evidence Paper"],
            dimensions=["core_counterfactual_inference"],
            dimension_labels={"core_counterfactual_inference": "Core Counterfactual Inference"},
            cells=[
                MatrixCell(
                    paper="Weak Evidence Paper",
                    dimension="core_counterfactual_inference",
                    dimension_label="Core Counterfactual Inference",
                    summary="Weak Evidence Paper has one isolated observation.",
                    evidence_count=1,
                    confidence=0.55,
                    source_types=["academic_paper"],
                    evidence_ids=["ev_weak"],
                    status="weak",
                )
            ],
            coverage_by_paper={"Weak Evidence Paper": 1.0},
            coverage_by_dimension={"core_counterfactual_inference": 1.0},
        ),
        [],
        _observability(),
    )

    assert "weak/single evidence" in report
    assert "当前维度只有弱单证据" not in report
    assert "Weak Evidence Paper has one isolated observation" not in report


def test_matrix_insights_prefer_report_ready_evidence_axes() -> None:
    insights = pipeline_module.build_matrix_insights(
        _large_matrix_for_report_ready(),
        [_report_ready_claim()],
        {},
    )

    rendered = "\n".join(insights)
    assert "report-ready evidence axis" in rendered
    assert "矩阵覆盖度只作为审计背景" in rendered
    assert "需要补证" not in rendered


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


def _large_matrix_for_report_ready() -> PaperPatternMatrix:
    papers = [f"Paper {letter}" for letter in "ABCDEFGH"]
    evidence_ids = [
        "ev_ready_1",
        "ev_ready_2",
        "ev_ready_3",
        "ev_ready_4",
        "ev_extra_5",
        "ev_extra_6",
        "ev_extra_7",
        "ev_extra_8",
    ]
    return PaperPatternMatrix(
        generated_at=datetime.now(timezone.utc),
        papers=papers,
        dimensions=["core_counterfactual_inference"],
        dimension_labels={"core_counterfactual_inference": "Core Counterfactual Inference"},
        cells=[
            MatrixCell(
                paper=paper,
                dimension="core_counterfactual_inference",
                dimension_label="Core Counterfactual Inference",
                summary=f"{paper} contributes evidence about counterfactual inference assumptions.",
                evidence_count=1,
                confidence=0.82,
                source_types=["academic_paper"],
                evidence_ids=[evidence_id],
                status="strong",
            )
            for paper, evidence_id in zip(papers, evidence_ids)
        ],
        coverage_by_paper={paper: 1.0 for paper in papers},
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
