"""Report composer pipeline fallbacks and observability."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cg.orchestrator import pipeline as pipeline_module
from cg.schemas.research import (
    Claim,
    CounterexampleAuditRow,
    Evidence,
    MatrixCell,
    ObservabilitySnapshot,
    PaperPatternMatrix,
    QualityGate,
    ResearchRequest,
    RunMetrics,
    SourceCandidate,
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
    assert "当前证据不是支持泛化结论" in summary
    assert "可被任务化的“核心反事实推断”瓶颈" in summary
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
    assert "source-role 分工" in summary
    assert "scientific discovery agent/workflow 2 篇" in summary
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

    summary_block = report.split("## 推理模式对比矩阵", 1)[0]
    assert "背景：Counterfactual Inference 当前已有可进入主体的证据轴" in summary_block
    assert "缺口：最稳健的切入点是把“核心反事实推断”从综述观察转为可复现实验变量" in summary_block
    assert "贡献边界：主体结论只接收通过 `g(c)` 证据门控的 claim" in summary_block
    assert "Multiple counterfactual inference papers organize evidence" not in summary_block
    assert "## 可检验研究假设与学术背书" in report
    assert "## 证据背书机会表" in report
    assert "## 形式化证据门控与数学背书" in report
    assert "## 可证伪实验设计与负结果记录" in report
    assert "Falsification coverage：1/1" in report
    assert "failure_observation" in report
    assert "g(c)=1 iff" in report
    assert "R_c subset R_reportable" in report
    assert "\\|E_c\\|=4" in report
    assert "\\|P_c\\|=4" in report
    assert "support=strong" in report
    assert "## 可投稿研究命题与实验化路径" in report
    assert "### P1. 把核心反事实推断中的证据轴转化为可投稿问题" in report
    assert "**问题定义**：在 Counterfactual Inference 的核心反事实推断 中" in report
    assert "**方法假设**：把“核心反事实推断”显式建模为 Counterfactual Inference 的可控实验变量" in report
    assert "**实验化验证**：以“核心反事实推断”为主轴" in report
    assert "**反例与负结果协议**" in report
    assert "机制消融若不能改变错误类型或指标，应作为负结果保留" in report
    assert "hard-negative 反例集和负结果记录" in report
    assert "| 机会 | 证据轴 | 学术背书 | 下一步验证 |" in report
    assert "### H1. 核心反事实推断中的证据轴实验问题" in report
    assert "把“核心反事实推断”定义为可控实验变量" in report
    assert "**反例与负结果**" in report
    assert "多论文共识转化为可复现实验协议" not in report
    assert "现有多论文共识是否能跨任务" not in report
    assert "**证据背书**：这一综合判断由 4 篇独立论文" in report
    assert "关注 report-ready evidence axis" in report
    assert "代表性支撑证据" in report
    assert "代表性证据显示" not in report
    assert "本维度当前证据仅支撑待验证观察" not in report
    assert "该 claim" not in report
    assert "不是直接写成领域趋势" in report


def test_counterexample_audit_rows_challenge_boundaries_without_supporting_claims() -> None:
    request = ResearchRequest(
        project_name="Counterexample audit regression",
        target_topic="Counterfactual Inference",
        analysis_dimensions=["core_counterfactual_inference"],
    )
    claim = _report_ready_claim().model_copy(
        update={"evidence_cluster_label": "counterfactual identifiability and assumptions"}
    )
    support_candidate = SourceCandidate(
        url="/papers/Paper A.pdf",
        title="Paper A",
        relevance_label="accept",
        relevance_score=0.91,
        source_subtype="core_counterfactual_inference",
    )
    rejected_candidate = SourceCandidate(
        url="/papers/adjacent_recource.pdf",
        title="Counterfactual recourse fairness under shifted assumptions",
        snippet="Counterfactual explanation and recourse benchmark with fairness constraints.",
        query="counterfactual inference identifiability assumptions benchmark",
        relevance_label="reject",
        rejection_reason="counterfactual explanation/fairness adjacent to core inference",
        relevance_score=0.58,
        source_subtype="counterfactual_explanation_or_fairness",
        source_subtype_reason="recourse/fairness source, not core inference support",
    )

    rows = pipeline_module.build_counterexample_audit_rows(
        request,
        [claim],
        [support_candidate, rejected_candidate],
        selected_source_urls=["/papers/Paper A.pdf"],
    )

    assert len(rows) == 1
    assert rows[0].target_claim_id == claim.claim_id
    assert rows[0].audit_role == "rejected_source_boundary"
    assert "source gate rejected it" in rows[0].audit_only_reason
    assert "只挑战适用边界，不提供正向支撑" in rows[0].boundary_challenge
    assert rows[0].report_visible is True
    assert rows[0].semantic_quality >= 0.45


def test_counterexample_audit_demotes_metadata_noise_from_report_table() -> None:
    request = ResearchRequest(
        project_name="Counterexample quality regression",
        target_topic="Mathematical Reasoning",
        analysis_dimensions=["formal_proof_symbolic_reasoning"],
    )
    claim = _report_ready_claim().model_copy(
        update={
            "claim_id": "claim_math_formal",
            "dimension": "formal_proof_symbolic_reasoning",
            "dimension_label": "形式证明与符号推理",
            "evidence_cluster_label": "mathematical proof verification protocol",
            "supporting_source_subtypes": ["formal_math_proving"],
            "supporting_source_subtype_counts": {"formal_math_proving": 4},
            "supporting_source_subtype_paper_counts": {"formal_math_proving": 4},
        }
    )
    metadata_noise = SourceCandidate(
        url="/papers/random_metadata.pdf",
        title="5ck9PIrTpH",
        query="mathematical reasoning proof verification",
        relevance_label="reject",
        rejection_reason="non-informative paper title metadata",
        relevance_score=0.0,
        source_subtype="mathematical_reasoning_adjacent",
    )
    proceedings_header = SourceCandidate(
        url="/papers/proceedings_header.pdf",
        title="Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume",
        query="mathematical reasoning proof verification",
        relevance_label="reject",
        rejection_reason="Mathematical reasoning query requires LLM plus mathematical reasoning/proof signal",
        relevance_score=0.0,
        source_subtype="formal_math_proving",
    )
    hard_negative = SourceCandidate(
        url="/papers/proofbridge.pdf",
        title="PROOFBRIDGE: Auto-Formalization of Natural Language Proofs in Lean via Joint Embeddings",
        snippet="Auto-formalization proof verification protocol in Lean.",
        query="mathematical reasoning proof verification Lean benchmark",
        relevance_label="reject",
        rejection_reason="Mathematical reasoning query requires LLM plus mathematical reasoning/proof signal",
        relevance_score=0.0,
        source_subtype="mathematical_reasoning_adjacent",
        source_subtype_reason="formal proof adjacent source without explicit LLM mathematical reasoning framing",
    )

    rows = pipeline_module.build_counterexample_audit_rows(
        request,
        [claim],
        [metadata_noise, proceedings_header, hard_negative],
    )

    by_title = {row.source_title: row for row in rows}
    assert by_title["5ck9PIrTpH"].counterexample_type == "metadata_noise"
    assert by_title["5ck9PIrTpH"].report_visible is False
    assert by_title["5ck9PIrTpH"].semantic_quality < 0.2
    assert by_title[proceedings_header.title].counterexample_type == "metadata_noise"
    assert by_title[proceedings_header.title].report_visible is False
    assert by_title[hard_negative.title].counterexample_type in {
        "hard_negative_boundary",
        "adjacent_boundary",
    }
    assert by_title[hard_negative.title].report_visible is True

    section = "\n".join(pipeline_module.build_counterexample_audit_section(rows, [claim]))
    assert "PROOFBRIDGE" in section
    assert "5ck9PIrTpH" not in section
    assert "metadata/noise row 已从下表降级" in section


def test_analysis_report_renders_counterexample_audit_as_audit_only_section() -> None:
    request = ResearchRequest(
        project_name="Counterexample audit report regression",
        target_topic="Counterfactual Inference",
        analysis_dimensions=["core_counterfactual_inference"],
    )
    evidence = [
        _evidence_for_report_ready("ev_ready_1", "Paper A"),
        _evidence_for_report_ready("ev_ready_2", "Paper B"),
        _evidence_for_report_ready("ev_ready_3", "Paper C"),
        _evidence_for_report_ready("ev_ready_4", "Paper D"),
    ]
    counterexample = CounterexampleAuditRow(
        audit_id="cex_test",
        target_claim_id="claim_report_ready",
        target_dimension="core_counterfactual_inference",
        target_dimension_label="核心反事实推断",
        target_axis="可识别性条件、结构因果假设、混杂处理或反事实查询约束",
        source_title="Counterfactual recourse fairness under shifted assumptions",
        source_url="/papers/adjacent_recource.pdf",
        source_subtype="counterfactual_explanation_or_fairness",
        relevance_score=0.58,
        relevance_label="reject",
        rejection_reason="counterfactual explanation/fairness adjacent to core inference",
        audit_role="rejected_source_boundary",
        audit_only_reason="source gate rejected it: adjacent source role",
        boundary_challenge="该来源只挑战适用边界，不提供正向支撑。",
    )

    report = pipeline_module.build_analysis_report(
        request,
        evidence,
        [_report_ready_claim()],
        RunMetrics(sources_fetched=4, evidence_count=4, claim_count=1, verified_claim_count=1),
        _matrix_for_report_ready(),
        [],
        _observability(),
        [counterexample],
    )

    assert "## 反例与负结果审计" in report
    assert "不进入 Evidence extraction、Claim synthesis 或 `g(c)` 证据证书" in report
    assert "Counterfactual recourse fairness under shifted assumptions" in report
    assert "source gate rejected it: adjacent source role" in report
    evidence_appendix = report.split("## Evidence 附录", 1)[1]
    assert "Counterfactual recourse fairness under shifted assumptions" not in evidence_appendix


def test_falsification_plan_rows_cover_report_ready_claims_and_link_hard_negatives() -> None:
    request = ResearchRequest(
        project_name="Falsification plan regression",
        target_topic="Counterfactual Inference",
        analysis_dimensions=["core_counterfactual_inference"],
    )
    counterexample = CounterexampleAuditRow(
        audit_id="cex_falsify",
        target_claim_id="claim_report_ready",
        target_dimension="core_counterfactual_inference",
        target_dimension_label="Core Counterfactual Inference",
        target_axis="Core Counterfactual Inference",
        source_title="Counterfactual recourse fairness under shifted assumptions",
        source_url="/papers/adjacent_recource.pdf",
        source_subtype="counterfactual_explanation_or_fairness",
        relevance_score=0.58,
        relevance_label="reject",
        rejection_reason="counterfactual explanation/fairness adjacent to core inference",
        counterexample_type="hard_negative_boundary",
        semantic_quality=0.7,
        report_visible=True,
        boundary_challenge="Adjacent fairness source challenges the inference boundary.",
    )

    rows = pipeline_module.build_falsification_plan_rows(
        request,
        [_report_ready_claim(), _sample_limited_claim()],
        [counterexample],
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.target_claim_id == "claim_report_ready"
    assert row.linked_counterexample_audit_ids == ["cex_falsify"]
    assert "去除或打乱" in row.falsification_criterion
    assert "paired benchmark" in row.benchmark_or_task_perturbation
    assert "错误类型不变" in row.expected_failure_mode
    assert row.negative_result_logging_schema["decision"] == "support, falsified, or scope_narrowing_required"


def test_falsification_plan_section_degrades_without_report_ready_claims() -> None:
    section = "\n".join(pipeline_module.build_falsification_plan_section([], []))

    assert "当前尚无通过 `g(c)` 的 report-ready claim" in section
    assert "不生成可证伪实验计划" in section


def test_formal_evidence_gate_records_cross_role_minimum_certificate() -> None:
    claim = Claim(
        claim_id="claim_cross_role_gate",
        run_id="run_test",
        dimension="lab_workflow_reasoning",
        dimension_label="实验流程推理",
        claim=(
            "在实验流程推理维度，4 篇论文共同围绕 scientific discovery workflow protocol "
            "形成来源角色分工对照。"
        ),
        supporting_evidence_ids=["ev_a", "ev_b", "ev_c", "ev_d"],
        confidence=0.9,
        risk_level="low",
        reasoning_summary="Supported by two benchmark papers and two workflow papers.",
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
        evidence_cluster_label="scientific discovery workflow protocol",
    )

    gate = "\n".join(
        pipeline_module.build_formal_evidence_gate(
            [claim],
            {"ev_a": 1, "ev_b": 2, "ev_c": 3, "ev_d": 4},
        )
    )

    assert "cross-role contrast" in gate
    assert "\\|E_c\\|=4" in gate
    assert "\\|P_c\\|=4" in gate
    assert "min_r \\|P_c,r\\|=2" in gate
    assert "scientific reasoning benchmark:2" in gate
    assert "scientific discovery agent/workflow:2" in gate
    assert "g(c)=1，可进入主体" in gate


def test_submission_framing_keeps_cross_role_contrast_as_research_problem() -> None:
    claim = Claim(
        claim_id="claim_cross_role_submission",
        run_id="run_test",
        dimension="formal_proof_symbolic_reasoning",
        dimension_label="形式证明与符号推理",
        claim=(
            "在形式证明与符号推理维度，4 篇论文共同围绕 mathematical proof verification protocol "
            "形成来源角色分工对照。"
        ),
        supporting_evidence_ids=["ev_a", "ev_b", "ev_c", "ev_d"],
        confidence=0.9,
        risk_level="low",
        reasoning_summary="Supported by two proof papers and two benchmark papers.",
        verification_status="verified",
        claim_type="cross_role_contrast",
        source_paper_count=4,
        evidence_support_level="strong",
        supporting_source_subtypes=[
            "formal_math_proving",
            "math_reasoning_benchmark",
        ],
        supporting_source_subtype_counts={
            "formal_math_proving": 2,
            "math_reasoning_benchmark": 2,
        },
        supporting_source_subtype_paper_counts={
            "formal_math_proving": 2,
            "math_reasoning_benchmark": 2,
        },
        evidence_cluster_label="mathematical proof verification protocol",
    )
    request = ResearchRequest(
        project_name="Submission framing regression",
        target_topic="Mathematical Reasoning",
        analysis_dimensions=["formal_proof_symbolic_reasoning"],
    )

    framing = "\n".join(
        pipeline_module.build_submission_framing(
            request,
            [claim],
            {"ev_a": 1, "ev_b": 2, "ev_c": 3, "ev_d": 4},
        )
    )

    assert "### P1. 把形式证明与符号推理中的来源分工转化为可投稿问题" in framing
    assert "source-role 分工" in framing
    assert "formal math proving 2 篇；mathematical reasoning benchmark 2 篇" in framing
    assert "构造一个覆盖形式证明与符号推理的统一 benchmark" in framing
    assert "分别加入/移除各 source role 对应的机制" in framing
    assert "source-role 交互失败" in framing
    assert "跨任务不迁移" in framing
    assert "无收益、性能下降或错误类型不变的负结果" in framing


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


def test_report_ready_representative_evidence_uses_axis_summary() -> None:
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
    assert "共同支撑" in body
    assert "主体结论只引用该范围限定证据" in body
    assert "原文片段保留在 Evidence 附录" in body
    assert "identifiability assumptions are evaluated through a shared protocol" not in body
    assert "相关论文文本中出现了可核验表述" not in body


def test_evidence_appendix_cleans_mechanical_fact_prefix() -> None:
    request = ResearchRequest(
        project_name="Evidence appendix cleanup regression",
        target_topic="Counterfactual Inference",
        analysis_dimensions=["core_counterfactual_inference"],
    )
    evidence = [
        _evidence_for_report_ready("ev_ready_1", "Paper A").model_copy(
            update={
                "fact": (
                    "Paper A 在核心反事实推断相关论文文本中出现了可核验表述："
                    "identifiability assumptions are evaluated through a shared protocol"
                ),
                "quote": "identifiability assumptions are evaluated through a shared protocol",
            }
        ),
        _evidence_for_report_ready("ev_ready_2", "Paper B"),
        _evidence_for_report_ready("ev_ready_3", "Paper C"),
        _evidence_for_report_ready("ev_ready_4", "Paper D"),
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

    appendix = report.split("## Evidence 附录", 1)[1]
    assert "相关论文文本中出现了可核验表述" not in appendix
    assert "identifiability assumptions are evaluated through a shared protocol" in appendix


def test_report_ready_representative_evidence_filters_ocr_fragments() -> None:
    request = ResearchRequest(
        project_name="Representative OCR cleanup regression",
        target_topic="Counterfactual Inference",
        analysis_dimensions=["core_counterfactual_inference"],
    )
    evidence = [
        _evidence_for_report_ready("ev_ready_1", "Paper A").model_copy(
            update={
                "quote": "nism for efficient and flexible subgraph retrieval while encoding directional struc",
                "fact": "nism for efficient and flexible subgraph retrieval while encoding directional struc",
                "source_subtype": "core_counterfactual_inference",
            }
        ),
        _evidence_for_report_ready("ev_ready_2", "Paper B").model_copy(
            update={
                "quote": "Figure 1: Retrieval effect on multi-hop/entity KGQA.",
                "fact": "Figure 1: Retrieval effect on multi-hop/entity KGQA.",
                "source_subtype": "core_counterfactual_inference",
            }
        ),
        _evidence_for_report_ready("ev_ready_3", "Paper C").model_copy(
            update={"quote": "The paper evaluates counterfactual identifiability assumptions."}
        ),
        _evidence_for_report_ready("ev_ready_4", "Paper D").model_copy(
            update={"quote": "These protocols compare structural assumptions across settings."}
        ),
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
    assert "nism for efficient" not in body
    assert "directional struc" not in body
    assert "Figure 1:" not in body
    assert "共同支撑" in body
    assert "主体结论只引用该范围限定证据" in body


def test_report_ready_representative_evidence_filters_title_fragments() -> None:
    item = _evidence_for_report_ready("ev_ready_1", "Correcting on Graph: Faithful Semantic Parsing over Knowledge Graphs with Large Language Models").model_copy(
        update={
            "quote": "Correcting on Graph: Faithful Semantic Parsing over Knowledge Graphs",
            "fact": "Correcting on Graph: Faithful Semantic Parsing over Knowledge Graphs",
            "source_subtype": "kgqa_or_graph_reasoning",
            "dimension_label": "KGQA/图推理",
        }
    )

    snippet = pipeline_module.report_evidence_snippet(item)

    assert "Correcting on Graph" not in snippet
    assert "围绕KGQA、语义解析和可执行查询构造提供可审计支撑" in snippet


def test_audit_only_claim_display_avoids_ocr_fragments() -> None:
    request = ResearchRequest(
        project_name="Audit-only OCR cleanup regression",
        target_topic="Counterfactual Inference",
        analysis_dimensions=["core_counterfactual_inference"],
    )
    claim = _sample_limited_claim().model_copy(
        update={
            "claim": (
                "当前样本内，作为跨论文对比性观察，core counterfactual inference 来源在核心反事实推断维度"
                "呈现出若干互补切入点：Paper A：nism for efficient mechanism；"
                "Paper B：velop an optimized protocol；不单独构成领域趋势。"
            ),
            "evidence_cluster_label": "counterfactual identifiability and assumptions",
        }
    )
    report = pipeline_module.build_analysis_report(
        request,
        [
            _evidence_for_report_ready("ev_ready_1", "Paper A"),
            _evidence_for_report_ready("ev_ready_2", "Paper B"),
            _evidence_for_report_ready("ev_ready_3", "Paper C"),
            _evidence_for_report_ready("ev_ready_4", "Paper D"),
        ],
        [claim],
        RunMetrics(sources_fetched=4, evidence_count=4, claim_count=1, verified_claim_count=1),
        _matrix_for_report_ready(),
        [],
        _observability(),
    )

    body = report.split("## Evidence 附录", 1)[0]
    assert "待验证综合观察" in body
    assert "可识别性条件、结构因果假设、混杂处理或反事实查询约束" in body
    assert "nism for efficient" not in body
    assert "velop an optimized" not in body


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
    assert recommendations[0].title == "核心反事实推断中的证据轴实验问题"
    assert "可控实验变量" in recommendations[0].recommendation
    assert "机制消融若不能改变错误类型或指标，应作为负结果保留" in recommendations[0].recommendation
    assert "多论文共识" not in recommendations[0].title
    assert "综述性观察" not in recommendations[0].recommendation
    assert recommendations[0].rationale.startswith("这一综合判断由 4 篇独立论文")
    assert any("预注册消融、反例集和负结果记录规则" in step for step in recommendations[0].next_steps)


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
