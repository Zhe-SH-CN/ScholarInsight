"""A real, lightweight research pipeline for the first runnable system."""

from __future__ import annotations

import hashlib
import asyncio
import json
import re
from collections import Counter, defaultdict
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable
from urllib.parse import urlparse
from uuid import uuid4

from cg.agents import (
    AGENT_SKILLS,
    AgentContext,
    AnalysisAndReviewAgent,
    EvidenceStructuringAgent,
    RESEARCH_AGENT_FLOW,
    ReportComposerAgent,
    ResearchPlanningAgent,
    SourceResearchAgent,
)
from cg.agents.research_agents import deterministic_red_team
from cg.llm import LLMClient
from cg.repositories.base import append_jsonl, atomic_write_json, write_text
from cg.repositories.evidence import EvidenceRepository
from cg.repositories.run import RunRepository
from cg.schemas.research import (
    Claim,
    EvidenceCluster,
    PaperPatternMatrix,
    PaperProfile,
    DEFAULT_DIMENSIONS,
    DIMENSION_LABELS,
    Evidence,
    EvidenceGraph,
    EvidenceGraphEdge,
    EvidenceGraphNode,
    MatrixCell,
    ObservabilitySnapshot,
    OpportunityRecommendation,
    QualityGate,
    QuickExtractRequest,
    RedTeamNote,
    ResearchFeedback,
    ResearchPlan,
    ResearchRequest,
    RunMetrics,
    RunStatus,
    SearchMemory,
    SourceCandidate,
    SourceDocument,
    TraceEvent,
)
from cg.settings import Settings, get_settings
from cg.tools.fetcher import Fetcher, RawPage
from cg.tools.search import SearchTool, classify_source
from cg.tools.local_paper_search import LocalPaperSearchTool


def atomic_write_json_sync(path: Path, data: Any) -> None:
    """Synchronous atomic JSON write for checkpoint saving."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


DIMENSION_KEYWORDS: dict[str, list[str]] = {
    "gap_driven_reframing": [
        "gap",
        "limitation",
        "challenge",
        "bottleneck",
        "reframe",
        "problem",
        "痛点",
        "瓶颈",
        "重构",
    ],
    "cross_domain_synthesis": [
        "cross-domain",
        "integrate",
        "combine",
        "hybrid",
        "knowledge graph",
        "multimodal",
        "跨领域",
        "融合",
    ],
    "representation_shift": [
        "representation",
        "embedding",
        "latent",
        "encode",
        "token",
        "graph representation",
        "表征",
        "嵌入",
    ],
    "modular_pipeline_composition": [
        "pipeline",
        "module",
        "component",
        "framework",
        "agent",
        "tool",
        "模块",
        "管线",
    ],
    "data_evaluation_engineering": [
        "benchmark",
        "dataset",
        "evaluation",
        "metric",
        "annotation",
        "test set",
        "评测",
        "数据集",
    ],
    "principled_probabilistic_modeling": [
        "probabilistic",
        "bayesian",
        "uncertainty",
        "distribution",
        "likelihood",
        "概率",
        "不确定性",
    ],
    "formal_experimental_tightening": [
        "theorem",
        "proof",
        "ablation",
        "controlled experiment",
        "formal",
        "rigorous",
        "理论",
        "消融",
    ],
    "approximation_engineering": [
        "approximation",
        "heuristic",
        "efficient",
        "scalable",
        "sampling",
        "近似",
        "启发式",
    ],
    "inference_time_control": [
        "inference-time",
        "decoding",
        "test-time",
        "planning",
        "search",
        "self-correction",
        "推理时",
        "解码",
    ],
    "structural_inductive_bias": [
        "inductive bias",
        "structure",
        "graph",
        "hierarchy",
        "constraint",
        "结构",
        "归纳偏置",
    ],
    "multiscale_hierarchical_modeling": [
        "hierarchical",
        "multi-scale",
        "coarse-to-fine",
        "level",
        "granularity",
        "分层",
        "多尺度",
    ],
    "mechanistic_decomposition": [
        "mechanism",
        "decompose",
        "interpretability",
        "causal mechanism",
        "analysis",
        "机制",
        "分解",
    ],
    "adversary_modeling": [
        "adversarial",
        "robust",
        "attack",
        "defense",
        "red-team",
        "jailbreak",
        "对抗",
        "鲁棒",
    ],
    "numerics_systems_codesign": [
        "system",
        "hardware",
        "kernel",
        "quantization",
        "latency",
        "throughput",
        "系统",
        "数值",
    ],
    "data_centric_optimization": [
        "data-centric",
        "data selection",
        "curation",
        "augmentation",
        "synthetic data",
        "quality",
        "数据中心",
        "数据选择",
    ],
    "kg_rag_architecture": [
        "kg-rag",
        "graph rag",
        "knowledge graph based retrieval-augmented generation",
        "retrieval-augmented generation",
        "framework",
        "architecture",
        "pipeline",
    ],
    "graph_retrieval_grounding": [
        "graph retrieval",
        "retrieval",
        "grounding",
        "graph traversal",
        "subgraph",
        "path retrieval",
        "knowledge graph retrieval",
    ],
    "kg_rag_evaluation": [
        "benchmark",
        "evaluation",
        "metric",
        "hallucination",
        "factuality",
        "faithfulness",
        "retrieve what you need",
    ],
    "multi_hop_kg_reasoning": [
        "multi-hop",
        "multi hop",
        "knowledge graph reasoning",
        "graph reasoning",
        "question answering",
        "path reasoning",
    ],
    "kg_construction_for_rag": [
        "knowledge graph construction",
        "kg construction",
        "entity extraction",
        "relation extraction",
        "schema",
        "graph construction",
    ],
    "rag_kg_boundary_analysis": [
        "boundary",
        "when to use graphs",
        "limitation",
        "failure",
        "comparison",
        "ablation",
    ],
    "scientific_problem_benchmarking": [
        "scientific problem",
        "benchmark",
        "dataset",
        "task",
        "evaluation",
        "scientific reasoning",
        "scientific question",
    ],
    "tool_augmented_scientific_reasoning": [
        "tool",
        "agent",
        "workflow",
        "experiment planning",
        "automated discovery",
        "search",
        "laboratory",
    ],
    "domain_grounding_verification": [
        "domain knowledge",
        "grounding",
        "verification",
        "evidence",
        "expert",
        "simulator",
        "scientific validity",
    ],
    "multimodal_scientific_reasoning": [
        "multimodal",
        "single-cell",
        "time series",
        "molecular",
        "microscopy",
        "image",
        "visual",
    ],
    "scientific_error_analysis": [
        "error",
        "failure",
        "limitation",
        "robustness",
        "uncertainty",
        "misleading",
        "invalid",
    ],
    "lab_workflow_reasoning": [
        "hypothesis",
        "experiment",
        "discovery",
        "lab",
        "laboratory",
        "protocol",
        "research workflow",
    ],
    "math_benchmark_evaluation": [
        "math benchmark",
        "gsm8k",
        "math",
        "mathematical reasoning",
        "word problem",
        "olympiad",
        "evaluation",
    ],
    "formal_proof_symbolic_reasoning": [
        "theorem",
        "proof",
        "formal",
        "lean",
        "isabelle",
        "symbolic",
        "formal verification",
    ],
    "program_tool_augmented_solving": [
        "program",
        "code",
        "python",
        "tool",
        "solver",
        "calculator",
        "execution",
    ],
    "self_consistency_search_verification": [
        "self-consistency",
        "search",
        "verifier",
        "verification",
        "step",
        "process reward",
        "chain-of-thought",
    ],
    "natural_language_to_formal_math": [
        "natural language",
        "formalization",
        "formal statement",
        "premise",
        "rationale",
        "translate",
        "math proof",
    ],
    "math_error_diagnosis": [
        "error",
        "mistake",
        "counterexample",
        "diagnosis",
        "failure",
        "robustness",
    ],
    "llm_causal_benchmarking": [
        "causal reasoning benchmark",
        "causalbench",
        "benchmark",
        "dataset",
        "evaluation",
        "causal question",
    ],
    "causal_intervention_counterfactual": [
        "intervention",
        "counterfactual",
        "do-calculus",
        "treatment",
        "potential outcome",
        "causal effect",
    ],
    "causal_explanation_mechanism": [
        "causal explanation",
        "mechanism",
        "cause",
        "causal graph",
        "structural causal model",
        "explain",
    ],
    "causal_pitfall_robustness": [
        "pitfall",
        "spurious",
        "robustness",
        "failure",
        "confounding",
        "bias",
        "shortcut",
    ],
    "causal_tool_symbolic_integration": [
        "tool",
        "symbolic",
        "causal graph",
        "scm",
        "do-calculus",
        "program",
    ],
    "causal_reasoning_evaluation_protocol": [
        "evaluation protocol",
        "metric",
        "task",
        "benchmark design",
        "controlled evaluation",
        "causal reasoning evaluation",
    ],
    "treatment_effect_estimation": [
        "treatment effect",
        "heterogeneous treatment",
        "causal effect",
        "potential outcomes",
        "uplift",
        "effect estimation",
    ],
    "core_counterfactual_inference": [
        "counterfactual inference",
        "structural causal model",
        "scm",
        "potential outcome",
        "counterfactual prediction",
        "counterfactual query",
    ],
    "counterfactual_explanation_fairness": [
        "counterfactual explanation",
        "recourse",
        "fairness",
        "counterfactual fairness",
        "algorithmic recourse",
    ],
    "temporal_counterfactual_estimation": [
        "temporal",
        "time series",
        "longitudinal",
        "dynamic",
        "sequential",
        "trajectory",
    ],
    "identifiability_assumption_sensitivity": [
        "identifiability",
        "assumption",
        "sensitivity",
        "unobserved confounding",
        "overlap",
        "positivity",
    ],
    "counterfactual_benchmarking": [
        "benchmark",
        "dataset",
        "evaluation",
        "metric",
        "counterfactual benchmark",
        "testbed",
    ],
    "graph_reasoning_benchmark": [
        "graph reasoning benchmark",
        "benchmark",
        "dataset",
        "multi-hop",
        "multi hop",
        "question answering",
    ],
    "kgqa_or_graph_reasoning": [
        "kgqa",
        "knowledge graph question answering",
        "graph question answering",
        "knowledge graph reasoning",
        "semantic parsing",
    ],
    "core_multi_hop_graph_reasoning": [
        "multi-hop reasoning",
        "multi hop reasoning",
        "path reasoning",
        "graph reasoning",
        "walk",
        "path composition",
    ],
    "semantic_parsing_grounding": [
        "semantic parsing",
        "grounding",
        "logical form",
        "query graph",
        "sparql",
        "knowledge base",
    ],
    "path_composition_reasoning": [
        "path composition",
        "reasoning path",
        "multi-hop path",
        "walk",
        "chain",
        "compositional reasoning",
    ],
    "graph_retrieval_boundary": [
        "graph retrieval",
        "retrieval augmented",
        "rag",
        "boundary",
        "retrieval guarantee",
        "graph-rag",
    ],
    "other": ["method", "result", "contribution", "approach", "方法", "结果", "贡献"],
}


class RunStopped(Exception):
    """Raised when the user requests a run to stop."""


class ResearchPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.runs = RunRepository(self.settings.data_dir)
        self.search = LocalPaperSearchTool(self.settings)
        self.fetcher = Fetcher(self.settings)
        self.llm = LLMClient(self.settings)
        # Red-team LLM uses independent model selection while reusing configured credentials.
        self._red_team_llm: LLMClient | None = None

    def _checkpoint_path(self, run_id: str) -> Path:
        return self.runs.run_dir(run_id) / "checkpoint.json"

    def _save_checkpoint(self, run_id: str, state: dict) -> None:
        path = self._checkpoint_path(run_id)
        atomic_write_json_sync(path, state)

    def _load_checkpoint(self, run_id: str) -> dict | None:
        path = self._checkpoint_path(run_id)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return None

    def _clear_checkpoint(self, run_id: str) -> None:
        path = self._checkpoint_path(run_id)
        if path.exists():
            path.unlink()

    def _red_team_provider_model(self) -> tuple[str, str]:
        provider = (self.settings.red_team_llm_provider or "gemini").strip().lower()
        model = (self.settings.red_team_llm_model or "").strip()
        if provider in {"gpt-5.5", "gpt55", "gpt_5_5"}:
            return "gemini", model or "gpt-5.5"
        if provider == "gemini":
            return "gemini", model or self.settings.gemini_model
        return provider, model or self.settings.cg_llm_model

    @property
    def red_team_llm(self) -> LLMClient:
        """Independent red-team reviewer LLM (lazy init)."""
        if self._red_team_llm is None:
            provider, model = self._red_team_provider_model()
            red_team_settings = Settings(
                cg_llm_provider=provider,
                cg_llm_model=model,
                cg_llm_temperature=self.settings.cg_llm_temperature,
                cg_llm_timeout_seconds=self.settings.cg_llm_timeout_seconds,
                cg_llm_min_interval_seconds=self.settings.cg_llm_min_interval_seconds,
                cg_llm_max_retries=self.settings.cg_llm_max_retries,
                cg_llm_rate_limit_cooldown_seconds=self.settings.cg_llm_rate_limit_cooldown_seconds,
                mimo_api_key=self.settings.mimo_api_key,
                mimo_base_url=self.settings.mimo_base_url,
                gemini_api_key=self.settings.gemini_api_key,
                gemini_base_url=self.settings.gemini_base_url,
                deepseek_api_key=self.settings.deepseek_api_key,
                deepseek_base_url=self.settings.deepseek_base_url,
                qwen_api_key=self.settings.qwen_api_key,
                qwen_base_url=self.settings.qwen_base_url,
                ark_api_key=self.settings.ark_api_key,
                ark_base_url=self.settings.ark_base_url,
            )
            self._red_team_llm = LLMClient(red_team_settings)
        return self._red_team_llm

    async def prepare_run(self, request: ResearchRequest, owner: str | None = None) -> RunStatus:
        return await self.runs.create(request, owner=owner)

    async def check_stop(self, run_id: str) -> None:
        if self.runs.is_stop_requested(run_id):
            raise RunStopped("Stopped by user")

    async def run(self, request: ResearchRequest, run_id: str) -> None:
        status = await self.runs.load_status(run_id)
        metrics = RunMetrics()

        if not self.llm.is_configured:
            status.status = "failed"
            status.current_stage = "Failed"
            status.error = (
                "⚠️ LLM API Key 未配置。ScholarInsight 需要 LLM 才能运行智能体搜索与分析。"
                "请在 backend/.env 中填写 MIMO_API_KEY，"
                "然后重启后端服务并重新发起研究。"
            )
            await self.runs.save_status(status)
            await self.trace(run_id, "Pipeline", "error", "failed", status.error)
            return

        try:
            status.status = "running"
            status.current_stage = "Planning"
            status.finished_at = None
            status.error = None
            await self.runs.save_status(status)

            ctx = self.agent_context(run_id)
            max_loops = request.max_research_loops or self.settings.cg_max_research_loops

            # ── 外层循环：Planning → Search → Analysis（可多轮）──
            feedback: ResearchFeedback | None = None
            all_evidence: list[Evidence] = []
            all_documents: list[SourceDocument] = []
            all_candidates: list[SourceCandidate] = []
            plan: ResearchPlan | None = None
            search_memory = SearchMemory(loop_round=1, remaining_source_slots=request.max_sources)

            for loop_round in range(1, max_loops + 1):
                await self.check_stop(run_id)
                loop_label = f"第 {loop_round} 轮"

                # ── Planning ──
                async with self.node(run_id, status, "ResearchPlanningAgent",
                                     f"{loop_label}：规划研究范围、搜索策略和质量规则"):
                    plan = await ResearchPlanningAgent(ctx).plan(
                        request, feedback=feedback, loop_round=loop_round
                    )
                    await atomic_write_json(self.runs.run_dir(run_id) / "plan.json", plan)
                    await atomic_write_json(self.runs.run_dir(run_id) / "dag.json", build_dag_snapshot(plan))
                    self._save_checkpoint(run_id, {"step": "source_research", "loop_round": loop_round, "max_loops": max_loops})

                # ── Source Research（ReAct 自主搜索）──
                async with self.node(run_id, status, "SourceResearchAgent",
                                     f"{loop_label}：两阶段内容搜索（批量Query快速获取信息 + Search LLM针对性补充）"):
                    source_agent = SourceResearchAgent(ctx)
                    candidates = await source_agent.discover(
                        request,
                        plan,
                        self.search,
                        memory=search_memory,
                        feedback=feedback,
                        loop_round=loop_round,
                    )
                    # 去重合并（多轮累积） — 保存所有发现的候选，不限制数量
                    existing_urls = {c.url for c in all_candidates}
                    remaining_slots = max(0, request.max_sources - len(all_candidates))
                    new_candidates = [c for c in candidates if c.url not in existing_urls][:remaining_slots]
                    dropped_for_limit = max(0, len([c for c in candidates if c.url not in existing_urls]) - len(new_candidates))
                    all_candidates.extend(new_candidates)
                    search_memory = SearchMemory(
                        seen_urls=[c.url for c in all_candidates],
                        seen_queries=unique_strings([c.query for c in all_candidates if c.query]),
                        remaining_source_slots=max(0, request.max_sources - len(all_candidates)),
                        resource_limit_reached=len(all_candidates) >= request.max_sources,
                        loop_round=loop_round,
                        feedback_message=feedback.message if feedback else "",
                    )
                    metrics.source_candidates = len(all_candidates)
                    metrics.sources_rejected = count_jsonl_lines(
                        self.runs.run_dir(run_id) / "sources" / "rejected_sources.jsonl"
                    )
                    # 保存全部新候选（不截断，让 UI 能展示所有来源）
                    await self.save_candidates(run_id, new_candidates)
                    if dropped_for_limit or len(all_candidates) >= request.max_sources:
                        await self.trace(
                            run_id,
                            "SourceResearchAgent",
                            "progress",
                            "resource_limit_reached",
                            f"已达到最大来源数 {request.max_sources}，停止保存和搜索更多来源，后续按现有内容生成报告",
                            {
                                "max_sources": request.max_sources,
                                "saved_sources": len(all_candidates),
                                "dropped_candidates": dropped_for_limit,
                            },
                        )
                    if not all_candidates:
                        status.warnings.append(
                            "未发现可用来源。请配置搜索 API Key（TAVILY_API_KEY / EXA_API_KEY / ZHIHU_API_KEY）"
                            "或填写 Seed URL。"
                        )

                    new_documents = await self.materialize_search_documents(run_id, new_candidates)
                    fallback_candidates = [
                        candidate
                        for candidate in sorted(new_candidates, key=lambda c: c.score, reverse=True)
                        if not candidate.content.strip()
                    ][: max(0, min(request.max_sources, 20))]
                    if fallback_candidates:
                        fetched_documents = await source_agent.collect(
                            fallback_candidates,
                            self.fetcher,
                            lambda candidate, page: self.save_document(run_id, candidate, page),
                            lambda phase, event_status, message, payload=None: self.trace(
                                run_id,
                                "SourceResearchAgent",
                                phase,
                                event_status,
                                f"本地抓取{'完成' if event_status == 'fetched' else '失败'}：{message}",
                                payload,
                            ),
                        )
                        new_documents.extend(fetched_documents)
                    all_documents.extend(new_documents)
                    metrics.sources_fetched = len([d for d in all_documents if d.ok])
                    metrics.sources_failed = len([d for d in all_documents if not d.ok])
                    status.metrics = metrics
                    await self.runs.save_status(status)
                    self._save_checkpoint(run_id, {"step": "evidence", "loop_round": loop_round, "max_loops": max_loops})

                if loop_round > 1 and not new_documents:
                    await self.trace(
                        run_id,
                        "Pipeline",
                        "progress",
                        "no_new_sources_to_analyze",
                        f"第 {loop_round} 轮没有新增来源，跳过重复 Evidence/Claim 分析并进入综合报告",
                        {"loop_round": loop_round, "existing_sources": len(all_candidates)},
                    )
                    self._save_checkpoint(
                        run_id,
                        {"step": "synthesis", "loop_round": loop_round - 1, "max_loops": max_loops},
                    )
                    break

                # ── Evidence Structuring ──
                async with self.node(run_id, status, "EvidenceStructuringAgent",
                                     f"{loop_label}：从已采集正文中抽取结构化 Evidence"):
                    new_evidence = await self.extract_and_save_evidence(
                        run_id, request, plan, new_documents, existing_evidence=all_evidence
                    )
                    # 去重合并（按 evidence_id）
                    existing_ev_ids = {ev.evidence_id for ev in all_evidence}
                    unique_new = [ev for ev in new_evidence if ev.evidence_id not in existing_ev_ids]
                    all_evidence.extend(unique_new)
                    metrics.evidence_count = len(all_evidence)
                    status.metrics = metrics
                    await self.runs.save_status(status)
                    if not all_evidence:
                        status.warnings.append("抓取完成，但没有抽取到有效 Evidence。")
                    self._save_checkpoint(run_id, {"step": "analysis", "loop_round": loop_round, "max_loops": max_loops})

                # ── Analysis & Review（含缺口评估）──
                async with self.node(run_id, status, "AnalysisAndReviewAgent",
                                     f"{loop_label}：生成 Claim、执行 Red Team 审查并评估证据缺口"):
                    claims = await self.generate_claims(run_id, all_evidence, request)
                    metrics.claim_count = len(claims)
                    reviewed_claims = await self.red_team(run_id, claims, all_evidence)
                    metrics.verified_claim_count = len(
                        [c for c in reviewed_claims if c.verification_status == "verified"]
                    )
                    metrics.challenged_claim_count = len(
                        [c for c in reviewed_claims if c.verification_status != "verified"]
                    )
                    status.metrics = metrics
                    await self.runs.save_status(status)

                    # 构建矩阵用于覆盖度计算
                    dimensions = plan.dimensions or request.analysis_dimensions or DEFAULT_DIMENSIONS
                    papers = select_matrix_papers(request, all_evidence)
                    interim_matrix = build_paper_pattern_matrix(all_evidence, papers, dimensions)
                    coverage_score = sum(interim_matrix.coverage_by_dimension.values()) / max(
                        1, len(interim_matrix.coverage_by_dimension)
                    )

                    # Analysis Agent 评估缺口
                    analysis_agent = AnalysisAndReviewAgent(ctx)
                    feedback = await analysis_agent.assess_gaps(
                        request, all_evidence, reviewed_claims, loop_round, coverage_score
                    )
                    await self.trace(
                        run_id, "AnalysisAndReviewAgent", "progress", "gap_assessed",
                        (
                            f"证据缺口评估完成：发现 {len(feedback.gaps)} 个待补充缺口，准备进入第 {loop_round + 1} 轮补充研究"
                            if feedback.needs_more_research
                            else f"证据覆盖已满足当前质量门禁，研究链路将在第 {loop_round} 轮后收敛，无需启动第 {loop_round + 1} 轮"
                        ),
                        {"loop_round": loop_round, "needs_more": feedback.needs_more_research,
                         "gap_count": len(feedback.gaps), "coverage": coverage_score},
                    )

                # 分析完成，保存 checkpoint（标记为 synthesis 或下一轮 planning）
                if feedback.needs_more_research and loop_round < max_loops and len(all_candidates) < request.max_sources:
                    self._save_checkpoint(run_id, {"step": "planning", "loop_round": loop_round + 1, "max_loops": max_loops})
                else:
                    self._save_checkpoint(run_id, {"step": "synthesis", "loop_round": loop_round, "max_loops": max_loops})

                # 是否继续循环
                if not feedback.needs_more_research or loop_round >= max_loops or len(all_candidates) >= request.max_sources:
                    if feedback.needs_more_research and loop_round >= max_loops:
                        await self.trace(
                            run_id,
                            "Pipeline",
                            "progress",
                            "loop_limit_reached",
                            f"已达到外层最大研究轮次 {max_loops}，仍有 {len(feedback.gaps)} 个缺口；停止补充搜索并进入报告生成",
                            {
                                "loop_round": loop_round,
                                "max_loops": max_loops,
                                "gap_count": len(feedback.gaps),
                                "gaps": [g.model_dump(mode="json") for g in feedback.gaps[:8]],
                            },
                        )
                    elif feedback.needs_more_research and len(all_candidates) >= request.max_sources:
                        await self.trace(
                            run_id,
                            "Pipeline",
                            "progress",
                            "resource_limit_reached",
                            f"已达到最大来源数 {request.max_sources}，即使仍有缺口也不再搜索，进入报告生成",
                            {
                                "max_sources": request.max_sources,
                                "source_count": len(all_candidates),
                                "gap_count": len(feedback.gaps),
                            },
                        )
                    elif not feedback.needs_more_research:
                        await self.trace(
                            run_id,
                            "Pipeline",
                            "progress",
                            "loop_converged",
                            f"第 {loop_round} 轮研究已完成证据闭环，跳过后续补充轮次并进入资产合成",
                            {
                                "loop_round": loop_round,
                                "max_loops": max_loops,
                                "coverage": feedback.coverage_score,
                            },
                        )
                    break

                await self.trace(
                    run_id, "Pipeline", "progress", "loop_continue",
                    f"第 {loop_round} 轮结束，发现 {len(feedback.gaps)} 个缺口，启动第 {loop_round + 1} 轮补充搜索",
                    {"loop_round": loop_round, "gaps": [g.model_dump() for g in feedback.gaps[:4]]},
                )

            # ── 最终 Analysis & Synthesize（只在最后一轮后执行完整资产合成）──
            async with self.node(run_id, status, "AnalysisAndReviewAgent", "整合全轮次结果，合成分析资产"):
                artifacts = await AnalysisAndReviewAgent(ctx).synthesize(
                    request,
                    plan,
                    all_evidence,
                    reviewed_claims,
                    metrics,
                    status.started_at,
                    matrix_builder=build_paper_pattern_matrix,
                    recommendations_builder=build_recommendations,
                    graph_builder=build_evidence_graph,
                    observability_builder=build_observability,
                    average_fn=average,
                    unique_strings_fn=unique_strings,
                )
                artifacts["evidence_clusters"] = build_evidence_clusters(all_evidence, reviewed_claims)
                await self.save_artifacts(run_id, artifacts)
                metrics.matrix_cell_count = len(artifacts["matrix"].cells)
                metrics.recommendation_count = len(artifacts["recommendations"])
                metrics.average_evidence_confidence = artifacts["average_evidence_confidence"]
                metrics.coverage_score = artifacts["observability"].evidence_coverage_score
                status.metrics = metrics
                await self.runs.save_status(status)

            # ── Report Composer ──
            async with self.node(run_id, status, "ReportComposerAgent",
                                 "生成本地 Markdown/JSON/CSV 交付文件"):
                report_fallbacks = await self.write_report(
                    run_id, request, all_evidence, reviewed_claims, metrics,
                    artifacts["matrix"], artifacts["recommendations"],
                    artifacts["observability"],
                )
                if report_fallbacks:
                    status.warnings.append(
                        f"ReportComposerAgent used deterministic fallback for {len(report_fallbacks)} section(s)."
                    )

            status.status = "completed"
            status.current_stage = "Completed"
            status.finished_at = datetime.now(timezone.utc)
            status.error = None
            status.metrics = metrics
            await self.runs.save_status(status)
            self._clear_checkpoint(run_id)
            await self.trace(
                run_id, "ReportComposerAgent", "complete", "completed",
                "运行完成", metrics.model_dump()
            )
        except RunStopped:
            status = await self.runs.mark_stopped(run_id, status)
            await self.trace(run_id, "Pipeline", "complete", "stopped", "Stopped by user")
        except Exception as exc:
            status.status = "failed"
            status.current_stage = "Failed"
            status.finished_at = datetime.now(timezone.utc)
            status.error = str(exc)
            status.metrics = metrics
            await self.runs.save_status(status)
            await self.trace(run_id, "Pipeline", "error", "failed", str(exc))

    async def resume(self, request: ResearchRequest, run_id: str) -> None:
        """Resume a run from its last checkpoint."""
        checkpoint = self._load_checkpoint(run_id)
        if not checkpoint:
            # No checkpoint, start fresh
            await self.run(request, run_id)
            return

        status = await self.runs.load_status(run_id)
        metrics = status.metrics or RunMetrics()

        if not self.llm.is_configured:
            status.status = "failed"
            status.current_stage = "Failed"
            status.error = "LLM API Key 未配置"
            await self.runs.save_status(status)
            return

        try:
            status.status = "running"
            status.finished_at = None
            status.error = None
            await self.runs.save_status(status)

            ctx = self.agent_context(run_id)
            max_loops = checkpoint.get(
                "max_loops",
                request.max_research_loops or self.settings.cg_max_research_loops,
            )
            step = checkpoint.get("step", "planning")
            loop_round = checkpoint.get("loop_round", 1)

            # Reconstruct state from persisted files
            run_dir = self.runs.run_dir(run_id)
            plan: ResearchPlan | None = None
            plan_path = run_dir / "plan.json"
            if plan_path.exists():
                plan = ResearchPlan(**json.loads(plan_path.read_text(encoding="utf-8")))

            # Reconstruct candidates from sources/_index.jsonl
            all_candidates: list[SourceCandidate] = []
            sources_index = run_dir / "sources" / "_index.jsonl"
            if sources_index.exists():
                for line in sources_index.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            all_candidates.append(SourceCandidate(**json.loads(line)))
                        except Exception:
                            pass

            # Reconstruct documents from documents/_index.jsonl
            all_documents: list[SourceDocument] = []
            docs_index = run_dir / "documents" / "_index.jsonl"
            if docs_index.exists():
                for line in docs_index.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            all_documents.append(SourceDocument(**json.loads(line)))
                        except Exception:
                            pass

            # Reconstruct evidence from evidence/*.json
            all_evidence: list[Evidence] = []
            evidence_repo = EvidenceRepository(run_dir)
            all_evidence = await evidence_repo.load_all()

            # Reconstruct search memory
            search_memory = SearchMemory(
                seen_urls=[c.url for c in all_candidates],
                seen_queries=unique_strings([c.query for c in all_candidates if c.query]),
                remaining_source_slots=max(0, request.max_sources - len(all_candidates)),
                resource_limit_reached=len(all_candidates) >= request.max_sources,
                loop_round=loop_round,
            )

            reviewed_claims: list[Claim] = []
            claims_by_id: dict[str, Claim] = {}
            claims_index = run_dir / "claims" / "_index.jsonl"
            if claims_index.exists():
                for line in claims_index.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            claim = Claim(**json.loads(line))
                            claims_by_id[claim.claim_id] = claim
                        except Exception:
                            pass
            if not claims_by_id:
                for path in sorted((run_dir / "claims").glob("claim_*.json")):
                    try:
                        claim = Claim(**json.loads(path.read_text(encoding="utf-8")))
                        claims_by_id[claim.claim_id] = claim
                    except Exception:
                        pass
            reviewed_claims = list(claims_by_id.values())
            feedback: ResearchFeedback | None = None
            if step == "synthesis":
                feedback = ResearchFeedback(
                    needs_more_research=False,
                    loop_round=loop_round,
                    message="Resumed at synthesis checkpoint.",
                )

            await self.trace(
                run_id, "Pipeline", "progress", "resumed",
                f"从 checkpoint 恢复：step={step}, loop_round={loop_round}, "
                f"candidates={len(all_candidates)}, documents={len(all_documents)}, evidence={len(all_evidence)}",
                {"step": step, "loop_round": loop_round},
            )

            for lr in range(loop_round, max_loops + 1):
                await self.check_stop(run_id)
                loop_label = f"第 {lr} 轮"
                new_documents: list[SourceDocument] = []

                # ── Planning ──
                if lr == loop_round and step not in ("planning",):
                    # Already done this round's planning, skip
                    pass
                else:
                    async with self.node(run_id, status, "ResearchPlanningAgent",
                                         f"{loop_label}：规划研究范围、搜索策略和质量规则"):
                        plan = await ResearchPlanningAgent(ctx).plan(
                            request, feedback=feedback, loop_round=lr
                        )
                        await atomic_write_json(run_dir / "plan.json", plan)
                        await atomic_write_json(run_dir / "dag.json", build_dag_snapshot(plan))
                        self._save_checkpoint(run_id, {"step": "source_research", "loop_round": lr, "max_loops": max_loops})

                # ── Source Research ──
                if lr == loop_round and step in ("evidence", "analysis", "synthesis"):
                    # Already done, skip
                    pass
                else:
                    async with self.node(run_id, status, "SourceResearchAgent",
                                         f"{loop_label}：两阶段内容搜索"):
                        source_agent = SourceResearchAgent(ctx)
                        candidates = await source_agent.discover(
                            request, plan, self.search,
                            memory=search_memory, feedback=feedback, loop_round=lr,
                        )
                        existing_urls = {c.url for c in all_candidates}
                        remaining_slots = max(0, request.max_sources - len(all_candidates))
                        new_candidates = [c for c in candidates if c.url not in existing_urls][:remaining_slots]
                        all_candidates.extend(new_candidates)
                        search_memory = SearchMemory(
                            seen_urls=[c.url for c in all_candidates],
                            seen_queries=unique_strings([c.query for c in all_candidates if c.query]),
                            remaining_source_slots=max(0, request.max_sources - len(all_candidates)),
                            resource_limit_reached=len(all_candidates) >= request.max_sources,
                            loop_round=lr,
                            feedback_message=feedback.message if feedback else "",
                        )
                        metrics.source_candidates = len(all_candidates)
                        metrics.sources_rejected = count_jsonl_lines(
                            self.runs.run_dir(run_id) / "sources" / "rejected_sources.jsonl"
                        )
                        await self.save_candidates(run_id, new_candidates)
                        new_documents = await self.materialize_search_documents(run_id, new_candidates)
                        fallback_candidates = [
                            c for c in sorted(new_candidates, key=lambda c: c.score, reverse=True)
                            if not c.content.strip()
                        ][: max(0, min(request.max_sources, 20))]
                        if fallback_candidates:
                            fetched = await source_agent.collect(
                                fallback_candidates, self.fetcher,
                                lambda c, p: self.save_document(run_id, c, p),
                                lambda ph, es, m, pl=None: self.trace(run_id, "SourceResearchAgent", ph, es, m, pl),
                            )
                            new_documents.extend(fetched)
                        all_documents.extend(new_documents)
                        metrics.sources_fetched = len([d for d in all_documents if d.ok])
                        metrics.sources_failed = len([d for d in all_documents if not d.ok])
                        status.metrics = metrics
                        await self.runs.save_status(status)
                        self._save_checkpoint(run_id, {"step": "evidence", "loop_round": lr, "max_loops": max_loops})

                if lr > 1 and step not in ("analysis", "synthesis") and not new_documents:
                    await self.trace(
                        run_id,
                        "Pipeline",
                        "progress",
                        "no_new_sources_to_analyze",
                        f"第 {lr} 轮没有新增来源，跳过重复 Evidence/Claim 分析并进入综合报告",
                        {"loop_round": lr, "existing_sources": len(all_candidates)},
                    )
                    break

                # ── Evidence Structuring ──
                if lr == loop_round and step in ("analysis", "synthesis"):
                    pass
                else:
                    async with self.node(run_id, status, "EvidenceStructuringAgent",
                                         f"{loop_label}：从已采集正文中抽取结构化 Evidence"):
                        new_evidence = await self.extract_and_save_evidence(
                            run_id, request, plan, new_documents, existing_evidence=all_evidence
                        )
                        existing_ev_ids = {ev.evidence_id for ev in all_evidence}
                        unique_new = [ev for ev in new_evidence if ev.evidence_id not in existing_ev_ids]
                        all_evidence.extend(unique_new)
                        metrics.evidence_count = len(all_evidence)
                        status.metrics = metrics
                        await self.runs.save_status(status)
                        self._save_checkpoint(run_id, {"step": "analysis", "loop_round": lr, "max_loops": max_loops})

                # ── Analysis & Review ──
                if lr == loop_round and step == "synthesis":
                    pass
                else:
                    async with self.node(run_id, status, "AnalysisAndReviewAgent",
                                         f"{loop_label}：生成 Claim、执行 Red Team 审查并评估证据缺口"):
                        claims = await self.generate_claims(run_id, all_evidence, request)
                        metrics.claim_count = len(claims)
                        reviewed_claims = await self.red_team(run_id, claims, all_evidence)
                        metrics.verified_claim_count = len(
                            [c for c in reviewed_claims if c.verification_status == "verified"]
                        )
                        metrics.challenged_claim_count = len(
                            [c for c in reviewed_claims if c.verification_status != "verified"]
                        )
                        status.metrics = metrics
                        await self.runs.save_status(status)

                        dimensions = plan.dimensions or request.analysis_dimensions or DEFAULT_DIMENSIONS
                        papers = select_matrix_papers(request, all_evidence)
                        interim_matrix = build_paper_pattern_matrix(all_evidence, papers, dimensions)
                        coverage_score = sum(interim_matrix.coverage_by_dimension.values()) / max(
                            1, len(interim_matrix.coverage_by_dimension)
                        )

                        analysis_agent = AnalysisAndReviewAgent(ctx)
                        feedback = await analysis_agent.assess_gaps(
                            request, all_evidence, reviewed_claims, lr, coverage_score
                        )

                # Loop decision
                if not feedback.needs_more_research or lr >= max_loops or len(all_candidates) >= request.max_sources:
                    break
                search_memory = SearchMemory(
                    seen_urls=[c.url for c in all_candidates],
                    seen_queries=unique_strings([c.query for c in all_candidates if c.query]),
                    remaining_source_slots=max(0, request.max_sources - len(all_candidates)),
                    resource_limit_reached=len(all_candidates) >= request.max_sources,
                    loop_round=lr + 1,
                    feedback_message=feedback.message if feedback else "",
                )

            # ── Synthesis & Report ──
            self._save_checkpoint(run_id, {"step": "synthesis", "loop_round": loop_round, "max_loops": max_loops})
            async with self.node(run_id, status, "AnalysisAndReviewAgent", "整合全轮次结果，合成分析资产"):
                artifacts = await AnalysisAndReviewAgent(ctx).synthesize(
                    request, plan, all_evidence, reviewed_claims, metrics,
                    status.started_at,
                    matrix_builder=build_paper_pattern_matrix,
                    recommendations_builder=build_recommendations,
                    graph_builder=build_evidence_graph,
                    observability_builder=build_observability,
                    average_fn=average,
                    unique_strings_fn=unique_strings,
                )
                artifacts["evidence_clusters"] = build_evidence_clusters(all_evidence, reviewed_claims)
                await self.save_artifacts(run_id, artifacts)
                metrics.matrix_cell_count = len(artifacts["matrix"].cells)
                metrics.recommendation_count = len(artifacts["recommendations"])
                metrics.average_evidence_confidence = artifacts["average_evidence_confidence"]
                metrics.coverage_score = artifacts["observability"].evidence_coverage_score
                status.metrics = metrics
                await self.runs.save_status(status)

            async with self.node(run_id, status, "ReportComposerAgent", "生成本地 Markdown/JSON/CSV 交付文件"):
                report_fallbacks = await self.write_report(
                    run_id, request, all_evidence, reviewed_claims, metrics,
                    artifacts["matrix"], artifacts["recommendations"], artifacts["observability"],
                )
                if report_fallbacks:
                    status.warnings.append(
                        f"ReportComposerAgent used deterministic fallback for {len(report_fallbacks)} section(s)."
                    )

            status.status = "completed"
            status.current_stage = "Completed"
            status.finished_at = datetime.now(timezone.utc)
            status.error = None
            status.metrics = metrics
            await self.runs.save_status(status)
            self._clear_checkpoint(run_id)
            await self.trace(run_id, "ReportComposerAgent", "complete", "completed", "运行完成", metrics.model_dump())
        except RunStopped:
            status = await self.runs.mark_stopped(run_id, status)
            await self.trace(run_id, "Pipeline", "complete", "stopped", "Stopped by user")
        except Exception as exc:
            status.status = "failed"
            status.current_stage = "Failed"
            status.finished_at = datetime.now(timezone.utc)
            status.error = str(exc)
            status.metrics = metrics
            await self.runs.save_status(status)
            await self.trace(run_id, "Pipeline", "error", "failed", str(exc))

    async def quick_extract(
        self, request: QuickExtractRequest, owner: str | None = None
    ) -> tuple[RunStatus, list[Evidence]]:
        if request.run_id:
            status = await self.runs.assert_access(request.run_id, owner)
            run_id = request.run_id
        else:
            status = await self.runs.create(
                ResearchRequest(
                    project_name="Quick Extract",
                    target_topic=urlparse(str(request.url)).netloc,
                    seed_papers=[],
                    seed_urls=[str(request.url)],
                    auto_discover_sources=False,
                    max_sources=1,
                ),
                owner=owner,
            )
            run_id = status.run_id

        page = await self.fetcher.fetch(str(request.url))
        source = SourceCandidate(
            url=str(request.url),
            title=page.title,
            source_type=(
                classify_source(str(request.url), page.title)
                if request.source_type == "other"
                else request.source_type
            ),
            score=0.7,
        )
        document = await self.save_document(run_id, source, page)
        evidence = await self.extract_and_save_evidence(
            run_id,
            ResearchRequest(
                project_name=status.project_name,
                target_topic=status.target_topic,
                seed_papers=[],
                analysis_dimensions=DEFAULT_DIMENSIONS.copy(),
                seed_urls=[str(request.url)],
                auto_discover_sources=False,
            ),
            [document],
        )
        status.status = "completed"
        status.current_stage = "QuickExtract"
        status.finished_at = datetime.now(timezone.utc)
        status.metrics = RunMetrics(sources_fetched=1 if document.ok else 0, evidence_count=len(evidence))
        await self.runs.save_status(status)
        return status, evidence

    def agent_context(self, run_id: str) -> AgentContext:
        return AgentContext(
            run_id=run_id,
            run_dir=self.runs.run_dir(run_id),
            settings=self.settings,
            llm=self.llm,
            trace=lambda node, phase, status, message, payload=None: self.trace(
                run_id,
                node,
                phase,
                status,
                message,
                payload,
            ),
        )

    async def save_candidates(self, run_id: str, candidates: list[SourceCandidate]) -> None:
        path = self.runs.run_dir(run_id) / "sources" / "_index.jsonl"
        existing_urls: set[str] = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if '"url"' in line:
                    match = re.search(r'"url"\s*:\s*"([^"]+)"', line)
                    if match:
                        existing_urls.add(match.group(1))
        for candidate in candidates:
            if candidate.url in existing_urls:
                continue
            existing_urls.add(candidate.url)
            await append_jsonl(path, candidate)

    async def save_document(
        self, run_id: str, candidate: SourceCandidate, page: RawPage
    ) -> SourceDocument:
        source_id = stable_id("src", page.final_url or candidate.url)
        document = SourceDocument(
            source_id=source_id,
            run_id=run_id,
            url=page.final_url or candidate.url,
            title=page.title or candidate.title or candidate.url,
            content=page.content,
            excerpt=page.content[:500],
            source_type=candidate.source_type or classify_source(candidate.url, page.title),
            http_status=page.http_status,
            content_hash=page.content_hash,
            fetched_at=page.fetched_at,
            parser=page.parser,
            provider=candidate.source_provider,
            query=candidate.query,
            content_source=candidate.content_source or "local_fetch",
            source_score=candidate.score,
            relevance_score=candidate.relevance_score,
            relevance_label=candidate.relevance_label,
            rejection_reason=candidate.rejection_reason,
            source_subtype=candidate.source_subtype,
            source_subtype_reason=candidate.source_subtype_reason,
            embedding_score=candidate.embedding_score,
            lexical_score=candidate.lexical_score,
            reranker_score=candidate.reranker_score,
            ok=page.ok,
            error=page.error,
            published_at=candidate.published_at,
            date_source=candidate.date_source,
        )
        run_dir = self.runs.run_dir(run_id)
        await atomic_write_json(run_dir / "sources" / f"{source_id}.json", document)
        await write_text(run_dir / "documents" / f"{source_id}.txt", page.content)
        await append_jsonl(run_dir / "documents" / "_index.jsonl", document)
        return document

    async def materialize_search_documents(
        self,
        run_id: str,
        candidates: list[SourceCandidate],
    ) -> list[SourceDocument]:
        """Persist provider-returned content as SourceDocument without local web fetching."""

        documents: list[SourceDocument] = []
        run_dir = self.runs.run_dir(run_id)
        seen_source_ids: set[str] = set()
        for candidate in candidates:
            content = (candidate.content or candidate.snippet or "").strip()
            if len(content) < 80:
                continue
            source_id = stable_id("src", f"{candidate.url}:{candidate.source_provider}:{candidate.query}")
            if source_id in seen_source_ids:
                continue
            seen_source_ids.add(source_id)
            content_hash = hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()
            document = SourceDocument(
                source_id=source_id,
                run_id=run_id,
                url=candidate.url,
                title=candidate.title or urlparse(candidate.url).netloc or candidate.url,
                content=content,
                excerpt=content[:500],
                source_type=candidate.source_type or classify_source(candidate.url, candidate.title),
                http_status=200,
                content_hash=content_hash,
                fetched_at=datetime.now(timezone.utc),
                parser=candidate.content_source or "provider_content",
                provider=candidate.source_provider,
                query=candidate.query,
                content_source=candidate.content_source or "provider_content",
                source_score=candidate.score,
                relevance_score=candidate.relevance_score,
                relevance_label=candidate.relevance_label,
                rejection_reason=candidate.rejection_reason,
                source_subtype=candidate.source_subtype,
                source_subtype_reason=candidate.source_subtype_reason,
                embedding_score=candidate.embedding_score,
                lexical_score=candidate.lexical_score,
                reranker_score=candidate.reranker_score,
                ok=True,
                error=None,
                published_at=candidate.published_at,
                date_source=candidate.date_source,
            )
            await atomic_write_json(run_dir / "sources" / f"{source_id}.json", document)
            await write_text(run_dir / "documents" / f"{source_id}.txt", document.content)
            await append_jsonl(run_dir / "documents" / "_index.jsonl", document)
            documents.append(document)
        if documents:
            await self.trace(
                run_id,
                "SourceResearchAgent",
                "progress",
                "content_ready",
                f"已直接保存 {len(documents)} 篇搜索引擎返回正文，跳过本地抓取",
                {
                    "count": len(documents),
                    "providers": dict(Counter(doc.provider for doc in documents if doc.provider)),
                    "results": [
                        {
                            "title": doc.title,
                            "url": doc.url,
                            "provider": doc.provider,
                            "content_source": doc.content_source,
                            "relevance_score": doc.relevance_score,
                            "relevance_label": doc.relevance_label,
                            "excerpt": doc.excerpt,
                        }
                        for doc in documents[:20]
                    ],
                },
            )
        return documents

    async def extract_and_save_evidence(
        self,
        run_id: str,
        request: ResearchRequest,
        plan_or_documents,
        maybe_documents: list[SourceDocument] | None = None,
        *,
        existing_evidence: list[Evidence] | None = None,
    ) -> list[Evidence]:
        if maybe_documents is None:
            plan = ResearchPlan(
                research_goal=request.research_goal,
                papers=request.seed_papers,
                dimensions=request.analysis_dimensions or DEFAULT_DIMENSIONS,
            )
            documents = plan_or_documents
        else:
            plan = plan_or_documents
            documents = maybe_documents
        repo = EvidenceRepository(self.runs.run_dir(run_id))
        evidence_items: list[Evidence] = []
        dimensions = plan.dimensions or request.analysis_dimensions or DEFAULT_DIMENSIONS
        entities = unique_strings(request.seed_papers)
        agent = EvidenceStructuringAgent(self.agent_context(run_id))
        coverage_counts = evidence_coverage_counts(existing_evidence or [], entities, dimensions)
        skipped_documents = 0
        rejected_documents = 0
        candidates_for_extraction: list[tuple[SourceDocument, list[Evidence]]] = []
        for document in documents:
            if not document.ok or len(document.content) < 80:
                continue
            if (
                document.relevance_label == "reject"
                or (
                    self.settings.scholar_source_gate_enabled
                    and document.relevance_score < self.settings.scholar_min_source_relevance
                )
            ):
                rejected_documents += 1
                await append_jsonl(
                    self.runs.run_dir(run_id) / "sources" / "rejected_sources.jsonl",
                    {
                        "source_id": document.source_id,
                        "url": document.url,
                        "title": document.title,
                        "query": document.query,
                        "provider": document.provider,
                        "relevance_score": document.relevance_score,
                        "relevance_label": document.relevance_label,
                        "rejection_reason": document.rejection_reason or "below evidence extraction relevance threshold",
                        "source_subtype": document.source_subtype,
                        "source_subtype_reason": document.source_subtype_reason,
                    },
                )
                continue
            doc_pairs = infer_document_pairs(document, entities, dimensions)
            if doc_pairs and all(evidence_cell_sufficient(coverage_counts.get(pair, 0)) for pair in doc_pairs):
                skipped_documents += 1
                continue
            deterministic = extract_evidence_from_document(run_id, document, dimensions, entities)
            candidates_for_extraction.append((document, deterministic))

        semaphore = asyncio.Semaphore(max(1, self.settings.cg_evidence_llm_parallelism))

        async def extract_one(document: SourceDocument, deterministic: list[Evidence]) -> tuple[SourceDocument, list[Evidence]]:
            async with semaphore:
                try:
                    extracted = await agent.extract(
                        request,
                        document,
                        dimensions,
                        entities,
                        deterministic,
                        stable_id,
                        source_weight,
                    )
                    return document, extracted
                except Exception as exc:
                    await self.trace(
                        run_id,
                        "EvidenceStructuringAgent",
                        "progress",
                        "extract_fallback",
                        f"抽取「{document.title}」失败，已使用规则抽取兜底",
                        {
                            "url": document.url,
                            "title": document.title,
                            "error": str(exc),
                            "fallback_count": len(deterministic),
                        },
                    )
                    return document, deterministic

        total_candidates = len(candidates_for_extraction)
        completed_documents = 0
        existing_count = len(existing_evidence or [])
        parallelism = max(1, self.settings.cg_evidence_llm_parallelism)
        await self.trace(
            run_id,
            "EvidenceStructuringAgent",
            "progress",
            "extract_started",
            f"开始抽取 Evidence：{total_candidates} 篇候选文档，LLM 并行度 {parallelism}",
            {"document_count": total_candidates, "parallelism": parallelism},
        )

        stop_heartbeat = asyncio.Event()

        async def heartbeat() -> None:
            while not stop_heartbeat.is_set():
                try:
                    await asyncio.wait_for(stop_heartbeat.wait(), timeout=60)
                except asyncio.TimeoutError:
                    await self.trace(
                        run_id,
                        "EvidenceStructuringAgent",
                        "progress",
                        "extract_heartbeat",
                        f"Evidence 抽取进行中：已完成 {completed_documents}/{total_candidates} 篇，已保存 {len(evidence_items)} 条",
                        {
                            "completed_documents": completed_documents,
                            "total_documents": total_candidates,
                            "evidence_saved": len(evidence_items),
                        },
                    )

        heartbeat_task = asyncio.create_task(heartbeat())
        extraction_tasks = [
            asyncio.create_task(extract_one(document, deterministic))
            for document, deterministic in candidates_for_extraction
        ]
        try:
            pending_tasks = set(extraction_tasks)
            while pending_tasks:
                await self.check_stop(run_id)
                done_tasks, pending_tasks = await asyncio.wait(
                    pending_tasks,
                    timeout=1.0,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done_tasks:
                    continue
                for extraction_task in done_tasks:
                    document, extracted = await extraction_task
                    completed_documents += 1
                    saved_for_document = 0
                    for item in extracted:
                        if item.paper and item.dimension:
                            key = (item.paper, item.dimension)
                            if evidence_cell_sufficient(coverage_counts.get(key, 0)):
                                continue
                            coverage_counts[key] = coverage_counts.get(key, 0) + 1
                        await repo.save(item)
                        evidence_items.append(item)
                        saved_for_document += 1
                    await self.trace(
                        run_id,
                        "EvidenceStructuringAgent",
                        "progress",
                        "extracted",
                        f"从「{document.title}」抽取 {saved_for_document} 条 Evidence（{completed_documents}/{total_candidates}）",
                        {
                            "url": document.url,
                            "title": document.title,
                            "count": saved_for_document,
                            "provider": document.provider,
                            "content_source": document.content_source,
                            "completed_documents": completed_documents,
                            "total_documents": total_candidates,
                        },
                    )
                    if completed_documents == 1 or completed_documents % 5 == 0 or completed_documents == total_candidates:
                        current_status = await self.runs.load_status(run_id)
                        if current_status.status == "running":
                            current_status.metrics.evidence_count = existing_count + len(evidence_items)
                            await self.runs.save_status(current_status)
        finally:
            for task in extraction_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*extraction_tasks, return_exceptions=True)
            stop_heartbeat.set()
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)

        if skipped_documents:
            await self.trace(
                run_id,
                "EvidenceStructuringAgent",
                "progress",
                "skipped_sufficient",
                f"跳过 {skipped_documents} 篇文档：对应推理模式×维度 Evidence 已较充分",
                {"count": skipped_documents},
            )
        if rejected_documents:
            await self.trace(
                run_id,
                "EvidenceStructuringAgent",
                "progress",
                "skipped_low_relevance",
                f"跳过 {rejected_documents} 篇低相关文档：未进入 Evidence 抽取",
                {"count": rejected_documents},
            )
        return evidence_items

    async def generate_claims(self, run_id: str, evidence: list[Evidence], request: ResearchRequest | None = None) -> list[Claim]:
        clusters = build_evidence_clusters(evidence)
        deterministic_claims = build_claims_from_clusters(run_id, clusters, evidence)
        representative_evidence = select_cluster_representative_evidence(evidence, clusters)
        await self.trace(
            run_id,
            "AnalysisAndReviewAgent",
            "progress",
            "evidence_clustered",
            f"构建 {len(clusters)} 个 Evidence cluster，用 {len(representative_evidence)} 条代表性 Evidence 生成 Claim",
            {
                "cluster_count": len(clusters),
                "representative_evidence_count": len(representative_evidence),
                "cluster_claim_count": len(deterministic_claims),
            },
        )
        claims = await AnalysisAndReviewAgent(self.agent_context(run_id)).generate(
            representative_evidence,
            deterministic_claims,
            request,
        )
        claims = prepare_claims_for_review(claims, evidence)
        return claims

    async def red_team(
        self, run_id: str, claims: list[Claim], evidence: list[Evidence]
    ) -> list[Claim]:
        red_team_ctx = AgentContext(
            run_id=run_id,
            run_dir=self.runs.run_dir(run_id),
            settings=self.settings,
            llm=self.red_team_llm,
            trace=lambda node, phase, status, message, payload=None: self.trace(
                run_id, node, phase, status, message, payload
            ),
        )
        reviewed = await self._run_red_team_review(
            run_id,
            AnalysisAndReviewAgent(red_team_ctx).review(claims, evidence),
            claims,
            evidence,
        )
        reviewed = apply_claim_discipline(reviewed, evidence)
        run_dir = self.runs.run_dir(run_id)
        await atomic_write_json(run_dir / "exports" / "claim_backlog.json", claim_backlog_rows(reviewed, evidence))
        for claim in reviewed:
            await atomic_write_json(run_dir / "claims" / f"{claim.claim_id}.json", claim)
            await append_jsonl(run_dir / "claims" / "_index.jsonl", claim)
        return reviewed

    async def _run_red_team_review(
        self,
        run_id: str,
        value: Awaitable[list[Claim]],
        claims: list[Claim],
        evidence: list[Evidence],
    ) -> list[Claim]:
        timeout_seconds = self._llm_step_timeout_seconds()
        await self.trace(
            run_id,
            "AnalysisAndReviewAgent",
            "progress",
            "red_team_review_started",
            "开始执行 Red Team claim 审查",
            {"claim_count": len(claims), "timeout_seconds": timeout_seconds},
        )

        async def heartbeat() -> None:
            elapsed = 0
            while True:
                await asyncio.sleep(60)
                elapsed += 60
                await self.trace(
                    run_id,
                    "AnalysisAndReviewAgent",
                    "progress",
                    "red_team_review_heartbeat",
                    f"Red Team 审查进行中：已等待 {elapsed}s",
                    {
                        "claim_count": len(claims),
                        "elapsed_seconds": elapsed,
                        "timeout_seconds": timeout_seconds,
                    },
                )

        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            reviewed = await asyncio.wait_for(value, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            await self.trace(
                run_id,
                "AnalysisAndReviewAgent",
                "error",
                "red_team_review_timeout",
                f"Red Team 审查超过 {timeout_seconds:.0f}s",
                {"claim_count": len(claims), "timeout_seconds": timeout_seconds},
            )
            reviewed = deterministic_red_team(claims, evidence)
            await self.trace(
                run_id,
                "AnalysisAndReviewAgent",
                "progress",
                "red_team_review_fallback",
                "Red Team LLM 超时，已使用确定性审查 fallback",
                {"claim_count": len(reviewed), "reason": "timeout"},
            )
        except Exception as exc:
            await self.trace(
                run_id,
                "AnalysisAndReviewAgent",
                "error",
                "red_team_review_failed",
                "Red Team 审查失败",
                {"claim_count": len(claims), "error_type": exc.__class__.__name__, "error": str(exc)[:500]},
            )
            reviewed = deterministic_red_team(claims, evidence)
            await self.trace(
                run_id,
                "AnalysisAndReviewAgent",
                "progress",
                "red_team_review_fallback",
                "Red Team LLM 失败，已使用确定性审查 fallback",
                {"claim_count": len(reviewed), "reason": exc.__class__.__name__},
            )
        finally:
            heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task

        await self.trace(
            run_id,
            "AnalysisAndReviewAgent",
            "progress",
            "red_team_review_completed",
            "Red Team claim 审查完成",
            {
                "claim_count": len(reviewed),
                "verified": len([claim for claim in reviewed if claim.verification_status == "verified"]),
                "challenged": len([claim for claim in reviewed if claim.verification_status != "verified"]),
            },
        )
        return reviewed

    async def save_artifacts(self, run_id: str, artifacts: dict) -> None:
        run_dir = self.runs.run_dir(run_id)
        matrix = artifacts["matrix"]
        recommendations = artifacts["recommendations"]
        evidence_graph = artifacts["evidence_graph"]
        observability = artifacts["observability"]
        evidence_clusters = artifacts.get("evidence_clusters", [])
        await atomic_write_json(run_dir / "exports" / "matrix.json", matrix)
        await write_text(run_dir / "exports" / "matrix.csv", build_matrix_csv(matrix))
        await atomic_write_json(run_dir / "exports" / "recommendations.json", recommendations)
        await write_text(run_dir / "exports" / "recommendations.md", build_recommendations_markdown(recommendations))
        await atomic_write_json(run_dir / "exports" / "observability.json", observability)
        await atomic_write_json(run_dir / "exports" / "evidence_graph.json", evidence_graph)
        await atomic_write_json(run_dir / "exports" / "evidence_clusters.json", evidence_clusters)

    async def write_report(
        self,
        run_id: str,
        request: ResearchRequest,
        evidence: list[Evidence],
        claims: list[Claim],
        metrics: RunMetrics,
        matrix: PaperPatternMatrix,
        recommendations: list[OpportunityRecommendation],
        observability: ObservabilitySnapshot,
    ) -> list[dict[str, str]]:
        composer = ReportComposerAgent(self.agent_context(run_id))
        fallbacks: list[dict[str, str]] = []
        run_dir = self.runs.run_dir(run_id)

        async def run_section(
            section: str,
            label: str,
            value: Awaitable[str],
            fallback: str,
        ) -> str:
            try:
                return await self._run_report_step(run_id, section, label, value)
            except Exception as exc:
                reason = {
                    "section": section,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc)[:500],
                }
                fallbacks.append(reason)
                await self.trace(
                    run_id,
                    "ReportComposerAgent",
                    "progress",
                    "report_step_fallback",
                    f"{label} 生成失败，已使用确定性 fallback",
                    reason,
                )
                return fallback

        report = await run_section(
            "report",
            "完整报告",
            composer.write(
                request,
                evidence,
                claims,
                metrics,
                {
                    "matrix": matrix,
                    "recommendations": recommendations,
                    "observability": observability,
                },
            ),
            build_analysis_report(request, evidence, claims, metrics, matrix, recommendations, observability),
        )
        await write_text(run_dir / "reports" / "report.md", report)

        executive_summary = await run_section(
            "executive_summary",
            "执行摘要",
            composer.write_executive_summary(
                request,
                evidence,
                claims,
                metrics,
                matrix,
                recommendations,
                observability,
            ),
            build_executive_summary(request, metrics, observability, recommendations, claims),
        )
        await write_text(run_dir / "reports" / "executive_summary.md", executive_summary)

        methodology = await run_section(
            "methodology",
            "方法说明",
            composer.write_methodology(
                request,
                evidence,
                claims,
                metrics,
                matrix,
                observability,
            ),
            build_methodology(request, observability),
        )
        await write_text(run_dir / "reports" / "methodology.md", methodology)
        if fallbacks:
            await atomic_write_json(run_dir / "reports" / "report_fallbacks.json", fallbacks)
        await atomic_write_json(
            run_dir / "reports" / "report.json",
            {
                "run_id": run_id,
                "request": request,
                "metrics": metrics,
                "claims": claims,
                "matrix": matrix,
                "recommendations": recommendations,
                "observability": observability,
                "evidence_ids": [ev.evidence_id for ev in evidence],
            },
        )
        await write_text(
            run_dir / "exports" / "evidence_matrix.csv",
            build_evidence_csv(evidence, {ev.evidence_id: ev for ev in evidence}),
        )
        await self.trace(
            run_id,
            "ReportComposerAgent",
            "progress",
            "reports_written",
            "报告文件已写入 reports/ 和 exports/",
            {
                "fallback_sections": [item["section"] for item in fallbacks],
                "files": [
                    "reports/report.md",
                    "reports/executive_summary.md",
                    "reports/methodology.md",
                    "reports/report.json",
                    "exports/evidence_matrix.csv",
                ],
            },
        )
        return fallbacks

    def _report_step_timeout_seconds(self) -> float:
        return self._llm_step_timeout_seconds()

    def _llm_step_timeout_seconds(self) -> float:
        return float(max(60, min(600, self.settings.cg_llm_timeout_seconds + 60)))

    async def _run_report_step(
        self,
        run_id: str,
        section: str,
        label: str,
        value: Awaitable[str],
    ) -> str:
        timeout_seconds = self._report_step_timeout_seconds()
        await self.trace(
            run_id,
            "ReportComposerAgent",
            "progress",
            "report_step_started",
            f"开始生成{label}",
            {"section": section, "timeout_seconds": timeout_seconds},
        )

        async def heartbeat() -> None:
            elapsed = 0
            while True:
                await asyncio.sleep(60)
                elapsed += 60
                await self.trace(
                    run_id,
                    "ReportComposerAgent",
                    "progress",
                    "report_step_heartbeat",
                    f"{label}生成中：已等待 {elapsed}s",
                    {
                        "section": section,
                        "elapsed_seconds": elapsed,
                        "timeout_seconds": timeout_seconds,
                    },
                )

        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            result = await asyncio.wait_for(value, timeout=timeout_seconds)
        except asyncio.TimeoutError as exc:
            await self.trace(
                run_id,
                "ReportComposerAgent",
                "error",
                "report_step_timeout",
                f"{label}生成超过 {timeout_seconds:.0f}s",
                {"section": section, "timeout_seconds": timeout_seconds},
            )
            raise TimeoutError(f"{section} generation exceeded {timeout_seconds:.0f}s") from exc
        except Exception as exc:
            await self.trace(
                run_id,
                "ReportComposerAgent",
                "error",
                "report_step_failed",
                f"{label}生成失败",
                {"section": section, "error_type": exc.__class__.__name__, "error": str(exc)[:500]},
            )
            raise
        finally:
            heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task

        await self.trace(
            run_id,
            "ReportComposerAgent",
            "progress",
            "report_step_completed",
            f"{label}生成完成",
            {"section": section, "characters": len(result or "")},
        )
        return result

    @asynccontextmanager
    async def node(
        self, run_id: str, status: RunStatus, node: str, message: str
    ) -> AsyncIterator[None]:
        status.current_stage = node
        status.node_status[node] = "running"
        await self.runs.save_status(status)
        await self.trace(run_id, node, "start", "running", message)
        try:
            yield
        except RunStopped:
            status.node_status[node] = "stopped"
            await self.runs.save_status(status)
            await self.trace(run_id, node, "complete", "stopped", "Stopped by user")
            raise
        except Exception as exc:
            status.node_status[node] = "failed"
            await self.runs.save_status(status)
            await self.trace(run_id, node, "error", "failed", str(exc))
            raise
        else:
            status.node_status[node] = "completed"
            await self.runs.save_status(status)
            await self.trace(run_id, node, "complete", "completed", message)

    async def trace(
        self,
        run_id: str,
        node: str,
        phase: str,
        status: str,
        message: str,
        payload: dict | None = None,
    ) -> None:
        event = TraceEvent(
            event_id=f"evt_{uuid4().hex[:12]}",
            run_id=run_id,
            timestamp=datetime.now(timezone.utc),
            node=node,
            phase=phase,  # type: ignore[arg-type]
            status=status,
            message=message,
            payload=payload or {},
        )
        await append_jsonl(self.runs.run_dir(run_id) / "trace" / "events.jsonl", event)

def build_dag_snapshot(plan: ResearchPlan) -> dict:
    requested = set(plan.required_agents or [])
    nodes = [agent for agent in RESEARCH_AGENT_FLOW if not requested or agent in requested]
    for agent in RESEARCH_AGENT_FLOW:
        if agent not in nodes:
            nodes.append(agent)
    return {
        "nodes": [
            {"id": node, "status": "planned", "skills": AGENT_SKILLS.get(node, [])}
            for node in nodes
        ],
        "edges": [{"from": nodes[i], "to": nodes[i + 1]} for i in range(len(nodes) - 1)],
        "queries": plan.queries,
        "source_tasks": [task.model_dump(mode="json") for task in plan.source_tasks],
        "quality_rules": plan.quality_rules,
        "planned_by": plan.planned_by,
    }


def build_paper_pattern_matrix(
    evidence: list[Evidence], papers: list[str], dimensions: list[str]
) -> PaperPatternMatrix:
    now = datetime.now(timezone.utc)
    evidence_by_cell: dict[tuple[str, str], list[Evidence]] = defaultdict(list)
    for ev in evidence:
        paper = ev.paper or detect_entity(ev.fact, ev.source_title, ev.source_url, papers)
        if not paper and len(papers) == 1:
            paper = papers[0]
        if paper and ev.dimension in dimensions:
            evidence_by_cell[(paper, ev.dimension)].append(ev)

    cells: list[MatrixCell] = []
    for paper in papers:
        for dimension in dimensions:
            items = sorted(evidence_by_cell.get((paper, dimension), []), key=lambda ev: ev.confidence, reverse=True)
            confidence = average([ev.confidence for ev in items[:5]])
            status = cell_status(len(items), confidence)
            cells.append(
                MatrixCell(
                    paper=paper,
                    dimension=dimension,
                    dimension_label=DIMENSION_LABELS.get(dimension, dimension),
                    summary=cell_summary(paper, dimension, items),
                    evidence_count=len(items),
                    confidence=round(confidence, 3),
                    source_types=sorted({ev.source_type for ev in items}),
                    evidence_ids=[ev.evidence_id for ev in items[:6]],
                    status=status,
                )
            )

    profiles: list[PaperProfile] = []
    for paper in papers:
        paper_items = [
            ev
            for ev in evidence
            if (ev.paper or detect_entity(ev.fact, ev.source_title, ev.source_url, papers)) == paper
        ]
        strong_dimensions = [
            DIMENSION_LABELS.get(cell.dimension, cell.dimension)
            for cell in cells
            if cell.paper == paper and cell.status in {"strong", "partial"}
        ]
        weak_dimensions = [
            DIMENSION_LABELS.get(cell.dimension, cell.dimension)
            for cell in cells
            if cell.paper == paper and cell.status in {"weak", "unknown"}
        ]
        profiles.append(
            PaperProfile(
                paper=paper,
                summary=profile_summary(paper, paper_items, strong_dimensions, weak_dimensions),
                evidence_count=len(paper_items),
                source_count=len({ev.source_url for ev in paper_items}),
                average_confidence=round(average([ev.confidence for ev in paper_items]), 3),
                strongest_dimensions=strong_dimensions[:4],
                weak_or_unknown_dimensions=weak_dimensions[:4],
                evidence_ids=[ev.evidence_id for ev in sorted(paper_items, key=lambda item: item.confidence, reverse=True)[:10]],
            )
        )

    coverage_by_paper: dict[str, float] = {}
    for paper in papers:
        paper_cells = [cell for cell in cells if cell.paper == paper]
        coverage_by_paper[paper] = round(
            sum(coverage_points(cell) for cell in paper_cells) / max(1, len(paper_cells)),
            3,
        )

    coverage_by_dimension: dict[str, float] = {}
    for dimension in dimensions:
        dimension_cells = [cell for cell in cells if cell.dimension == dimension]
        coverage_by_dimension[dimension] = round(
            sum(coverage_points(cell) for cell in dimension_cells) / max(1, len(dimension_cells)),
            3,
        )

    return PaperPatternMatrix(
        generated_at=now,
        papers=papers,
        dimensions=dimensions,
        dimension_labels={dimension: DIMENSION_LABELS.get(dimension, dimension) for dimension in dimensions},
        cells=cells,
        profiles=profiles,
        coverage_by_paper=coverage_by_paper,
        coverage_by_dimension=coverage_by_dimension,
    )


def build_claim_from_support(
    run_id: str,
    dimension: str,
    dimension_label: str,
    claim_text: str,
    supporting: list[Evidence],
    reasoning: str,
) -> Claim:
    source_types = sorted({ev.source_type for ev in supporting})
    source_count = len({ev.source_url for ev in supporting})
    paper_count = len({paper_key(ev) for ev in supporting})
    source_role_counts = claim_source_role_counts(supporting)
    source_role_paper_counts = claim_source_role_paper_counts(supporting)
    claim_type = "comparative" if paper_count >= 2 else "single_paper_observation"
    support_level = evidence_support_level(len(supporting), paper_count)
    confidence = min(0.92, average([ev.confidence for ev in supporting]) * min(1.0, 0.65 + len(supporting) * 0.08))
    return Claim(
        claim_id=stable_id("claim", f"{run_id}:{dimension}:{claim_text}:{[ev.evidence_id for ev in supporting]}"),
        run_id=run_id,
        dimension=dimension,
        dimension_label=dimension_label,
        claim=claim_text[:700],
        supporting_evidence_ids=[ev.evidence_id for ev in supporting],
        confidence=round(confidence, 3),
        risk_level="low" if len(supporting) >= 3 and source_count >= 2 else "medium",
        reasoning_summary=f"{reasoning} 来源类型包括：{'、'.join(source_types) or '未知'}。",
        claim_type=claim_type,
        source_paper_count=paper_count,
        evidence_support_level=support_level,
        supporting_source_subtypes=sorted(source_role_counts),
        supporting_source_subtype_counts=source_role_counts,
        supporting_source_subtype_paper_counts=source_role_paper_counts,
        verification_status="draft",
    )


CLUSTER_HINTS: dict[str, tuple[str, ...]] = {
    "evaluation_benchmark": (
        "benchmark",
        "dataset",
        "evaluation",
        "metric",
        "contamination",
        "leaderboard",
    ),
    "verification_reward": (
        "verifier",
        "verification",
        "reward model",
        "process reward",
        "proof",
        "grader",
    ),
    "search_inference_control": (
        "self-consistency",
        "tree search",
        "best-of",
        "test-time",
        "inference-time",
        "decoding",
        "search",
    ),
    "data_generation_training": (
        "synthetic data",
        "instruction tuning",
        "fine-tuning",
        "training data",
        "distillation",
        "data generation",
    ),
    "modular_system_pipeline": (
        "pipeline",
        "module",
        "planner",
        "solver",
        "agent",
        "tool",
        "retrieval",
    ),
    "representation_formalization": (
        "representation",
        "symbolic",
        "formal",
        "equation",
        "program",
        "natural language",
    ),
    "uncertainty_probabilistic": (
        "uncertainty",
        "probabilistic",
        "bayesian",
        "calibration",
        "confidence",
    ),
    "efficiency_scaling": (
        "efficient",
        "efficiency",
        "compression",
        "token",
        "cost",
        "scaling",
    ),
    "robustness_adversarial": (
        "adversarial",
        "robustness",
        "bias",
        "attack",
        "failure",
    ),
    "treatment_effect_estimation_protocol": (
        "treatment effect",
        "heterogeneous treatment",
        "causal effect",
        "counterfactual outcome",
        "counterfactual outcomes",
        "counterfactual regression",
        "potential outcome",
        "potential outcomes",
        "individual treatment",
    ),
    "counterfactual_identifiability_assumption": (
        "identifiability",
        "identifiable",
        "ignorability",
        "unobserved confounding",
        "confounding",
        "structural causal",
        "acyclicity",
        "assumption",
    ),
    "kgqa_semantic_parsing_pipeline": (
        "knowledge graph question answering",
        "kgqa",
        "graph question answering",
        "multi-hop question answering",
        "semantic parsing",
        "logical form",
        "executable query",
        "query graph",
    ),
    "path_composition_graph_reasoning": (
        "multi-hop path",
        "graph path",
        "path reasoning",
        "path search",
        "graph traversal",
        "knowledge graph reasoning",
        "hop reasoning",
        "multi-hop reasoning",
    ),
    "math_proof_verification_protocol": (
        "formal proof",
        "formal proofs",
        "formal verification",
        "proof verification",
        "proof checking",
        "proof correctness",
        "proof assistant",
        "theorem proving",
        "natural language math proof",
        "natural language math proofs",
        "ground-truth proof",
        "lean proof",
        "lean",
    ),
    "math_benchmark_problem_protocol": (
        "gsm8k",
        "math benchmark",
        "mathematical benchmark",
        "word problem",
        "word problems",
        "grade school arithmetic",
        "olympiad-level",
        "olympiad",
    ),
    "scientific_discovery_workflow_protocol": (
        "scientific discovery",
        "discovery process",
        "data-driven discovery",
        "hypothesis generation",
        "hypothesis",
        "experimental design",
        "experiment planning",
        "research workflow",
        "complete research workflow",
        "scientific workflow",
        "discovery pipeline",
        "biological discovery pipeline",
        "scientific partners",
        "autonomous scientific",
        "multi-agent architecture",
        "agent plans",
        "agentic",
    ),
}

CLUSTER_LABELS: dict[str, str] = {
    "evaluation_benchmark": "evaluation and benchmark design",
    "verification_reward": "verification and reward modeling",
    "search_inference_control": "search and inference-time control",
    "data_generation_training": "data generation and training",
    "modular_system_pipeline": "modular system pipeline",
    "representation_formalization": "representation and formalization",
    "uncertainty_probabilistic": "uncertainty and probabilistic modeling",
    "efficiency_scaling": "efficiency and scaling",
    "robustness_adversarial": "robustness and adversarial analysis",
    "treatment_effect_estimation_protocol": "treatment-effect estimation protocol",
    "counterfactual_identifiability_assumption": "counterfactual identifiability and assumptions",
    "kgqa_semantic_parsing_pipeline": "KGQA and semantic parsing pipeline",
    "path_composition_graph_reasoning": "path composition and graph traversal",
    "math_proof_verification_protocol": "mathematical proof verification protocol",
    "math_benchmark_problem_protocol": "mathematical benchmark/problem protocol",
    "scientific_discovery_workflow_protocol": "scientific discovery workflow protocol",
}

CLUSTER_DIMENSION_OVERRIDES: dict[str, str] = {
    "math_proof_verification_protocol": "formal_proof_symbolic_reasoning",
    "math_benchmark_problem_protocol": "math_benchmark_evaluation",
    "scientific_discovery_workflow_protocol": "lab_workflow_reasoning",
}

MATH_CLUSTER_KEYS = {
    "math_proof_verification_protocol",
    "math_benchmark_problem_protocol",
}
SCIENTIFIC_CLUSTER_KEYS = {"scientific_discovery_workflow_protocol"}

MATH_DIMENSIONS = {
    "math_benchmark_evaluation",
    "formal_proof_symbolic_reasoning",
    "program_tool_augmented_solving",
    "self_consistency_search_verification",
    "natural_language_to_formal_math",
    "math_error_diagnosis",
}

SCIENTIFIC_DIMENSIONS = {
    "scientific_problem_benchmarking",
    "tool_augmented_scientific_reasoning",
    "domain_grounding_verification",
    "scientific_error_analysis",
    "lab_workflow_reasoning",
    "multimodal_scientific_reasoning",
}

CLUSTER_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "large",
    "language",
    "models",
    "model",
    "paper",
    "method",
    "methods",
    "reasoning",
    "using",
    "based",
    "through",
    "approach",
    "task",
    "tasks",
}

SYNTHESIS_SOURCE_ROLES = {
    "core_kg_rag_method",
    "benchmark_or_analysis",
    "core_scientific_reasoning_method",
    "scientific_reasoning_benchmark",
    "scientific_equation_discovery",
    "scientific_discovery_agent",
    "domain_science_reasoning",
    "core_mathematical_reasoning",
    "math_reasoning_benchmark",
    "formal_math_proving",
    "math_training_data",
    "math_reasoning_method",
    "core_llm_causal_reasoning",
    "causal_reasoning_benchmark",
    "llm_counterfactual_reasoning",
    "core_counterfactual_inference",
    "treatment_effect_estimation",
    "counterfactual_explanation_or_fairness",
    "core_multi_hop_graph_reasoning",
    "kgqa_or_graph_reasoning",
    "graph_reasoning_benchmark",
}

SOURCE_ROLE_LABELS = {
    "core_kg_rag_method": "core KG-RAG method",
    "benchmark_or_analysis": "benchmark/analysis",
    "core_scientific_reasoning_method": "core scientific reasoning method",
    "scientific_reasoning_benchmark": "scientific reasoning benchmark",
    "scientific_equation_discovery": "scientific equation discovery",
    "scientific_discovery_agent": "scientific discovery agent/workflow",
    "domain_science_reasoning": "domain scientific reasoning",
    "scientific_reasoning_adjacent": "scientific reasoning adjacent",
    "core_mathematical_reasoning": "core mathematical reasoning",
    "math_reasoning_benchmark": "mathematical reasoning benchmark",
    "formal_math_proving": "formal math proving",
    "math_training_data": "math training/data scaling",
    "math_reasoning_method": "mathematical reasoning method",
    "mathematical_reasoning_adjacent": "mathematical reasoning adjacent",
    "core_llm_causal_reasoning": "core LLM causal reasoning",
    "causal_reasoning_benchmark": "LLM causal reasoning benchmark",
    "llm_counterfactual_reasoning": "LLM counterfactual reasoning",
    "llm_training_testbed_adjacent": "LLM training/testbed adjacent",
    "core_counterfactual_inference": "core counterfactual inference",
    "treatment_effect_estimation": "treatment-effect estimation",
    "counterfactual_explanation_or_fairness": "counterfactual explanation/fairness",
    "causal_inference_adjacent": "causal inference adjacent",
    "core_multi_hop_graph_reasoning": "core multi-hop graph reasoning",
    "kgqa_or_graph_reasoning": "KGQA/graph reasoning",
    "graph_reasoning_benchmark": "graph reasoning benchmark",
    "graph_retrieval_rag_adjacent": "graph retrieval/RAG adjacent",
}


def build_evidence_clusters(evidence: list[Evidence], claims: list[Claim] | None = None) -> list[EvidenceCluster]:
    grouped: dict[tuple[str, str], list[Evidence]] = defaultdict(list)
    labels: dict[tuple[str, str], str] = {}
    for ev in evidence:
        key, label = evidence_cluster_key(ev)
        dimension = ev.dimension or "other"
        if key in MATH_CLUSTER_KEYS and ev.dimension in MATH_DIMENSIONS:
            dimension = CLUSTER_DIMENSION_OVERRIDES[key]
        elif key in SCIENTIFIC_CLUSTER_KEYS and ev.dimension in SCIENTIFIC_DIMENSIONS:
            dimension = CLUSTER_DIMENSION_OVERRIDES[key]
        grouped[(dimension, key)].append(ev)
        labels[(dimension, key)] = label

    claims = claims or []
    claim_cluster_ids: dict[str, str] = {}
    for claim in claims:
        cluster_id = claim.evidence_cluster_id or infer_claim_cluster_id(claim, grouped)
        if cluster_id:
            claim_cluster_ids[claim.claim_id] = cluster_id

    clusters: list[EvidenceCluster] = []
    for (dimension, key), items in grouped.items():
        sorted_items = sorted(items, key=lambda ev: ev.confidence, reverse=True)
        papers = unique_strings([paper_key(ev) for ev in sorted_items])
        cluster_id = stable_id("cluster", f"{dimension}:{key}")
        verified_claim_ids = [
            claim.claim_id
            for claim in claims
            if claim_cluster_ids.get(claim.claim_id) == cluster_id
            and claim.verification_status == "verified"
            and claim.risk_level != "high"
        ]
        challenged_claim_ids = [
            claim.claim_id
            for claim in claims
            if claim_cluster_ids.get(claim.claim_id) == cluster_id
            and claim.claim_id not in verified_claim_ids
        ]
        status = (
            "verified"
            if verified_claim_ids
            else "single_paper"
            if len(papers) < 2
            else "backlog"
            if challenged_claim_ids
            else "candidate"
        )
        clusters.append(
            EvidenceCluster(
                cluster_id=cluster_id,
                dimension=dimension,
                dimension_label=DIMENSION_LABELS.get(dimension, dimension),
                label=labels.get((dimension, key), key),
                summary=cluster_summary(sorted_items),
                mechanism=most_common_nonempty([ev.mechanism for ev in sorted_items]),
                bottleneck=most_common_nonempty([ev.bottleneck for ev in sorted_items]),
                evidence_ids=[ev.evidence_id for ev in sorted_items],
                papers=papers,
                evidence_count=len(sorted_items),
                independent_paper_count=len(papers),
                average_confidence=round(average([ev.confidence for ev in sorted_items]), 3),
                verified_claim_ids=verified_claim_ids,
                challenged_claim_ids=challenged_claim_ids,
                status=status,
            )
        )
    return sorted(
        clusters,
        key=lambda cluster: (
            cluster.status != "verified",
            -cluster.independent_paper_count,
            -cluster.evidence_count,
            cluster.dimension,
            cluster.label,
        ),
    )


def build_claims_from_clusters(run_id: str, clusters: list[EvidenceCluster], evidence: list[Evidence], limit: int = 32) -> list[Claim]:
    by_id = {ev.evidence_id: ev for ev in evidence}
    claims: list[Claim] = []
    candidate_clusters = sorted(
        clusters,
        key=lambda cluster: (
            -cluster.independent_paper_count,
            -cluster.evidence_count,
            -cluster.average_confidence,
            cluster.dimension,
        ),
    )

    atomic_limit = max(1, limit // 2)
    for cluster in candidate_clusters:
        for supporting in select_atomic_observation_supports(cluster, by_id):
            paper = paper_key(supporting[0])
            best_fact = evidence_observation_text(supporting[0], 260)
            claim_text = (
                f"作为单论文观察，{paper} 在{cluster.dimension_label}维度显示："
                f"{best_fact}"
            )
            reasoning = (
                f"该 observation 只绑定 {paper} 的 {len(supporting)} 条 Evidence，"
                "不作为跨论文领域趋势。"
            )
            claim = build_claim_from_support(
                run_id,
                cluster.dimension,
                cluster.dimension_label,
                claim_text,
                supporting,
                reasoning,
            )
            claim.claim_type = "single_paper_observation"
            claim.evidence_support_level = "single_paper"
            claim.source_paper_count = 1
            claim.evidence_cluster_id = cluster.cluster_id
            claim.evidence_cluster_label = cluster.label
            claims.append(claim)
            if len(claims) >= atomic_limit:
                break
        if len(claims) >= atomic_limit:
            break

    for cluster in candidate_clusters:
        supporting, source_role = select_synthesis_support(cluster, by_id)
        if len(supporting) < 2:
            continue
        paper_count = len({paper_key(ev) for ev in supporting})
        if paper_count < 2:
            continue
        source_role_label = SOURCE_ROLE_LABELS.get(source_role, source_role.replace("_", " "))
        if cluster_supports_report_ready_synthesis(cluster, supporting):
            claim_text = report_ready_synthesis_text(cluster, source_role_label, paper_count)
        else:
            paper_facts = cluster_paper_facts(supporting)
            claim_text = (
                f"作为跨论文对比性观察，{source_role_label} 来源在{cluster.dimension_label}维度"
                f"呈现出若干互补切入点：{paper_facts}。"
                "这只说明当前样本内的机制差异，不单独构成领域趋势。"
            )
        reasoning = (
            f"该 synthesis 只合并同一 source role（{source_role_label}）内的证据，"
            f"覆盖 {paper_count} 篇独立论文。"
        )
        claim = build_claim_from_support(
            run_id,
            cluster.dimension,
            cluster.dimension_label,
            claim_text,
            supporting,
            reasoning,
        )
        claim.evidence_cluster_id = cluster.cluster_id
        claim.evidence_cluster_label = cluster.label
        claims.append(claim)
        if len(claims) >= limit:
            break
    if len(claims) < limit:
        for cluster in candidate_clusters:
            supporting, source_roles = select_cross_role_synthesis_support(cluster, by_id)
            if len(supporting) < 4 or len(source_roles) < 2:
                continue
            paper_count = len({paper_key(ev) for ev in supporting})
            if paper_count < 4:
                continue
            claim_text = cross_role_report_ready_synthesis_text(cluster, supporting, source_roles)
            reasoning = (
                "该 cross-role synthesis 只比较不同 source role 在同一证据轴上的分工，"
                f"覆盖 {paper_count} 篇独立论文。"
            )
            claim = build_claim_from_support(
                run_id,
                cluster.dimension,
                cluster.dimension_label,
                claim_text,
                supporting,
                reasoning,
            )
            claim.claim_type = "cross_role_contrast"
            claim.evidence_cluster_id = cluster.cluster_id
            claim.evidence_cluster_label = cluster.label
            claims.append(claim)
            if len(claims) >= limit:
                break
    return claims


def select_cluster_representative_evidence(
    evidence: list[Evidence], clusters: list[EvidenceCluster], max_evidence: int = 120
) -> list[Evidence]:
    if len(evidence) <= max_evidence:
        return evidence
    by_id = {ev.evidence_id: ev for ev in evidence}
    selected: list[Evidence] = []
    seen: set[str] = set()
    ranked_clusters = sorted(
        clusters,
        key=lambda cluster: (
            -cluster.independent_paper_count,
            -cluster.evidence_count,
            -cluster.average_confidence,
        ),
    )
    for cluster in ranked_clusters:
        per_cluster_limit = 8 if cluster.independent_paper_count >= 2 else 4
        for ev_id in cluster.evidence_ids[:per_cluster_limit]:
            if ev_id in seen or ev_id not in by_id:
                continue
            selected.append(by_id[ev_id])
            seen.add(ev_id)
            if len(selected) >= max_evidence:
                return selected
    return selected or evidence[:max_evidence]


def cluster_supports_report_ready_synthesis(cluster: EvidenceCluster, supporting: list[Evidence]) -> bool:
    if cluster.label.startswith("lexical_"):
        return False
    if len({paper_key(ev) for ev in supporting}) < 4:
        return False
    if len(supporting) < 4:
        return False
    if average([ev.confidence for ev in supporting]) < 0.68:
        return False
    return True


def report_ready_synthesis_text(cluster: EvidenceCluster, source_role_label: str, paper_count: int) -> str:
    axis = cluster_axis_phrase(cluster.label)
    return (
        f"在{cluster.dimension_label}维度，{paper_count} 篇 {source_role_label} 来源"
        f"共同围绕“{cluster.label}”这一分析轴组织证据，主要体现在{axis}。"
        f"该结论限定于同一 source role 内的多论文证据，可作为报告主体中的范围限定综合结论。"
    )


def cross_role_report_ready_synthesis_text(
    cluster: EvidenceCluster,
    supporting: list[Evidence],
    source_roles: list[str],
) -> str:
    role_counts = claim_source_role_paper_counts(supporting)
    role_parts = []
    for role in source_roles:
        phrase = source_role_contribution_phrase(role, cluster.label)
        separator = " " if phrase[:1].isascii() else ""
        role_parts.append(
            f"{source_role_label(role)} 来源（{role_counts.get(role, 0)} 篇）侧重"
            f"{separator}{phrase}"
        )
    axis = cluster_axis_phrase(cluster.label)
    return (
        f"在{cluster.dimension_label}维度，{len({paper_key(ev) for ev in supporting})} 篇论文"
        f"共同围绕“{cluster.label}”形成来源角色分工对照："
        f"{'；'.join(role_parts)}。"
        f"该结论限定于{axis}这一证据轴，可作为报告主体中的范围限定综合结论。"
    )


def source_role_contribution_phrase(role: str, cluster_label: str) -> str:
    if role == "scientific_reasoning_benchmark":
        return "任务定义、数据集构造和评测协议"
    if role == "scientific_discovery_agent":
        return "agent workflow、工具调用和研究流程编排"
    if role == "core_scientific_reasoning_method":
        return "科学推理机制、领域知识整合和方法设计"
    if role == "scientific_equation_discovery":
        return "方程发现、符号回归和科学规律抽取"
    if role == "core_counterfactual_inference":
        return "结构因果假设、反事实查询和可识别性条件"
    if role == "treatment_effect_estimation":
        return "处理效应目标、潜在结果建模和异质效应估计"
    if role == "counterfactual_explanation_or_fairness":
        return "反事实解释、recourse 和公平性约束"
    if role == "math_reasoning_benchmark":
        return "数学题集、评测协议和错误模式测量"
    if role == "formal_math_proving":
        return "形式证明、证明检查和符号验证"
    if role == "math_training_data":
        return "数学训练数据、合成题和扩展调优"
    if role == "kgqa_or_graph_reasoning":
        return "KGQA、语义解析和可执行查询构造"
    if role == "core_multi_hop_graph_reasoning":
        return "多跳路径组合、图遍历和关系链推理"
    if role == "graph_reasoning_benchmark":
        return "图推理任务定义、数据集和评测指标"
    if role == "core_kg_rag_method":
        return "KG-RAG 架构、图检索和生成增强机制"
    if role == "benchmark_or_analysis":
        return "基准分析、适用条件和对照评测"
    return f"{cluster_axis_phrase(cluster_label)}相关证据"


def cluster_axis_phrase(label: str) -> str:
    return {
        "evaluation and benchmark design": "任务定义、数据集构造、评测指标或对照协议",
        "verification and reward modeling": "验证器、奖励模型、过程监督或证明检查",
        "search and inference-time control": "搜索、推理时控制、自一致性或测试时计算",
        "data generation and training": "数据生成、训练语料、指令调优或蒸馏策略",
        "modular system pipeline": "模块组合、工具调用、agent workflow 或系统管线",
        "representation and formalization": "符号表征、形式化表示、程序化描述或自然语言到结构化对象的转换",
        "uncertainty and probabilistic modeling": "不确定性、概率假设、校准或分布建模",
        "efficiency and scaling": "效率、扩展性、成本控制或近似工程",
        "robustness and adversarial analysis": "鲁棒性、失败模式、对抗扰动或偏差分析",
        "treatment-effect estimation protocol": "处理效应目标、反事实结果建模、异质效应评估或潜在结果假设",
        "counterfactual identifiability and assumptions": "可识别性条件、结构因果假设、混杂处理或反事实查询约束",
        "KGQA and semantic parsing pipeline": "KGQA 任务形式、语义解析、逻辑形式生成或可执行查询构造",
        "path composition and graph traversal": "多跳路径组合、图遍历、关系链搜索或路径级推理控制",
        "mathematical proof verification protocol": "形式证明、证明检查、Lean/证明助手验证或证明正确性评估",
        "mathematical benchmark/problem protocol": "数学题集、GSM8K/MATH 类基准、奥赛题或 word-problem 评测协议",
        "scientific discovery workflow protocol": "假设生成、实验设计、数据驱动发现、agent 协作或研究流程编排",
    }.get(label, label)


def evidence_cluster_key(ev: Evidence) -> tuple[str, str]:
    text = " ".join([ev.mechanism, ev.bottleneck, ev.fact, ev.quote, ev.paper or "", ev.source_title]).lower()
    preferred_keys = {
        "inference_time_control": ["search_inference_control", "verification_reward"],
        "data_evaluation_engineering": ["evaluation_benchmark"],
        "principled_probabilistic_modeling": ["uncertainty_probabilistic"],
        "approximation_engineering": ["efficiency_scaling"],
        "treatment_effect_estimation": [
            "treatment_effect_estimation_protocol",
            "evaluation_benchmark",
            "counterfactual_identifiability_assumption",
        ],
        "counterfactual_benchmarking": [
            "evaluation_benchmark",
            "treatment_effect_estimation_protocol",
        ],
        "core_counterfactual_inference": [
            "counterfactual_identifiability_assumption",
            "treatment_effect_estimation_protocol",
        ],
        "identifiability_assumption_sensitivity": [
            "counterfactual_identifiability_assumption",
        ],
        "temporal_counterfactual_estimation": [
            "treatment_effect_estimation_protocol",
            "counterfactual_identifiability_assumption",
        ],
        "graph_reasoning_benchmark": [
            "evaluation_benchmark",
            "kgqa_semantic_parsing_pipeline",
            "path_composition_graph_reasoning",
        ],
        "kgqa_or_graph_reasoning": [
            "kgqa_semantic_parsing_pipeline",
            "path_composition_graph_reasoning",
            "modular_system_pipeline",
        ],
        "core_multi_hop_graph_reasoning": [
            "path_composition_graph_reasoning",
            "kgqa_semantic_parsing_pipeline",
            "modular_system_pipeline",
        ],
        "semantic_parsing_grounding": [
            "kgqa_semantic_parsing_pipeline",
            "representation_formalization",
        ],
        "path_composition_reasoning": [
            "path_composition_graph_reasoning",
            "kgqa_semantic_parsing_pipeline",
        ],
        "graph_retrieval_boundary": [
            "path_composition_graph_reasoning",
            "kgqa_semantic_parsing_pipeline",
            "modular_system_pipeline",
        ],
        "math_benchmark_evaluation": [
            "math_benchmark_problem_protocol",
            "math_proof_verification_protocol",
            "evaluation_benchmark",
        ],
        "formal_proof_symbolic_reasoning": [
            "math_proof_verification_protocol",
            "representation_formalization",
            "verification_reward",
        ],
        "program_tool_augmented_solving": [
            "math_proof_verification_protocol",
            "modular_system_pipeline",
            "data_generation_training",
        ],
        "self_consistency_search_verification": [
            "math_proof_verification_protocol",
            "search_inference_control",
            "verification_reward",
        ],
        "natural_language_to_formal_math": [
            "math_proof_verification_protocol",
            "representation_formalization",
        ],
        "scientific_problem_benchmarking": [
            "evaluation_benchmark",
            "scientific_discovery_workflow_protocol",
        ],
        "tool_augmented_scientific_reasoning": [
            "scientific_discovery_workflow_protocol",
            "search_inference_control",
            "modular_system_pipeline",
        ],
        "lab_workflow_reasoning": [
            "scientific_discovery_workflow_protocol",
            "modular_system_pipeline",
            "evaluation_benchmark",
        ],
        "domain_grounding_verification": [
            "scientific_discovery_workflow_protocol",
            "verification_reward",
            "evaluation_benchmark",
        ],
        "multimodal_scientific_reasoning": [
            "scientific_discovery_workflow_protocol",
            "modular_system_pipeline",
        ],
    }.get(ev.dimension, [])
    for key in preferred_keys:
        hints = CLUSTER_HINTS.get(key, ())
        if any(hint in text for hint in hints):
            return key, CLUSTER_LABELS[key]
    for key, hints in CLUSTER_HINTS.items():
        if key in MATH_CLUSTER_KEYS and ev.dimension not in MATH_DIMENSIONS:
            continue
        if key in SCIENTIFIC_CLUSTER_KEYS and ev.dimension not in SCIENTIFIC_DIMENSIONS:
            continue
        if any(hint in text for hint in hints):
            return key, CLUSTER_LABELS[key]
    tokens = [
        token
        for token in re.findall(r"[a-z][a-z0-9-]{3,}", text)
        if token not in CLUSTER_STOPWORDS
    ]
    token_counts = Counter(tokens)
    label_tokens = [token for token, _ in token_counts.most_common(3)]
    if label_tokens:
        label = " / ".join(label_tokens)
        return f"lexical_{'_'.join(label_tokens)}", label
    return "general", "general evidence pattern"


def infer_claim_cluster_id(claim: Claim, grouped: dict[tuple[str, str], list[Evidence]]) -> str:
    evidence_ids = set(claim.supporting_evidence_ids)
    best_cluster_id = ""
    best_overlap = 0
    for (dimension, key), items in grouped.items():
        if dimension != claim.dimension:
            continue
        overlap = len(evidence_ids.intersection({ev.evidence_id for ev in items}))
        if overlap > best_overlap:
            best_overlap = overlap
            best_cluster_id = stable_id("cluster", f"{dimension}:{key}")
    return best_cluster_id


def select_cluster_support(cluster: EvidenceCluster, by_id: dict[str, Evidence]) -> list[Evidence]:
    items = [by_id[ev_id] for ev_id in cluster.evidence_ids if ev_id in by_id]
    by_paper: dict[str, list[Evidence]] = defaultdict(list)
    for item in items:
        by_paper[paper_key(item)].append(item)
    supporting: list[Evidence] = []
    for paper in sorted(by_paper, key=lambda name: len(by_paper[name]), reverse=True):
        supporting.extend(sorted(by_paper[paper], key=lambda ev: ev.confidence, reverse=True)[:2])
        if len(supporting) >= 8:
            break
    return supporting[:8]


def select_atomic_observation_supports(cluster: EvidenceCluster, by_id: dict[str, Evidence]) -> list[list[Evidence]]:
    items = [by_id[ev_id] for ev_id in cluster.evidence_ids if ev_id in by_id]
    by_paper: dict[str, list[Evidence]] = defaultdict(list)
    for item in items:
        by_paper[paper_key(item)].append(item)
    supports: list[list[Evidence]] = []
    for paper in sorted(by_paper, key=lambda name: (len(by_paper[name]), average([ev.confidence for ev in by_paper[name]])), reverse=True):
        paper_items = sorted(by_paper[paper], key=lambda ev: ev.confidence, reverse=True)
        if len(paper_items) < 2:
            continue
        supports.append(paper_items[: min(3, len(paper_items))])
        if len(supports) >= 2:
            break
    return supports


def select_synthesis_support(cluster: EvidenceCluster, by_id: dict[str, Evidence]) -> tuple[list[Evidence], str]:
    items = [by_id[ev_id] for ev_id in cluster.evidence_ids if ev_id in by_id]
    by_role: dict[str, list[Evidence]] = defaultdict(list)
    for item in items:
        role = item.source_subtype or "unclassified"
        if role not in SYNTHESIS_SOURCE_ROLES:
            continue
        by_role[role].append(item)
    if not by_role:
        return [], ""
    ranked_roles = sorted(
        by_role,
        key=lambda role: (
            len({paper_key(ev) for ev in by_role[role]}),
            len(by_role[role]),
            average([ev.confidence for ev in by_role[role]]),
        ),
        reverse=True,
    )
    for role in ranked_roles:
        role_items = by_role[role]
        if len({paper_key(ev) for ev in role_items}) < 2:
            continue
        role_cluster = EvidenceCluster(
            cluster_id=cluster.cluster_id,
            dimension=cluster.dimension,
            dimension_label=cluster.dimension_label,
            label=cluster.label,
            summary=cluster.summary,
            mechanism=cluster.mechanism,
            bottleneck=cluster.bottleneck,
            evidence_ids=[ev.evidence_id for ev in sorted(role_items, key=lambda item: item.confidence, reverse=True)],
            papers=unique_strings([paper_key(ev) for ev in role_items]),
            evidence_count=len(role_items),
            independent_paper_count=len({paper_key(ev) for ev in role_items}),
            average_confidence=round(average([ev.confidence for ev in role_items]), 3),
        )
        return select_cluster_support(role_cluster, by_id), role
    return [], ""


def select_cross_role_synthesis_support(
    cluster: EvidenceCluster,
    by_id: dict[str, Evidence],
) -> tuple[list[Evidence], list[str]]:
    if cluster.label.startswith("lexical_"):
        return [], []
    items = [by_id[ev_id] for ev_id in cluster.evidence_ids if ev_id in by_id]
    by_role: dict[str, list[Evidence]] = defaultdict(list)
    for item in items:
        role = item.source_subtype or "unclassified"
        if role not in SYNTHESIS_SOURCE_ROLES:
            continue
        by_role[role].append(item)
    eligible_roles = [
        role
        for role, role_items in by_role.items()
        if len({paper_key(ev) for ev in role_items}) >= 2
    ]
    if len(eligible_roles) < 2:
        return [], []
    ranked_roles = sorted(
        eligible_roles,
        key=lambda role: (
            len({paper_key(ev) for ev in by_role[role]}),
            len(by_role[role]),
            average([ev.confidence for ev in by_role[role]]),
        ),
        reverse=True,
    )[:3]
    supporting: list[Evidence] = []
    for role in ranked_roles:
        role_items = sorted(by_role[role], key=lambda ev: ev.confidence, reverse=True)
        by_paper: dict[str, list[Evidence]] = defaultdict(list)
        for item in role_items:
            by_paper[paper_key(item)].append(item)
        for paper in sorted(
            by_paper,
            key=lambda name: average([ev.confidence for ev in by_paper[name]]),
            reverse=True,
        )[:2]:
            supporting.append(sorted(by_paper[paper], key=lambda ev: ev.confidence, reverse=True)[0])
    if not cluster_supports_report_ready_synthesis(cluster, supporting):
        return [], []
    return supporting, ranked_roles


def cluster_summary(items: list[Evidence]) -> str:
    if not items:
        return ""
    papers = unique_strings([paper_key(ev) for ev in items])[:3]
    facts = "；".join(evidence_observation_text(ev, 120) for ev in items[:3])
    return f"覆盖 {len(items)} 条 Evidence、{len(papers)} 篇代表论文（{'; '.join(papers)}）：{facts}"


def cluster_paper_facts(items: list[Evidence]) -> str:
    by_paper: dict[str, list[Evidence]] = defaultdict(list)
    for item in items:
        by_paper[paper_key(item)].append(item)
    parts: list[str] = []
    for paper, paper_items in list(by_paper.items())[:3]:
        best = sorted(paper_items, key=lambda ev: ev.confidence, reverse=True)[0]
        parts.append(f"{compact_paper_label(paper)}：{evidence_observation_text(best, 80)}")
    return "；".join(parts)


def evidence_observation_text(evidence: Evidence, max_len: int = 120) -> str:
    text = evidence.quote or evidence.content or evidence.fact
    return compact_fact(text, max_len)


def compact_paper_label(text: str, max_len: int = 82) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned if len(cleaned) <= max_len else cleaned[: max_len - 3] + "..."


def most_common_nonempty(values: list[str]) -> str:
    cleaned = [value.strip() for value in values if value and value.strip()]
    if not cleaned:
        return ""
    return Counter(cleaned).most_common(1)[0][0]


def paper_key(evidence: Evidence) -> str:
    return (evidence.paper or evidence.source_title or evidence.source_url or evidence.url or "unknown").strip()


def evidence_support_level(evidence_count: int, paper_count: int) -> str:
    if paper_count >= 2 and evidence_count >= 2:
        return "strong"
    if paper_count == 1 and evidence_count >= 2:
        return "single_paper"
    if evidence_count == 1:
        return "insufficient"
    return "weak"


def claim_source_role_counts(supporting: list[Evidence]) -> dict[str, int]:
    counts = Counter(ev.source_subtype or "unclassified" for ev in supporting)
    return dict(sorted(counts.items()))


def claim_source_role_paper_counts(supporting: list[Evidence]) -> dict[str, int]:
    by_role: dict[str, set[str]] = defaultdict(set)
    for ev in supporting:
        by_role[ev.source_subtype or "unclassified"].add(paper_key(ev))
    return {role: len(papers) for role, papers in sorted(by_role.items())}


def source_role_label(role: str) -> str:
    return SOURCE_ROLE_LABELS.get(role, role.replace("_", " "))


def role_contrast_text(claim: Claim, supporting: list[Evidence], reportable_roles: set[str]) -> str:
    parts: list[str] = []
    for role in sorted(reportable_roles):
        role_items = sorted(
            [ev for ev in supporting if ev.source_subtype == role],
            key=lambda ev: ev.confidence,
            reverse=True,
        )
        if not role_items:
            continue
        papers = unique_strings([paper_key(ev) for ev in role_items])[:2]
        best_fact = compact_fact(role_items[0].fact, 110)
        paper_text = f"（{'; '.join(papers)}）" if papers else ""
        parts.append(f"{source_role_label(role)}{paper_text}侧重{best_fact}")
    if not parts:
        return claim.claim
    return (
        f"作为跨 source-role 对照，{claim.dimension_label}维度中，"
        f"{'；'.join(parts)}。"
        "这只比较 benchmark/core/testbed 等来源角色的证据分工，不作为统一领域趋势。"
    )


def prepare_claims_for_review(claims: list[Claim], evidence: list[Evidence]) -> list[Claim]:
    by_id = {ev.evidence_id: ev for ev in evidence}
    for claim in claims:
        supporting = [by_id[ev_id] for ev_id in claim.supporting_evidence_ids if ev_id in by_id]
        paper_count = len({paper_key(ev) for ev in supporting})
        source_roles = {ev.source_subtype for ev in supporting if ev.source_subtype}
        reportable_roles = {role for role in source_roles if role in SYNTHESIS_SOURCE_ROLES}
        source_role_counts = claim_source_role_counts(supporting)
        source_role_paper_counts = claim_source_role_paper_counts(supporting)
        claim.source_paper_count = paper_count
        claim.evidence_support_level = evidence_support_level(len(supporting), paper_count)
        claim.supporting_source_subtypes = sorted(source_role_counts)
        claim.supporting_source_subtype_counts = source_role_counts
        claim.supporting_source_subtype_paper_counts = source_role_paper_counts
        if len(supporting) < 2:
            claim.claim_type = "backlog"
            claim.backlog_reason = "single_evidence_claim"
            claim.confidence = min(claim.confidence, 0.55)
            continue
        if source_roles and not source_roles.issubset(SYNTHESIS_SOURCE_ROLES):
            claim.claim_type = "backlog"
            claim.backlog_reason = "unsupported_source_role"
            claim.confidence = min(claim.confidence, 0.55)
            continue
        if paper_count >= 2 and len(reportable_roles) > 1:
            claim.claim_type = "cross_role_contrast"
            scoped_cross_role = (
                "来源角色分工对照" in claim.claim
                and "范围限定综合结论" in claim.claim
            )
            if not scoped_cross_role:
                claim.claim = role_contrast_text(claim, supporting, reportable_roles)
            claim.final_wording = claim.claim
            claim.reasoning_summary = (
                f"{claim.reasoning_summary} Role-aware rewrite: this claim contrasts "
                f"{', '.join(source_role_label(role) for role in sorted(reportable_roles))} "
                "instead of treating mixed source roles as one field-wide trend."
            ).strip()
            if not scoped_cross_role:
                claim.confidence = min(claim.confidence, 0.72)
                claim.risk_level = "medium"
            claim.backlog_reason = ""
        elif paper_count >= 2:
            claim.claim_type = "comparative"
        else:
            claim.claim_type = "single_paper_observation"
    return claims


def apply_claim_discipline(claims: list[Claim], evidence: list[Evidence]) -> list[Claim]:
    by_id = {ev.evidence_id: ev for ev in evidence}
    for claim in claims:
        supporting = [by_id[ev_id] for ev_id in claim.supporting_evidence_ids if ev_id in by_id]
        paper_count = len({paper_key(ev) for ev in supporting})
        source_roles = {ev.source_subtype for ev in supporting if ev.source_subtype}
        reportable_roles = {role for role in source_roles if role in SYNTHESIS_SOURCE_ROLES}
        source_role_counts = claim_source_role_counts(supporting)
        source_role_paper_counts = claim_source_role_paper_counts(supporting)
        claim.source_paper_count = paper_count
        claim.evidence_support_level = evidence_support_level(len(supporting), paper_count)
        claim.supporting_source_subtypes = sorted(source_role_counts)
        claim.supporting_source_subtype_counts = source_role_counts
        claim.supporting_source_subtype_paper_counts = source_role_paper_counts
        if claim.claim_type == "cross_role_contrast" and paper_count >= 2:
            claim.claim_type = "cross_role_contrast"
        elif paper_count >= 2:
            claim.claim_type = "comparative"
        elif len(supporting) >= 2:
            claim.claim_type = "single_paper_observation"
        else:
            claim.claim_type = "backlog"

        backlog_reason = ""
        if len(supporting) < 2:
            backlog_reason = "single_evidence_claim"
        elif claim.claim_type == "comparative" and paper_count < 2:
            backlog_reason = "comparative_claim_without_independent_papers"
        elif source_roles and not source_roles.issubset(SYNTHESIS_SOURCE_ROLES):
            backlog_reason = "unsupported_source_role"
        elif claim.claim_type == "cross_role_contrast" and len(reportable_roles) < 2:
            backlog_reason = "invalid_cross_role_contrast"
        elif claim.claim_type == "comparative" and len(reportable_roles) != 1:
            backlog_reason = "mixed_source_roles"
        elif claim.verification_status in {"needs_evidence", "challenged", "rejected"}:
            backlog_reason = f"red_team_{claim.verification_status}"

        if backlog_reason:
            claim.backlog_reason = backlog_reason
            if claim.verification_status == "verified":
                claim.verification_status = "needs_evidence"
            if claim.risk_level == "low":
                claim.risk_level = "medium"
        elif claim.claim_type == "single_paper_observation" and claim.verification_status == "verified":
            claim.final_wording = claim.final_wording or f"作为单论文观察，{claim.claim}"
    return claims


def claim_report_ready_reason(claim: Claim) -> str:
    """Return an empty string only when a claim is strong enough for report body use."""
    if claim.verification_status != "verified":
        return f"red_team_{claim.verification_status}"
    if claim.risk_level == "high":
        return "high_risk"
    if claim.backlog_reason:
        return claim.backlog_reason
    if claim.claim_type not in {"comparative", "cross_role_contrast"}:
        return "not_synthesis_claim"
    if claim.source_paper_count < 2:
        return "insufficient_independent_papers"
    if len(claim.supporting_evidence_ids) < 2:
        return "insufficient_evidence"
    if claim.evidence_support_level != "strong":
        return "weak_evidence_support"
    source_roles = set(claim.supporting_source_subtypes or [])
    if not source_roles:
        return "missing_source_role_counts"
    if not source_roles.issubset(SYNTHESIS_SOURCE_ROLES):
        return "unsupported_source_role"
    role_paper_counts = getattr(claim, "supporting_source_subtype_paper_counts", {}) or {}
    if not role_paper_counts and len(source_roles) == 1:
        role = next(iter(source_roles))
        role_paper_counts = {role: claim.source_paper_count}
    if claim.claim_type == "cross_role_contrast":
        reportable_roles = {role for role in source_roles if role in SYNTHESIS_SOURCE_ROLES}
        if len(reportable_roles) < 2:
            return "invalid_cross_role_contrast"
        if not role_paper_counts:
            return "missing_source_role_paper_counts"
        if not source_roles.issubset(set(role_paper_counts)):
            return "missing_source_role_paper_counts"
        if claim.source_paper_count < 4:
            return "insufficient_cross_role_total_papers"
        if any(role_paper_counts.get(role, 0) < 2 for role in reportable_roles):
            return "insufficient_cross_role_role_papers"
    text = best_claim_text(claim)
    if any(
        marker in text
        for marker in (
            "当前样本内",
            "不单独构成领域趋势",
            "不作为统一领域趋势",
            "作为单论文观察",
        )
    ):
        return "sample_limited_observation"
    lowered = text.lower()
    if "..." in text or "…" in text:
        return "truncated_or_fragmentary_wording"
    if re.search(
        r"(submitted to|available at https?://|https?://github|code and dataset|figure [0-9]|section [0-9])",
        lowered,
    ):
        return "artifact_or_citation_fragment"
    return ""


def is_report_ready_claim(claim: Claim) -> bool:
    return claim_report_ready_reason(claim) == ""


def claim_backlog_rows(claims: list[Claim], evidence: list[Evidence]) -> list[dict[str, Any]]:
    by_id = {ev.evidence_id: ev for ev in evidence}
    rows: list[dict[str, Any]] = []
    for claim in claims:
        report_ready_rejection_reason = claim_report_ready_reason(claim)
        if not report_ready_rejection_reason:
            continue
        supporting = [by_id[ev_id] for ev_id in claim.supporting_evidence_ids if ev_id in by_id]
        rows.append(
            {
                "claim_id": claim.claim_id,
                "dimension": claim.dimension,
                "dimension_label": claim.dimension_label,
                "claim_type": claim.claim_type,
                "evidence_cluster_id": claim.evidence_cluster_id,
                "evidence_cluster_label": claim.evidence_cluster_label,
                "verification_status": claim.verification_status,
                "risk_level": claim.risk_level,
                "backlog_reason": claim.backlog_reason
                or report_ready_rejection_reason
                or f"red_team_{claim.verification_status}",
                "report_ready_rejection_reason": report_ready_rejection_reason,
                "source_paper_count": claim.source_paper_count,
                "supporting_evidence_count": len(claim.supporting_evidence_ids),
                "supporting_papers": sorted({paper_key(ev) for ev in supporting}),
                "supporting_source_subtypes": claim.supporting_source_subtypes,
                "supporting_source_subtype_counts": claim.supporting_source_subtype_counts,
                "supporting_source_subtype_paper_counts": claim.supporting_source_subtype_paper_counts,
                "claim": claim.final_wording or claim.claim,
                "red_team_notes": [note.model_dump(mode="json") for note in claim.red_team_notes],
                "supporting_evidence_ids": claim.supporting_evidence_ids,
            }
        )
    return rows


def build_recommendations(
    run_id: str,
    request: ResearchRequest,
    claims: list[Claim],
    evidence: list[Evidence],
    matrix: PaperPatternMatrix,
) -> list[OpportunityRecommendation]:
    recommendations: list[OpportunityRecommendation] = []
    by_dimension: dict[str, list[Claim]] = defaultdict(list)
    for claim in sorted(claims, key=lambda item: item.confidence, reverse=True):
        by_dimension[claim.dimension].append(claim)

    weak_dimensions = sorted(
        matrix.coverage_by_dimension,
        key=lambda dimension: matrix.coverage_by_dimension.get(dimension, 0),
    )
    target_profile = next(
        (profile for profile in matrix.profiles if profile.paper == request.target_topic),
        None,
    )
    target_weak = target_profile.weak_or_unknown_dimensions if target_profile else []

    strategic_claims = [
        claim
        for claim in sorted(claims, key=lambda item: item.confidence, reverse=True)
        if is_report_ready_claim(claim)
    ][:3]
    if strategic_claims:
        evidence_ids = unique_strings(
            [ev_id for claim in strategic_claims[:3] for ev_id in claim.supporting_evidence_ids]
        )
        lead_claim = strategic_claims[0]
        recommendations.append(
            OpportunityRecommendation(
                recommendation_id=stable_id("rec", f"{run_id}:strategy:{evidence_ids}"),
                title=grounded_opportunity_title(lead_claim),
                recommendation=grounded_opportunity_text(request, lead_claim),
                priority="high",
                target_audience="researcher",
                rationale=grounded_backing_text(lead_claim),
                expected_value="把综述观察转化为带证据边界的可检验研究假设，降低选题阶段的主观跳跃。",
                based_on_claim_ids=[claim.claim_id for claim in strategic_claims[:3]],
                evidence_ids=evidence_ids[:8],
                next_steps=[
                    f"围绕 {DIMENSION_LABELS.get(lead_claim.dimension, lead_claim.dimension)} 写出一个可检验 research question",
                    "把支持这一综合判断的论文按 source role 分组，分别设计实验协议、数据需求和失败判据",
                    "优先补充反例或边界论文，确认该机会不是检索样本造成的局部现象",
                ],
                risks=[
                    "该机会来自本地论文证据综合，不等于已经验证的新方法贡献。",
                    "正式写作前仍需人工复核原文、补充反例和实验可行性分析。",
                ],
                confidence=round(average([claim.confidence for claim in strategic_claims[:3]]), 3),
            )
        )
    elif claims:
        weak_claims = [
            claim
            for claim in sorted(claims, key=lambda item: item.confidence, reverse=True)
            if not is_report_ready_claim(claim) and claim.risk_level != "high"
        ][:5]
        evidence_ids = unique_strings(
            [ev_id for claim in weak_claims for ev_id in claim.supporting_evidence_ids]
        )
        recommendations.append(
            OpportunityRecommendation(
                recommendation_id=stable_id("rec", f"{run_id}:claim-review:{len(weak_claims)}"),
                title="先完成高风险结论的补证闭环",
                recommendation=(
                    f"本轮 {request.target_topic} 尚缺少足够的 Red Team verified claim；"
                    "下一步应优先围绕被挑战但仍有信息量的观察补充交叉证据，再决定是否写成研究机会。"
                ),
                priority="high",
                target_audience="researcher",
                rationale="当前建议不应建立在 challenged 或 needs_evidence claim 上，否则会把检索噪声包装成研究结论。",
                expected_value="把报告从一次性文本产物转为可审计的研究证据工作台，降低误导性结论风险。",
                based_on_claim_ids=[],
                evidence_ids=evidence_ids[:8],
                next_steps=[
                    "按 Red Team 风险类型拆分补证队列",
                    "为每个待验证观察补充至少两个独立论文来源",
                    "重新运行 Claim review 后再生成正式研究机会建议",
                ],
                risks=["如果不补证，报告只能作为探索性阅读线索，不能作为论文选题依据。"],
                confidence=0.48,
            )
        )

    weak_dimensions = [
        dimension
        for dimension in weak_dimensions
        if matrix.coverage_by_dimension.get(dimension, 0) < 0.58
    ]
    if weak_dimensions:
        dimension = weak_dimensions[0]
        evidence_ids = [
            ev.evidence_id
            for ev in sorted(evidence, key=lambda item: item.confidence, reverse=True)
            if ev.dimension == dimension
        ][:6]
        recommendations.append(
            OpportunityRecommendation(
                recommendation_id=stable_id("rec", f"{run_id}:coverage:{dimension}"),
                title=f"优先补强{DIMENSION_LABELS.get(dimension, dimension)}证据覆盖",
                recommendation=(
                    f"下一轮研究应优先补充 {DIMENSION_LABELS.get(dimension, dimension)} 相关来源，"
                    "特别是代表性论文、最新进展和方法创新论文。"
                ),
                priority="medium" if evidence_ids else "high",
                target_audience="researcher",
                rationale=f"该维度当前覆盖度为 {matrix.coverage_by_dimension.get(dimension, 0):.0%}，仍有补证空间。",
                expected_value="提升报告可信度，并降低 Red Team 对关键结论的挑战率。",
                evidence_ids=evidence_ids,
                next_steps=[
                    f"新增 3-5 个 {DIMENSION_LABELS.get(dimension, dimension)} 高质量来源",
                    "重新运行 EvidenceStructuringAgent 与 AnalysisAndReviewAgent",
                    "观察质量门禁中的覆盖度是否提升",
                ],
                risks=["补采集可能受网站反爬或搜索质量影响。"],
                confidence=0.68 if evidence_ids else 0.52,
            )
        )

    if target_weak:
        recommendations.append(
            OpportunityRecommendation(
                recommendation_id=stable_id("rec", f"{run_id}:target-position:{target_weak}"),
                title=f"明确 {request.target_topic} 的研究空白",
                recommendation=(
                    f"当前资料中 {request.target_topic} 在 {'、'.join(target_weak[:3])} 上的可核验证据偏少，"
                    "建议补充相关论文和最新进展。"
                ),
                priority="high",
                target_audience="researcher",
                rationale="论文分析不仅看研究方向是否有进展，也看公开论文是否能形成清晰认知。",
                expected_value="提升研究叙事一致性，让后续分析更容易复用同一套材料。",
                evidence_ids=target_profile.evidence_ids[:6] if target_profile else [],
                next_steps=[
                    "梳理研究方向的核心论文与证据素材",
                    "补充最新进展和方法创新论文",
                    "将新增论文作为下一次 Run 的 seed papers",
                ],
                risks=["若研究方向本身尚未成熟，分析结论需要保持谨慎，避免过度推断。"],
                confidence=0.7,
            )
        )

    return recommendations[:5]


def grounded_opportunity_title(claim: Claim) -> str:
    dimension = DIMENSION_LABELS.get(claim.dimension, claim.dimension)
    if claim.claim_type == "cross_role_contrast":
        return f"把{dimension}中的来源角色分工转化为可检验研究问题"
    return f"将{dimension}中的多论文共识转化为可复现实验协议"


def grounded_opportunity_text(request: ResearchRequest, claim: Claim) -> str:
    dimension = DIMENSION_LABELS.get(claim.dimension, claim.dimension)
    axis = cluster_axis_phrase(claim.evidence_cluster_label) if claim.evidence_cluster_label else dimension
    if claim.claim_type == "cross_role_contrast":
        role_text = role_backing_phrase(claim)
        return (
            f"围绕 {request.target_topic} 的 {dimension} 方向，构造一个检验“{axis}”的研究问题："
            f"比较并连接 {role_text} 的证据分工，观察同一协议下的任务定义、方法机制和失败模式是否一致。"
        )
    return (
        f"围绕 {request.target_topic} 的 {dimension} 方向，把“{axis}”从综述性观察改写为可复现实验协议，"
        "明确任务输入、评价指标、对照设置和失败判据。"
    )


def grounded_backing_text(claim: Claim) -> str:
    role_text = role_backing_phrase(claim)
    support = f"{claim.source_paper_count} 篇独立论文"
    if role_text:
        support = f"{support}（{role_text}）"
    axis = cluster_axis_phrase(claim.evidence_cluster_label) if claim.evidence_cluster_label else claim.dimension_label
    return (
        f"这一综合判断由 {support}支撑，且限定在“{axis}”这一证据轴；"
        "因此适合作为 research question 的起点，而不是直接写成领域趋势。"
    )


def role_backing_phrase(claim: Claim) -> str:
    counts = claim.supporting_source_subtype_paper_counts or {}
    if not counts:
        return ""
    return "；".join(
        f"{source_role_label(role)} {count} 篇"
        for role, count in sorted(counts.items(), key=lambda item: (-item[1], source_role_label(item[0])))
    )


def build_observability(
    started_at: datetime,
    plan: ResearchPlan,
    metrics: RunMetrics,
    evidence: list[Evidence],
    claims: list[Claim],
    matrix: PaperPatternMatrix,
    recommendations: list[OpportunityRecommendation],
) -> ObservabilitySnapshot:
    now = datetime.now(timezone.utc)
    source_mix = Counter(ev.source_type for ev in evidence)
    claim_pass_rate = metrics.verified_claim_count / max(1, metrics.claim_count)
    red_team_rate = metrics.challenged_claim_count / max(1, metrics.claim_count)
    coverage_score = average(list(matrix.coverage_by_paper.values()) + list(matrix.coverage_by_dimension.values()))
    confidence = average([ev.confidence for ev in evidence] + [claim.confidence for claim in claims])
    report_confidence = round(min(0.95, confidence * (0.55 + coverage_score * 0.45)), 3)
    report_ready = claim_pass_rate >= 0.2 and coverage_score >= 0.25 and report_confidence >= 0.55
    report_warn = claim_pass_rate >= 0.1 and coverage_score >= 0.18
    gates = [
        QualityGate(
            gate_id="gate_evidence_volume",
            name="Evidence 数量",
            status=gate_status(metrics.evidence_count, 10, 3),
            score=min(1.0, metrics.evidence_count / 10),
            message=f"当前抽取 {metrics.evidence_count} 条 Evidence。",
            suggested_action="若少于 10 条，建议增加搜索来源或 seed URL。",
        ),
        QualityGate(
            gate_id="gate_source_success",
            name="内容可用率",
            status=gate_status(metrics.sources_fetched / max(1, metrics.source_candidates), 0.65, 0.35),
            score=round(metrics.sources_fetched / max(1, metrics.source_candidates), 3),
            message=f"{metrics.sources_fetched}/{max(1, metrics.source_candidates)} 个候选来源已有正文内容。",
            suggested_action="优先配置 Tavily/Exa/知乎内容搜索；少量无正文来源可用本地抓取或 seed URL 补充。",
        ),
        QualityGate(
            gate_id="gate_claim_review",
            name="Claim 审查通过率",
            status=gate_status(claim_pass_rate, 0.6, 0.25),
            score=round(claim_pass_rate, 3),
            message=f"{metrics.verified_claim_count}/{max(1, metrics.claim_count)} 条 Claim 通过 Red Team。",
            suggested_action="被挑战的 Claim 需要补证或降低措辞强度。",
        ),
        QualityGate(
            gate_id="gate_coverage",
            name="推理模式矩阵覆盖度",
            status=gate_status(coverage_score, 0.58, 0.28),
            score=round(coverage_score, 3),
            message=f"矩阵覆盖度为 {coverage_score:.0%}。",
            suggested_action="补齐低覆盖推理模式或低覆盖维度的来源。",
        ),
        QualityGate(
            gate_id="gate_report_readiness",
            name="报告结论可用性",
            status="pass" if report_ready else "warn" if report_warn else "fail",
            score=round(
                min(
                    1.0,
                    claim_pass_rate / 0.2 if claim_pass_rate else 0.0,
                    coverage_score / 0.25 if coverage_score else 0.0,
                    report_confidence / 0.55 if report_confidence else 0.0,
                ),
                3,
            ),
            message=(
                f"Claim 通过率 {claim_pass_rate:.0%}，矩阵覆盖度 {coverage_score:.0%}，"
                f"报告置信度 {report_confidence:.0%}。"
            ),
            suggested_action=(
                "未通过时，报告只能标注为探索性 pilot；主结论只使用 verified claims，"
                "其余观察进入补证队列。"
            ),
        ),
    ]
    return ObservabilitySnapshot(
        generated_at=now,
        total_duration_seconds=round((now - started_at).total_seconds(), 2),
        agent_count=len(plan.required_agents or []),
        skill_calls=sum(len(AGENT_SKILLS.get(agent, [])) for agent in (plan.required_agents or RESEARCH_AGENT_FLOW)),
        tool_calls=metrics.source_candidates + metrics.sources_fetched,
        source_mix=dict(source_mix),
        dimension_coverage=matrix.coverage_by_dimension,
        paper_coverage=matrix.coverage_by_paper,
        claim_pass_rate=round(claim_pass_rate, 3),
        red_team_challenge_rate=round(red_team_rate, 3),
        evidence_coverage_score=round(coverage_score, 3),
        report_confidence=report_confidence,
        quality_gates=gates,
        export_files={
            "markdown_report": "reports/report.md",
            "executive_summary": "reports/executive_summary.md",
            "methodology": "reports/methodology.md",
            "matrix_json": "exports/matrix.json",
            "matrix_csv": "exports/matrix.csv",
            "recommendations": "exports/recommendations.json",
            "evidence_clusters": "exports/evidence_clusters.json",
            "evidence_matrix_csv": "exports/evidence_matrix.csv",
        },
    )


def build_evidence_graph(evidence: list[Evidence], claims: list[Claim]) -> EvidenceGraph:
    nodes: dict[str, EvidenceGraphNode] = {}
    edges: list[EvidenceGraphEdge] = []
    for claim in claims:
        nodes[claim.claim_id] = EvidenceGraphNode(
            id=claim.claim_id,
            label=(claim.final_wording or claim.claim)[:80],
            node_type="claim",
            score=claim.confidence,
            meta={"dimension": claim.dimension, "status": claim.verification_status},
        )
        dimension_id = f"dim_{claim.dimension}"
        nodes.setdefault(
            dimension_id,
            EvidenceGraphNode(
                id=dimension_id,
                label=claim.dimension_label,
                node_type="dimension",
                score=0.7,
            ),
        )
        edges.append(EvidenceGraphEdge(source=dimension_id, target=claim.claim_id, edge_type="contains", weight=0.6))
    for ev in evidence:
        nodes[ev.evidence_id] = EvidenceGraphNode(
            id=ev.evidence_id,
            label=ev.fact[:80],
            node_type="evidence",
            score=ev.confidence,
            meta={"source_url": ev.source_url, "dimension": ev.dimension},
        )
        source_id = stable_id("source", ev.source_url)
        nodes.setdefault(
            source_id,
            EvidenceGraphNode(
                id=source_id,
                label=ev.source_title[:80] or ev.source_url,
                node_type="source",
                score=ev.authority_score,
                meta={"url": ev.source_url, "source_type": ev.source_type},
            ),
        )
        edges.append(EvidenceGraphEdge(source=ev.evidence_id, target=source_id, edge_type="from_source", weight=ev.confidence))
        if ev.paper:
            paper_id = stable_id("paper", ev.paper)
            nodes.setdefault(
                paper_id,
                EvidenceGraphNode(
                    id=paper_id,
                    label=ev.paper,
                    node_type="paper",
                    score=0.75,
                ),
            )
            edges.append(EvidenceGraphEdge(source=paper_id, target=ev.evidence_id, edge_type="has_evidence", weight=ev.confidence))
    evidence_ids = {ev.evidence_id for ev in evidence}
    for claim in claims:
        for evidence_id in claim.supporting_evidence_ids:
            if evidence_id in evidence_ids:
                edges.append(
                    EvidenceGraphEdge(
                        source=claim.claim_id,
                        target=evidence_id,
                        edge_type="supported_by",
                        weight=claim.confidence,
                    )
                )
    return EvidenceGraph(generated_at=datetime.now(timezone.utc), nodes=list(nodes.values()), edges=edges)


def extract_evidence_from_document(
    run_id: str, document: SourceDocument, dimensions: list[str], entities: list[str]
) -> list[Evidence]:
    sentences = split_sentences(document.content)
    evidence_items: list[Evidence] = []
    used_quotes: set[str] = set()
    for dimension in dimensions:
        keywords = DIMENSION_KEYWORDS.get(dimension, [])
        candidates = rank_sentences(sentences, keywords)
        for sentence, score in candidates[:2]:
            if sentence in used_quotes or len(sentence) < 45:
                continue
            used_quotes.add(sentence)
            paper = document.title or detect_entity(sentence, document.title, document.url, entities)
            quote = sentence[:500]
            fact = build_fact(quote, paper, dimension)
            confidence = min(0.95, 0.38 + score * 0.09 + source_weight(document.source_type))
            evidence_id = stable_id("ev", f"{run_id}:{document.url}:{dimension}:{quote}")
            evidence_items.append(
                Evidence(
                    evidence_id=evidence_id,
                    run_id=run_id,
                    url=document.url,
                    title=document.title,
                    content=quote,
                    fetched_at=document.fetched_at,
                    source_type=document.source_type,
                    dimension=dimension,
                    dimension_label=DIMENSION_LABELS.get(dimension, dimension),
                    paper=paper,
                    fact=fact,
                    quote=quote,
                    source_title=document.title,
                    source_url=document.url,
                    source_id=document.source_id,
                    source_subtype=document.source_subtype,
                    source_subtype_reason=document.source_subtype_reason,
                    confidence=round(confidence, 3),
                    authority_score=round(source_weight(document.source_type) + 0.3, 3),
                    freshness_score=0.75,
                    relevance_score=round(min(1.0, 0.35 + score * 0.1), 3),
                )
            )
            break
    if not evidence_items and sentences:
        sentence = max(sentences, key=len)[:500]
        paper = document.title or detect_entity(sentence, document.title, document.url, entities)
        evidence_items.append(
            Evidence(
                evidence_id=stable_id("ev", f"{run_id}:{document.url}:real_text:{sentence}"),
                run_id=run_id,
                url=document.url,
                title=document.title,
                content=sentence,
                fetched_at=document.fetched_at,
                source_type=document.source_type,
                dimension="other",
                dimension_label=DIMENSION_LABELS["other"],
                paper=paper,
                fact=build_fact(sentence, paper, "other"),
                quote=sentence,
                source_title=document.title,
                source_url=document.url,
                source_id=document.source_id,
                source_subtype=document.source_subtype,
                source_subtype_reason=document.source_subtype_reason,
                confidence=0.45,
                authority_score=round(source_weight(document.source_type) + 0.3, 3),
                freshness_score=0.7,
                relevance_score=0.45,
            )
        )
    return evidence_items


def split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    cleaned = [re.sub(r"\s+", " ", chunk).strip(" -•\t") for chunk in chunks]
    return [chunk for chunk in cleaned if 35 <= len(chunk) <= 700]


def rank_sentences(sentences: list[str], keywords: list[str]) -> list[tuple[str, int]]:
    ranked: list[tuple[str, int]] = []
    for sentence in sentences:
        score = sum(1 for keyword in keywords if keyword_matches(sentence, keyword))
        if score:
            ranked.append((sentence, score))
    ranked.sort(key=lambda item: (item[1], len(item[0])), reverse=True)
    return ranked


def keyword_matches(sentence: str, keyword: str) -> bool:
    lower_keyword = keyword.lower()
    if re.fullmatch(r"[a-z0-9][a-z0-9 -]*", lower_keyword):
        pattern = r"\b" + re.escape(lower_keyword).replace(r"\ ", r"\s+") + r"\b"
        return bool(re.search(pattern, sentence.lower()))
    return lower_keyword in sentence.lower()


def detect_entity(sentence: str, title: str, url: str, entities: list[str]) -> str | None:
    haystack = f"{sentence} {title} {url}".lower()
    for entity in entities:
        if entity and entity.lower() in haystack:
            return entity
    host = urlparse(url).netloc.lower()
    for entity in entities:
        token = re.sub(r"[^a-z0-9]+", "", entity.lower())
        if token and token in re.sub(r"[^a-z0-9]+", "", host):
            return entity
    return None


def infer_document_pairs(
    document: SourceDocument,
    entities: list[str],
    dimensions: list[str],
) -> set[tuple[str, str]]:
    text = f"{document.url} {document.title} {document.content[:1200]} {document.query}".lower()
    found_entities: set[str] = set()
    for entity in entities:
        entity_key = re.sub(r"[^a-z0-9]+", "", entity.lower())
        host_key = re.sub(r"[^a-z0-9]+", "", urlparse(document.url).netloc.lower())
        if entity.lower() in text or (entity_key and entity_key in host_key):
            found_entities.add(entity)
    found_dimensions: set[str] = set()
    for dimension in dimensions:
        if dimension in text or any(keyword.lower() in text for keyword in DIMENSION_KEYWORDS.get(dimension, [dimension])):
            found_dimensions.add(dimension)
    return {(entity, dimension) for entity in found_entities for dimension in found_dimensions}


def evidence_coverage_counts(
    evidence: list[Evidence],
    entities: list[str],
    dimensions: list[str],
) -> dict[tuple[str, str], int]:
    entity_set = set(entities)
    dimension_set = set(dimensions)
    counts: dict[tuple[str, str], int] = {}
    for item in evidence:
        if item.paper in entity_set and item.dimension in dimension_set:
            key = (item.paper, item.dimension)
            counts[key] = counts.get(key, 0) + 1
    return counts


def evidence_cell_sufficient(count: int) -> bool:
    # 严一点：同一论文/方向×推理模式至少 5 条 Evidence 后才跳过后续同格文档。
    return count >= 5


def build_fact(quote: str, paper: str | None, dimension: str) -> str:
    subject = paper or "该论文"
    label = DIMENSION_LABELS.get(dimension, dimension)
    trimmed = quote.strip()
    if len(trimmed) > 160:
        trimmed = trimmed[:157] + "..."
    return f"{subject} 在{label}相关论文文本中出现了可核验表述：{trimmed}"


def source_weight(source_type: str) -> float:
    weights = {
        "academic_paper": 0.42,
        "official_website": 0.36,
        "docs": 0.35,
        "changelog": 0.32,
        "github": 0.28,
        "review_platform": 0.22,
        "user_review": 0.20,   # legacy run compatibility
        "blog": 0.22,
        "other": 0.18,
    }
    return weights.get(source_type, 0.2)


def stable_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def build_evidence_csv(evidence: list[Evidence], by_evidence: dict[str, Evidence]) -> str:
    _ = by_evidence
    rows = ["evidence_id,dimension,paper,confidence,source_url,fact"]
    for ev in evidence:
        rows.append(
            ",".join(
                csv_escape(value)
                for value in [
                    ev.evidence_id,
                    ev.dimension,
                    ev.paper or "",
                    f"{ev.confidence:.3f}",
                    ev.source_url,
                    ev.fact,
                ]
            )
        )
    return "\n".join(rows) + "\n"


def csv_escape(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def unique_strings(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = (value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def select_matrix_papers(request: ResearchRequest, evidence: list[Evidence], limit: int = 12) -> list[str]:
    paper_counts = Counter(ev.paper for ev in evidence if ev.paper)
    papers = unique_strings(
        [
            *request.seed_papers,
            *[paper for paper, _ in paper_counts.most_common()],
        ]
    )[:limit]
    return papers or [request.target_topic]


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0


def cell_status(count: int, confidence: float) -> str:
    if count >= 3 and confidence >= 0.68:
        return "strong"
    if count >= 2 and confidence >= 0.55:
        return "partial"
    if count >= 1:
        return "weak"
    return "unknown"


def coverage_points(cell: MatrixCell) -> float:
    if cell.status == "strong":
        return 1.0
    if cell.status == "partial":
        return 0.68
    if cell.status == "weak":
        return 0.35
    return 0.0


def cell_summary(paper: str, dimension: str, items: list[Evidence]) -> str:
    label = DIMENSION_LABELS.get(dimension, dimension)
    if not items:
        return f"尚未在本次公开资料中找到 {paper} 的{label}强证据。"
    top = sorted(items, key=lambda ev: ev.confidence, reverse=True)[:2]
    facts = "；".join(compact_fact(ev.fact) for ev in top)
    return f"{paper} 在{label}维度有 {len(items)} 条证据：{facts}"


def profile_summary(
    paper: str, evidence: list[Evidence], strong_dimensions: list[str], weak_dimensions: list[str]
) -> str:
    if not evidence:
        return f"本次运行尚未形成 {paper} 的有效公开证据画像。"
    source_count = len({ev.source_url for ev in evidence})
    strong = "、".join(strong_dimensions[:3]) or "暂无明显强覆盖维度"
    weak = "、".join(weak_dimensions[:3]) or "暂无明显缺口"
    return (
        f"{paper} 当前画像来自 {len(evidence)} 条 Evidence 和 {source_count} 个来源；"
        f"证据覆盖较强的方向是 {strong}，仍需补充验证的方向是 {weak}。"
    )


def build_core_findings(request: ResearchRequest, matrix: PaperPatternMatrix, claims: list[Claim]) -> list[str]:
    findings: list[str] = []
    target = request.target_topic
    by_paper = {profile.paper: profile for profile in matrix.profiles}
    target_profile = by_paper.get(target)
    if target_profile:
        strong = "、".join(target_profile.strongest_dimensions[:3]) or "暂无强覆盖维度"
        weak = "、".join(target_profile.weak_or_unknown_dimensions[:3]) or "暂无明显缺口"
        findings.append(
            f"{target} 的证据画像目前最集中在 {strong}；需要谨慎解读或继续补证的方向是 {weak}。"
        )

    strongest_dimension = sorted(
        matrix.coverage_by_dimension.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    if strongest_dimension:
        label = matrix.dimension_labels.get(strongest_dimension[0][0], strongest_dimension[0][0])
        findings.append(f"本次资料覆盖最充分的维度是 {label}，覆盖度约 {strongest_dimension[0][1]:.0%}。")

    verified = [
        claim for claim in claims
        if is_report_ready_claim(claim)
    ]
    lead_claims = sorted(verified, key=lambda item: item.confidence, reverse=True)[:3]
    if lead_claims:
        for claim in lead_claims:
            findings.append(f"{DIMENSION_LABELS.get(claim.dimension, claim.dimension)}：{compact_fact(best_claim_text(claim), 300)}")
    elif claims:
        findings.append("本轮 Red Team 尚未确认足够多的稳健 Claim；以下内容只能作为待验证观察和补证线索。")

    return findings[:5] or ["本次运行已形成基础证据库，但仍需要更多可交叉验证的高质量来源来支撑强结论。"]


def report_ready_finding_lines(claims: list[Claim], limit: int = 3, max_len: int = 300) -> list[str]:
    ready_claims = sorted(
        [claim for claim in claims if is_report_ready_claim(claim)],
        key=lambda item: (item.confidence, item.source_paper_count, len(item.supporting_evidence_ids)),
        reverse=True,
    )[:limit]
    return [
        f"- **{DIMENSION_LABELS.get(claim.dimension, claim.dimension)}**："
        f"{compact_fact(best_claim_text(claim), max_len)}"
        for claim in ready_claims
    ]


def build_analysis_report(
    request: ResearchRequest,
    evidence: list[Evidence],
    claims: list[Claim],
    metrics: RunMetrics,
    matrix: PaperPatternMatrix,
    recommendations: list[OpportunityRecommendation],
    observability: ObservabilitySnapshot,
) -> str:
    citation_numbers, reference_lines = build_citation_index(evidence)
    core_findings = build_core_findings(request, matrix, claims)
    target_strengths, target_weaknesses = build_target_advantage_analysis(request, matrix, claims, citation_numbers)
    user_attention = build_user_attention_analysis(request, matrix, claims, citation_numbers)
    research_guidance = build_research_guidance(request, matrix, target_strengths, target_weaknesses)
    by_dimension: dict[str, list[Claim]] = defaultdict(list)
    for claim in claims:
        by_dimension[claim.dimension].append(claim)

    lines = [
        f"# {request.project_name}",
        "",
        "## 摘要",
        "",
        *[f"- {finding}" for finding in core_findings],
        "",
        "## 推理模式对比矩阵",
        "",
        build_matrix_markdown(
            matrix,
            citation_numbers,
            focus_evidence_ids=report_ready_support_evidence_ids(claims),
        ),
        "",
        "### 矩阵解读",
        "",
        *build_matrix_insights(matrix, claims, citation_numbers),
        "",
        "## 证据背书机会表",
        "",
        *build_evidence_backed_opportunity_table(request, claims, citation_numbers),
        "",
        "## 形式化证据门控与数学背书",
        "",
        *build_formal_evidence_gate(claims, citation_numbers),
        "",
        "## 可投稿研究命题与实验化路径",
        "",
        *build_submission_framing(request, claims, citation_numbers),
        "",
        f"## {request.target_topic} 的研究进展与空白",
        "",
        "### 已具备的优势",
        "",
        *target_strengths,
        "",
        "### 需要补齐或谨慎表达的不足",
        "",
        *target_weaknesses,
        "",
        "## 推理模式关注点",
        "",
        *user_attention,
        "",
        "## 后续研究建议",
        "",
        *research_guidance,
        "",
        "## 分维度深度分析",
        "",
    ]

    report_dimensions = request.analysis_dimensions or matrix.dimensions or DEFAULT_DIMENSIONS
    for dimension in report_dimensions:
        dimension_claims = sorted(by_dimension.get(dimension, []), key=lambda item: item.confidence, reverse=True)
        dimension_cells = [cell for cell in matrix.cells if cell.dimension == dimension]
        dimension_evidence = sorted(
            [item for item in evidence if item.dimension == dimension],
            key=lambda item: item.confidence,
            reverse=True,
        )
        lines.extend([f"### {DIMENSION_LABELS.get(dimension, dimension)}", ""])
        report_ready_claims = [claim for claim in dimension_claims if is_report_ready_claim(claim)]
        audit_only_claims = [claim for claim in dimension_claims if not is_report_ready_claim(claim)]
        lines.extend(build_dimension_overview(dimension, dimension_cells, report_ready_claims))
        if report_ready_claims:
            lines.extend(["", "#### 可进入报告主体的综合结论", ""])
        for claim in report_ready_claims[:4]:
            cite = citations_for_ids(claim.supporting_evidence_ids, citation_numbers)
            wording = compact_fact(best_claim_text(claim), 280)
            lines.append(f"- {wording}{cite}")
        representative_evidence = report_ready_representative_evidence(
            dimension,
            report_ready_claims,
            evidence,
        )
        if representative_evidence:
            lines.append(
                f"- 代表性支撑证据："
                f"{representative_evidence_axis_summary(report_ready_claims, representative_evidence, citation_numbers)}"
            )
        if audit_only_claims:
            lines.extend(["", "#### 待验证观察与补证线索", ""])
        for claim in audit_only_claims[:4]:
            cite = citations_for_ids(claim.supporting_evidence_ids, citation_numbers)
            reason = claim_report_ready_reason(claim)
            wording = audit_only_claim_display_text(claim)
            lines.append(f"- {wording}{cite}（暂不进入主结论：{reason or 'needs_review'}）")
        if not dimension_claims and has_citable_dimension_cells(dimension_cells):
            lines.extend(build_dimension_fallback_points(dimension_cells, citation_numbers))
        elif audit_only_claims and not report_ready_claims and dimension_evidence:
            lines.append("- 本维度当前证据仅支撑待验证观察，不单列为主体代表性证据；详见 Evidence 附录。")
        lines.append("")

    lines.extend(["## 可检验研究假设与学术背书", ""])
    lines.extend(build_grounded_hypotheses(request, claims, citation_numbers))
    lines.append("")

    lines.extend(["## 研究机会与下一步", ""])
    if recommendations:
        for item in recommendations[:5]:
            cite = citations_for_ids(item.evidence_ids, citation_numbers)
            lines.extend(
                [
                    f"### {item.title}",
                    "",
                    f"{item.recommendation}{cite}",
                    "",
                    f"- 依据：{item.rationale}",
                    f"- 预期价值：{item.expected_value}",
                    f"- 下一步：{'；'.join(item.next_steps) or '继续补充验证'}",
                    "",
                ]
            )
    else:
        lines.extend(["当前证据不足以形成强研究建议，应优先补齐相关论文和最新进展。", ""])

    lines.extend(
        [
            "## 风险与局限性",
            "",
            *build_risk_notes(claims, matrix, observability),
            "",
            "## 参考来源",
            "",
            *reference_lines,
            "",
            "## Evidence 附录",
            "",
        ]
    )
    for ev in evidence[:80]:
        cite = citations_for_ids([ev.evidence_id], citation_numbers)
        lines.extend(
            [
                f"### {ev.evidence_id} {cite} · {ev.dimension_label}",
                "",
                f"- 论文：{ev.paper or '未识别'}",
                f"- 事实：{ev.fact}",
                f"- 原文片段：{ev.quote}",
                "",
            ]
        )
    return "\n".join(lines)


def build_evidence_gaps(matrix: PaperPatternMatrix) -> list[str]:
    weak_cells = sorted(
        [cell for cell in matrix.cells if cell.status in {"weak", "unknown"}],
        key=lambda cell: (cell.status != "unknown", cell.evidence_count),
    )
    if not weak_cells:
        return ["按当前论文×推理模式矩阵看，未发现明显空白格；后续重点应转向证据交叉验证和时效性复核。"]
    return [
        f"{cell.paper} × {cell.dimension_label}：{cell.status}，当前 {cell.evidence_count} 条证据。"
        for cell in weak_cells[:8]
    ]


def build_citation_index(evidence: list[Evidence]) -> tuple[dict[str, int], list[str]]:
    citation_numbers: dict[str, int] = {}
    references: list[str] = []
    seen_urls: set[str] = set()
    for ev in sorted(evidence, key=lambda item: item.confidence, reverse=True):
        if ev.evidence_id in citation_numbers:
            continue
        normalized = normalize_reference_url(ev.source_url)
        if normalized in seen_urls:
            number = next(
                (
                    citation_numbers[item.evidence_id]
                    for item in evidence
                    if normalize_reference_url(item.source_url) == normalized and item.evidence_id in citation_numbers
                ),
                len(references) + 1,
            )
            citation_numbers[ev.evidence_id] = number
            continue
        seen_urls.add(normalized)
        number = len(references) + 1
        citation_numbers[ev.evidence_id] = number
        references.append(f"[{number}] {ev.source_title or ev.title}. {ev.source_url}")
        if len(references) >= 120:
            break
    return citation_numbers, references


def normalize_reference_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.netloc.lower()}{parsed.path}".rstrip("/")


def citations_for_ids(evidence_ids: list[str], citation_numbers: dict[str, int], limit: int = 4) -> str:
    numbers = []
    for evidence_id in evidence_ids:
        number = citation_numbers.get(evidence_id)
        if number and number not in numbers:
            numbers.append(number)
        if len(numbers) >= limit:
            break
    return "".join(f"[{number}]" for number in numbers)


def report_ready_support_evidence_ids(claims: list[Claim]) -> set[str]:
    return {
        evidence_id
        for claim in claims
        if is_report_ready_claim(claim)
        for evidence_id in claim.supporting_evidence_ids
    }


def report_ready_representative_evidence(
    dimension: str,
    report_ready_claims: list[Claim],
    evidence: list[Evidence],
) -> list[Evidence]:
    support_ids = {
        evidence_id
        for claim in report_ready_claims
        if claim.dimension == dimension
        for evidence_id in claim.supporting_evidence_ids
    }
    if not support_ids:
        return []
    return sorted(
        [
            item for item in evidence
            if item.dimension == dimension and item.evidence_id in support_ids
        ],
        key=lambda item: (report_evidence_snippet_score(item), item.confidence),
        reverse=True,
    )


def representative_evidence_axis_summary(
    report_ready_claims: list[Claim],
    representative_evidence: list[Evidence],
    citation_numbers: dict[str, int],
) -> str:
    lead_claim = sorted(
        report_ready_claims,
        key=lambda item: (item.source_paper_count, len(item.supporting_evidence_ids), item.confidence),
        reverse=True,
    )[0]
    axis = cluster_axis_phrase(lead_claim.evidence_cluster_label) if lead_claim.evidence_cluster_label else lead_claim.dimension_label
    role_text = role_backing_phrase(lead_claim) or f"{lead_claim.source_paper_count} 篇独立论文"
    paper_bits = []
    seen: set[str] = set()
    for item in representative_evidence:
        paper = compact_paper_title(item.paper or item.source_title or "相关论文", 58)
        if paper in seen:
            continue
        seen.add(paper)
        paper_bits.append(f"{paper}{citations_for_ids([item.evidence_id], citation_numbers)}")
        if len(paper_bits) >= 4:
            break
    papers = "、".join(paper_bits) or "相关论文"
    return (
        f"{papers} 共同支撑“{axis}”证据轴；当前由 {role_text} 支撑，"
        "主体结论只引用该范围限定证据，原文片段保留在 Evidence 附录。"
    )


REPORT_SNIPPET_BAD_PREFIXES = {
    "based",
    "ilarity",
    "lacker",
    "nations",
    "nism",
    "ods",
    "over",
    "plex",
    "ques",
    "sive",
    "tions",
    "ural",
    "velop",
}
REPORT_SNIPPET_ALLOWED_LOWER_STARTS = {
    "a",
    "an",
    "as",
    "can",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "our",
    "the",
    "their",
    "there",
    "these",
    "this",
    "through",
    "to",
    "using",
    "was",
    "we",
    "when",
    "while",
    "with",
}
REPORT_SNIPPET_BAD_ENDINGS = {"en", "knowl", "ques", "rea", "sig", "struc"}


def report_evidence_snippet(item: Evidence, max_len: int = 92) -> str:
    for text in [item.quote, item.fact]:
        cleaned = clean_report_evidence_text(item, text)
        if cleaned and not is_low_quality_report_snippet(cleaned):
            return compact_fact(cleaned, max_len)
    return report_evidence_fallback_phrase(item, max_len)


def report_evidence_snippet_score(item: Evidence) -> int:
    for text in [item.quote, item.fact]:
        cleaned = clean_report_evidence_text(item, text)
        if cleaned and not is_low_quality_report_snippet(cleaned):
            return 1
    return 0


def clean_report_evidence_text(item: Evidence, text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(
        r"^.+?在[^:：]{0,120}相关论文文本中出现了可核验表述[:：]\s*",
        "",
        cleaned,
    ).strip()
    paper = (item.paper or item.source_title or "").strip()
    if paper:
        paper_lower = paper.lower()
        cleaned_lower = cleaned.lower()
        if cleaned.startswith(paper):
            cleaned = cleaned[len(paper):].lstrip(" :：-–—,.;，。")
        elif len(cleaned) >= 24 and (paper_lower.startswith(cleaned_lower) or cleaned_lower.startswith(paper_lower)):
            cleaned = ""
    return cleaned


def is_low_quality_report_snippet(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) < 24:
        return True
    if re.match(r"(?i)^(figure|table|algorithm)\s*\d+\b", cleaned):
        return True
    words = re.findall(r"[A-Za-z]+", cleaned)
    if words:
        first = words[0]
        lower = first.lower()
        if lower in REPORT_SNIPPET_BAD_PREFIXES:
            return True
        if first[:1].islower() and len(lower) <= 6 and lower not in REPORT_SNIPPET_ALLOWED_LOWER_STARTS:
            return True
        last = words[-1].lower()
        if last in REPORT_SNIPPET_BAD_ENDINGS:
            return True
    return False


def report_evidence_fallback_phrase(item: Evidence, max_len: int = 92) -> str:
    if item.source_subtype in SYNTHESIS_SOURCE_ROLES:
        axis = source_role_contribution_phrase(item.source_subtype, item.dimension_label)
    else:
        axis = item.mechanism or item.bottleneck or item.reasoning_pattern or item.dimension_label
    return compact_fact(f"围绕{axis}提供可审计支撑，详见 Evidence 附录", max_len)


def audit_only_claim_display_text(claim: Claim) -> str:
    axis = cluster_axis_phrase(claim.evidence_cluster_label) if claim.evidence_cluster_label else claim.dimension_label
    role_text = role_backing_phrase(claim)
    support_text = role_text or (
        f"{claim.source_paper_count} 篇独立论文"
        if claim.source_paper_count > 1
        else "单篇论文"
    )
    if claim.claim_type == "single_paper_observation" or claim.source_paper_count <= 1:
        return (
            f"单论文待验证观察：{claim.dimension_label} 上有关于“{axis}”的可审计线索，"
            "需补充独立论文后再写入主体。"
        )
    return (
        f"待验证综合观察：{claim.dimension_label} 的“{axis}”已有来自 {support_text} 的线索，"
        "但当前不满足 report-ready 条件。"
    )


def has_citable_dimension_cells(cells: list[MatrixCell]) -> bool:
    return any(cell.evidence_count > 0 and cell.status in {"strong", "partial"} for cell in cells)


def build_evidence_backed_opportunity_table(
    request: ResearchRequest,
    claims: list[Claim],
    citation_numbers: dict[str, int],
    limit: int = 4,
) -> list[str]:
    ready_claims = sorted(
        [claim for claim in claims if is_report_ready_claim(claim)],
        key=lambda item: (
            item.claim_type == "cross_role_contrast",
            item.source_paper_count,
            item.confidence,
        ),
        reverse=True,
    )[:limit]
    if not ready_claims:
        return ["当前尚无足够 report-ready synthesis 形成机会表；应先补充核心论文和反例证据。"]

    rows = [
        "| 机会 | 证据轴 | 学术背书 | 下一步验证 |",
        "|---|---|---|---|",
    ]
    for claim in ready_claims:
        citations = citations_for_ids(claim.supporting_evidence_ids, citation_numbers, limit=3)
        axis = cluster_axis_phrase(claim.evidence_cluster_label) if claim.evidence_cluster_label else claim.dimension_label
        role_text = role_backing_phrase(claim) or "source role 尚未分组"
        if claim.claim_type == "cross_role_contrast":
            next_step = "用同一任务协议检验不同 source role 的机制分工是否互补。"
        else:
            next_step = "把该证据轴固化为任务、指标、对照和失败判据。"
        rows.append(
            "|"
            + "|".join(
                csv_safe_markdown(value)
                for value in [
                    grounded_opportunity_title(claim),
                    compact_fact(axis, 90),
                    f"{claim.source_paper_count} 篇独立论文；{role_text}{citations}",
                    next_step,
                ]
            )
            + "|"
        )
    return rows


def build_formal_evidence_gate(
    claims: list[Claim],
    citation_numbers: dict[str, int],
    limit: int = 4,
) -> list[str]:
    ready_claims = sorted(
        [claim for claim in claims if is_report_ready_claim(claim)],
        key=lambda item: (
            item.claim_type == "cross_role_contrast",
            item.source_paper_count,
            len(item.supporting_evidence_ids),
            item.confidence,
        ),
        reverse=True,
    )[:limit]
    if not ready_claims:
        return [
            "当前尚无 report-ready synthesis，因此不生成形式化主体结论证书；应先补充独立论文、反例和可审计 evidence。"
        ]

    lines = [
        "记一条候选综合结论为 c，E_c 为 supporting evidence 集合，P_c 为独立论文集合，"
        "R_c 为 source role 集合。报告主体只接收满足以下充分条件的 c：",
        "",
        "`g(c)=1 iff verified(c) and risk(c)<high and |E_c|>=2 and |P_c|>=2 and support(c)=strong and R_c subset R_reportable`。",
        "",
        "若 c 是 cross-role contrast，还要求 `|P_c|>=4` 且每个 reportable role 至少由 2 篇独立论文支撑；"
        "否则只能写成待验证观察或补证 backlog。",
        "",
        "| 综合结论范围 | 类型 | 证据证书 | 门控结论 | 引用 |",
        "|---|---|---|---|---|",
    ]
    for claim in ready_claims:
        axis = (
            cluster_axis_phrase(claim.evidence_cluster_label)
            if claim.evidence_cluster_label
            else DIMENSION_LABELS.get(claim.dimension, claim.dimension_label)
        )
        certificate = formal_gate_certificate(claim)
        citations = citations_for_ids(claim.supporting_evidence_ids, citation_numbers, limit=4)
        lines.append(
            "|"
            + "|".join(
                csv_safe_markdown(value)
                for value in [
                    compact_fact(axis, 88),
                    claim.claim_type,
                    certificate,
                    "g(c)=1，可进入主体；边界条件仍需在实验中复核",
                    citations,
                ]
            )
            + "|"
        )
    return lines


def formal_gate_certificate(claim: Claim) -> str:
    evidence_count = len(claim.supporting_evidence_ids)
    role_paper_counts = getattr(claim, "supporting_source_subtype_paper_counts", {}) or {}
    if claim.claim_type == "cross_role_contrast" and role_paper_counts:
        min_role_papers = min(role_paper_counts.values())
        role_bits = ", ".join(
            f"{source_role_label(role)}:{count}"
            for role, count in sorted(role_paper_counts.items())
        )
        return (
            f"verified; |E_c|={evidence_count}; |P_c|={claim.source_paper_count}; "
            f"min_r |P_c,r|={min_role_papers}; roles=({role_bits}); support={claim.evidence_support_level}"
        )
    role_text = role_backing_phrase(claim) or "source role 已记录"
    return (
        f"verified; |E_c|={evidence_count}; |P_c|={claim.source_paper_count}; "
        f"roles={role_text}; support={claim.evidence_support_level}"
    )


def build_submission_framing(
    request: ResearchRequest,
    claims: list[Claim],
    citation_numbers: dict[str, int],
    limit: int = 3,
) -> list[str]:
    ready_claims = sorted(
        [claim for claim in claims if is_report_ready_claim(claim)],
        key=lambda item: (
            item.claim_type == "cross_role_contrast",
            item.source_paper_count,
            len(item.supporting_evidence_ids),
            item.confidence,
        ),
        reverse=True,
    )[:limit]
    if not ready_claims:
        return [
            "当前尚无可进入主体的 report-ready synthesis；不应包装成投稿命题，应先补充核心论文、反例和可复现实验协议。"
        ]

    lines: list[str] = []
    for index, claim in enumerate(ready_claims, start=1):
        citations = citations_for_ids(claim.supporting_evidence_ids, citation_numbers, limit=4)
        lines.extend(
            [
                f"### P{index}. {submission_framing_title(claim)}",
                "",
                f"- **问题定义**：{submission_problem_statement(request, claim)}{citations}",
                f"- **方法假设**：{submission_method_hypothesis(request, claim)}",
                f"- **实验化验证**：{submission_evaluation_plan(claim)}",
                "- **边界条件**：该命题只继承 report-ready evidence 的范围；正式写作前必须补充反例论文、消融实验和人工原文复核。",
                "",
            ]
        )
    return lines


def submission_framing_title(claim: Claim) -> str:
    dimension = DIMENSION_LABELS.get(claim.dimension, claim.dimension)
    if claim.claim_type == "cross_role_contrast":
        return f"把{dimension}中的来源分工转化为可投稿问题"
    return f"把{dimension}中的证据轴转化为可投稿问题"


def submission_problem_statement(request: ResearchRequest, claim: Claim) -> str:
    dimension = DIMENSION_LABELS.get(claim.dimension, claim.dimension)
    axis = cluster_axis_phrase(claim.evidence_cluster_label) if claim.evidence_cluster_label else dimension
    role_text = role_backing_phrase(claim) or f"{claim.source_paper_count} 篇独立论文"
    scope = f"{request.target_topic} 的{dimension}"
    if claim.claim_type == "cross_role_contrast":
        return (
            f"在 {scope} 中，当前证据不是支持一个宽泛领域趋势，"
            f"而是暴露出围绕“{axis}”的 source-role 分工；已有 {role_text} 支撑。"
        )
    return (
        f"在 {scope} 中，当前证据不是支持泛化结论，"
        f"而是指向一个可被任务化的“{axis}”瓶颈；已有 {role_text} 支撑。"
    )


def submission_method_hypothesis(request: ResearchRequest, claim: Claim) -> str:
    dimension = DIMENSION_LABELS.get(claim.dimension, claim.dimension)
    axis = cluster_axis_phrase(claim.evidence_cluster_label) if claim.evidence_cluster_label else dimension
    if claim.claim_type == "cross_role_contrast":
        return (
            f"把不同 source role 的贡献拆成协议变量，构造一个覆盖{dimension}的统一 benchmark，"
            f"检验“{axis}”中的任务定义、方法机制和失败模式是否能在同一设置下互补。"
        )
    return (
        f"把“{axis}”显式建模为 {request.target_topic} 的可控实验变量，"
        f"检验它是否解释{dimension}中跨论文反复出现的能力边界。"
    )


def submission_evaluation_plan(claim: Claim) -> str:
    axis = (
        cluster_axis_phrase(claim.evidence_cluster_label)
        if claim.evidence_cluster_label
        else DIMENSION_LABELS.get(claim.dimension, claim.dimension_label)
    )
    if claim.claim_type == "cross_role_contrast":
        return (
            f"以“{axis}”为主轴，固定任务输入、指标和失败判据，分别加入/移除各 source role 对应的机制，"
            "报告跨角色互补性、冲突点和负例。"
        )
    return (
        f"以“{axis}”为主轴，设计同一任务下的 baseline、机制增强、消融和反例集，"
        "只在独立论文证据与实验结果一致时写成贡献。"
    )


def build_grounded_hypotheses(
    request: ResearchRequest,
    claims: list[Claim],
    citation_numbers: dict[str, int],
    limit: int = 3,
) -> list[str]:
    ready_claims = sorted(
        [claim for claim in claims if is_report_ready_claim(claim)],
        key=lambda item: (
            item.claim_type == "cross_role_contrast",
            item.source_paper_count,
            item.confidence,
        ),
        reverse=True,
    )[:limit]
    if not ready_claims:
        return [
            "当前尚未形成足够强的 report-ready synthesis；不应强行提出研究假设，应先补充核心论文和反例证据。"
        ]

    lines: list[str] = []
    for index, claim in enumerate(ready_claims, start=1):
        dimension = DIMENSION_LABELS.get(claim.dimension, claim.dimension)
        citations = citations_for_ids(claim.supporting_evidence_ids, citation_numbers)
        axis = cluster_axis_phrase(claim.evidence_cluster_label) if claim.evidence_cluster_label else dimension
        if claim.claim_type == "cross_role_contrast":
            hypothesis = (
                f"在 {request.target_topic} 的 {dimension} 中，检验不同 source role 在“{axis}”上的分工是否可以"
                "形成同一套可复现评测或方法协议。"
            )
        else:
            hypothesis = (
                f"在 {request.target_topic} 的 {dimension} 中，将“{axis}”固化为可复现实验协议，"
                "检验现有多论文共识是否能跨任务、数据集或方法设置保持成立。"
            )
        lines.extend(
            [
                f"### H{index}. {grounded_opportunity_title(claim)}",
                "",
                f"- **研究假设**：{hypothesis}{citations}",
                f"- **证据背书**：{grounded_backing_text(claim)}",
                "- **边界条件**：该假设只由当前 report-ready evidence 支撑；正式写作前需要补充反例、消融设置和人工原文复核。",
                "",
            ]
        )
    return lines


def build_target_advantage_analysis(
    request: ResearchRequest,
    matrix: PaperPatternMatrix,
    claims: list[Claim],
    citation_numbers: dict[str, int],
) -> tuple[list[str], list[str]]:
    target_cells = [cell for cell in matrix.cells if cell.paper == request.target_topic]
    strengths: list[str] = []
    weaknesses: list[str] = []
    for cell in sorted(target_cells, key=lambda item: (coverage_points(item), item.confidence, item.evidence_count), reverse=True):
        cite = citations_for_ids(cell.evidence_ids, citation_numbers)
        if cell.status in {"strong", "partial"}:
            strengths.append(f"- **{cell.dimension_label}**：{cell.summary}{cite}")
        else:
            weaknesses.append(
                f"- **{cell.dimension_label}**：当前公开证据偏弱，不能把它作为强结论；建议补充可被引用的论文、实验结果或最新进展。{cite}"
            )
    if not strengths:
        lead_claims = sorted(
            [
                claim for claim in claims
                if is_report_ready_claim(claim)
            ],
            key=lambda item: item.confidence,
            reverse=True,
        )[:3]
        strengths = [
            f"- {compact_fact(best_claim_text(claim), 180)}{citations_for_ids(claim.supporting_evidence_ids, citation_numbers)}"
            for claim in lead_claims
        ] or [f"- 暂未形成 {request.target_topic} 的强研究结论，应优先补齐相关论文和最新进展。"]
    if not weaknesses:
        weaknesses = ["- 暂无明显空白维度；下一步应重点验证优势是否能被第三方论文和实验结果交叉支持。"]
    return strengths[:6], weaknesses[:6]


def build_user_attention_analysis(
    request: ResearchRequest,
    matrix: PaperPatternMatrix,
    claims: list[Claim],
    citation_numbers: dict[str, int],
) -> list[str]:
    _ = request
    _ = matrix
    ready_claims = sorted(
        [claim for claim in claims if is_report_ready_claim(claim)],
        key=lambda item: (
            item.claim_type == "cross_role_contrast",
            item.source_paper_count,
            item.confidence,
        ),
        reverse=True,
    )
    lines: list[str] = []
    seen_axes: set[tuple[str, str]] = set()
    for claim in ready_claims:
        axis = cluster_axis_phrase(claim.evidence_cluster_label) if claim.evidence_cluster_label else claim.dimension_label
        key = (claim.dimension, axis)
        if key in seen_axes:
            continue
        seen_axes.add(key)
        label = DIMENSION_LABELS.get(claim.dimension, claim.dimension)
        cite = citations_for_ids(claim.supporting_evidence_ids, citation_numbers, limit=3)
        role_text = role_backing_phrase(claim) or f"{claim.source_paper_count} 篇独立论文"
        lines.append(
            f"- **{label}**：关注 report-ready evidence axis “{axis}”，"
            f"当前由 {role_text} 支撑；后续写作应围绕该轴设计研究问题、实验协议和失败判据。{cite}"
        )
        if len(lines) >= 6:
            break
    return lines or ["- 当前尚无 report-ready 推理模式关注点；应先补充核心论文、反例和可复现实验，再写入报告主体。"]


def build_research_guidance(
    request: ResearchRequest,
    matrix: PaperPatternMatrix,
    strengths: list[str],
    weaknesses: list[str],
) -> list[str]:
    target = request.target_topic
    target_cells = {cell.dimension: cell for cell in matrix.cells if cell.paper == target}
    guidance = [
        f"- 把 {target} 的后续分析重心放在已有证据支撑的推理模式上，优先解释具体瓶颈、机制和实验支撑。",
        "- 对已有强证据的模式，不做绝对优劣判断；改为说明它适合解决什么研究约束，以及哪些假设仍需验证。",
    ]
    weak_modes = [
        cell.dimension_label
        for cell in target_cells.values()
        if cell.status in {"weak", "unknown"}
    ][:3]
    if weak_modes:
        guidance.append(
            f"- 证据不足的推理模式（{'、'.join(weak_modes)}）暂不适合写成强结论；下一轮应补代表性论文、实验结果和方法对比。"
        )
    return guidance[:7]


def build_risk_notes(
    claims: list[Claim],
    matrix: PaperPatternMatrix,
    observability: ObservabilitySnapshot,
) -> list[str]:
    challenged = [claim for claim in claims if claim.verification_status != "verified"]
    gaps = build_evidence_gaps(matrix)
    if observability.report_confidence >= 0.55 and observability.claim_pass_rate >= 0.2:
        confidence_note = "适合用于方向判断；强学术结论仍应结合领域知识和原文复核"
    else:
        confidence_note = "仅适合作为探索性 pilot 和补证清单，不适合作为正式研究结论"
    lines = [
        f"- 当前报告可信度约为 {observability.report_confidence:.0%}，{confidence_note}。",
    ]
    if challenged:
        lines.append(f"- 仍有 {len(challenged)} 条 Claim 未通过基础审查，报告正文已尽量采用保守措辞。")
    for gap in gaps[:4]:
        lines.append(f"- {gap}")
    return lines


def build_matrix_insights(
    matrix: PaperPatternMatrix,
    claims: list[Claim] | None = None,
    citation_numbers: dict[str, int] | None = None,
) -> list[str]:
    citation_numbers = citation_numbers or {}
    ready_claims = sorted(
        [claim for claim in (claims or []) if is_report_ready_claim(claim)],
        key=lambda item: (
            item.claim_type == "cross_role_contrast",
            item.source_paper_count,
            item.confidence,
        ),
        reverse=True,
    )
    if ready_claims:
        lines: list[str] = []
        for claim in ready_claims[:6]:
            dimension = matrix.dimension_labels.get(claim.dimension, DIMENSION_LABELS.get(claim.dimension, claim.dimension))
            axis = cluster_axis_phrase(claim.evidence_cluster_label) if claim.evidence_cluster_label else dimension
            role_text = role_backing_phrase(claim) or f"{claim.source_paper_count} 篇独立论文"
            citations = citations_for_ids(claim.supporting_evidence_ids, citation_numbers, limit=3)
            if claim.claim_type == "cross_role_contrast":
                lines.append(
                    f"- **{dimension}**：主要可写结论来自“{axis}”上的来源角色分工，"
                    f"当前由 {role_text} 支撑；矩阵覆盖度只作为审计背景，不应替代这一证据轴。{citations}"
                )
            else:
                lines.append(
                    f"- **{dimension}**：已有 report-ready evidence axis “{axis}”，"
                    f"当前由 {role_text} 支撑；报告主体应围绕该轴组织研究假设和验证计划，"
                    f"矩阵覆盖度只作为审计背景。{citations}"
                )
        return lines

    lines: list[str] = []
    for dimension in matrix.dimensions:
        cells = sorted(
            [cell for cell in matrix.cells if cell.dimension == dimension],
            key=lambda cell: (coverage_points(cell), cell.confidence, cell.evidence_count),
            reverse=True,
        )
        if not cells:
            continue
        leader = cells[0]
        laggards = [cell for cell in cells if cell.status in {"weak", "unknown"}]
        sentence = (
            f"- **{matrix.dimension_labels.get(dimension, dimension)}**："
            f"{compact_paper_title(leader.paper)} 当前证据最强（{leader.evidence_count} 条，置信度 {leader.confidence:.2f}）。"
        )
        if laggards:
            sentence += " 需要补证：" + "、".join(
                f"{compact_paper_title(cell.paper, 38)}({cell.evidence_count}条)"
                for cell in laggards[:3]
            ) + "。"
        else:
            sentence += " 该推理模式已有可用证据，适合进入横向对比。"
        lines.append(sentence)
    return lines or ["- 当前矩阵暂无足够内容生成解读。"]


def build_dimension_overview(
    dimension: str,
    cells: list[MatrixCell],
    report_ready_claims: list[Claim] | None = None,
) -> list[str]:
    if not cells:
        return ["当前维度暂无矩阵证据。", ""]
    label = DIMENSION_LABELS.get(dimension, dimension)
    total = sum(cell.evidence_count for cell in cells)
    report_ready_claims = report_ready_claims or []
    if report_ready_claims:
        axes = unique_strings(
            [
                cluster_axis_phrase(claim.evidence_cluster_label) if claim.evidence_cluster_label else label
                for claim in report_ready_claims[:4]
            ]
        )
        lines = [
            (
                f"该维度共关联 {total} 条 Evidence；可进入主体的证据轴是 "
                f"{'、'.join(axes[:3]) or label}。下方综合结论只基于 report-ready support，"
                "矩阵覆盖度仅作为审计背景。"
            ),
            "",
        ]
        return lines
    strong = sorted(
        [cell for cell in cells if cell.status in {"strong", "partial"}],
        key=lambda cell: (cell.evidence_count, cell.confidence),
        reverse=True,
    )
    if not strong:
        return [
            (
                f"该维度共关联 {total} 条 Evidence，但当前主要是 weak/single evidence；"
                "不应把它写成主体结论，后续需补充同轴多论文证据。"
            ),
            "",
        ]
    lines = [
        f"该维度共关联 {total} 条 Evidence；可审计支撑论文：{compact_paper_names(strong) or '暂无'}。"
    ]
    best = max(strong, key=lambda cell: (coverage_points(cell), cell.confidence, cell.evidence_count))
    lines.append(f"{label}当前可引用的对比锚点是：{compact_fact(best.summary, 160)}")
    lines.append("")
    return lines


def compact_paper_names(cells: list[MatrixCell], limit: int = 3) -> str:
    if not cells:
        return ""
    names = [cell.paper for cell in cells[:limit]]
    if len(cells) > limit:
        names.append(f"等 {len(cells)} 篇")
    return "、".join(names)


def compact_paper_title(title: str, max_len: int = 56) -> str:
    return compact_fact(title, max_len)


def build_dimension_fallback_points(
    cells: list[MatrixCell],
    citation_numbers: dict[str, int] | None = None,
) -> list[str]:
    if not cells:
        return ["- 当前没有抽取到足够证据形成结论。", ""]
    citation_numbers = citation_numbers or {}
    lines: list[str] = []
    evidence_cells = [
        cell
        for cell in cells
        if cell.evidence_count > 0 and cell.status in {"strong", "partial"}
    ]
    if not evidence_cells:
        return ["- 当前维度只有弱单证据或暂无可引用证据；不应把它写成报告主体结论。", ""]
    for cell in sorted(evidence_cells, key=lambda item: item.evidence_count, reverse=True)[:5]:
        cite = citations_for_ids(cell.evidence_ids, citation_numbers)
        lines.append(
            f"- **{cell.paper}**：{cell.summary} "
            f"（状态：{cell.status}；置信度：{cell.confidence:.2f}）{cite}"
        )
    lines.append("")
    return lines


def compact_fact(text: str, max_len: int = 96) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_len:
        return cleaned
    cutoff = max(1, max_len - 3)
    snippet = cleaned[:cutoff].rstrip()
    boundary = max(snippet.rfind(mark) for mark in ("。", "；", "，", "、", ";", ",", " "))
    if boundary >= int(cutoff * 0.7):
        snippet = snippet[: boundary + 1].rstrip()
    elif cutoff < len(cleaned) and snippet and cleaned[cutoff : cutoff + 1]:
        next_char = cleaned[cutoff]
        if snippet[-1].isascii() and snippet[-1].isalnum() and next_char.isascii() and next_char.isalnum():
            word_boundary = max(snippet.rfind(" "), snippet.rfind(";"), snippet.rfind(","))
            if word_boundary >= int(cutoff * 0.6):
                snippet = snippet[:word_boundary].rstrip()
    return snippet.rstrip(" ,;，；、") + "..."


def best_claim_text(claim: Claim) -> str:
    return claim.final_wording or claim.claim


def scenario_for_dimension(dimension: str) -> str:
    return {
        "gap_driven_reframing": "研究者关注论文如何从瓶颈重新定义问题",
        "cross_domain_synthesis": "研究者关注跨领域知识如何合成为新方法",
        "representation_shift": "研究者关注表征改变带来的能力边界变化",
        "modular_pipeline_composition": "研究者关注模块化管线如何组合出可复用系统",
        "data_evaluation_engineering": "研究者关注数据与评测设计是否支撑结论",
        "principled_probabilistic_modeling": "研究者关注概率假设和不确定性建模是否合理",
        "formal_experimental_tightening": "研究者关注理论与实验是否互相收紧",
        "approximation_engineering": "研究者关注近似方法如何换取可扩展性",
        "inference_time_control": "研究者关注推理时控制如何改变输出质量",
        "structural_inductive_bias": "研究者关注结构归纳偏置如何约束模型行为",
        "multiscale_hierarchical_modeling": "研究者关注多尺度层次如何组织复杂问题",
        "mechanistic_decomposition": "研究者关注机制分解是否解释了方法有效性",
        "adversary_modeling": "研究者关注对抗设定如何暴露鲁棒性边界",
        "numerics_systems_codesign": "研究者关注数值与系统协同如何提升效率",
        "data_centric_optimization": "研究者关注数据中心优化如何提升上限",
    }.get(dimension, "研究者需要快速理解推理模式差异")


def response_for_dimension(target_topic: str, dimension: str) -> str:
    label = DIMENSION_LABELS.get(dimension, dimension)
    return f"围绕 {target_topic} 的{label}证据，说明代表性论文解决的瓶颈、机制和仍需验证的边界。"


def talk_track_for_dimension(target_topic: str, paper: str, dimension: str) -> str:
    label = DIMENSION_LABELS.get(dimension, dimension)
    return (
        f"{paper} 在公开论文中提供了{label}相关证据；比较重点应是它解决了什么瓶颈、"
        f"采用了什么机制，以及它与 {target_topic} 方向内其他工作形成了怎样的互补或冲突。"
    )


def response_for_cell(target_topic: str, target_cell: MatrixCell | None, paper_cell: MatrixCell) -> str:
    label = paper_cell.dimension_label
    if target_cell and target_cell.status in {"strong", "partial"}:
        return (
            f"把讨论转到研究者的{label}证据链：{target_topic} 已有可引用证据，"
            f"重点强调 {compact_fact(target_cell.summary, 150)}"
        )
    return (
        f"承认 {paper_cell.paper} 在{label}上的公开证据更完整；"
        f"{target_topic} 当前应避免强行下判断，并把{label}相关论文和实验结果列为补齐项。"
    )


def talk_track_for_cell(
    target_topic: str,
    paper: str,
    target_cell: MatrixCell | None,
    paper_cell: MatrixCell,
) -> str:
    label = paper_cell.dimension_label
    if target_cell and target_cell.status in {"strong", "partial"}:
        return (
            f"如果您关注{label}，{paper} 的公开资料确实覆盖了这些点；"
            f"但比较应放到具体研究假设里看。{target_topic} 目前能拿出来对照的是："
            f"{compact_fact(target_cell.summary, 130)}。这更适合判断它是否贴合您的研究约束。"
        )
    return (
        f"在{label}上，{paper} 的公开证据更充分，不建议用一句话判断谁绝对更好。"
        f"如果这个模式是研究关键项，应先补充 {target_topic} 的可验证材料；"
        f"再基于实验设置和适用边界判断是否进入下一轮评估。"
    )


def followup_for_cell(target_topic: str, target_cell: MatrixCell | None, paper_cell: MatrixCell) -> str:
    label = paper_cell.dimension_label
    if target_cell and target_cell.status in {"strong", "partial"}:
        return (
            f"整理一页 {label} 证据表，放入 {target_topic} 的论文链接、机制摘要和可复现实验；"
            "研究讨论中只引用已验证点，避免延展到未覆盖能力。"
        )
    return (
        f"补齐 {target_topic} 的{label}公开证据：论文说明、实验结果、方法对比、"
        "应用场景或第三方评测至少补齐两类来源后，再把它作为主要研究结论。"
    )


def gate_status(score: float, pass_threshold: float, fail_threshold: float) -> str:
    if score >= pass_threshold:
        return "pass"
    if score < fail_threshold:
        return "fail"
    return "warn"


def build_matrix_markdown(
    matrix: PaperPatternMatrix,
    citation_numbers: dict[str, int] | None = None,
    focus_evidence_ids: set[str] | None = None,
    max_papers: int = 6,
) -> str:
    citation_numbers = citation_numbers or {}
    papers = select_report_matrix_papers(matrix, focus_evidence_ids or set(), max_papers)
    headers = ["维度", *papers]
    rows = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    by_key = {(cell.paper, cell.dimension): cell for cell in matrix.cells}
    for dimension in matrix.dimensions:
        row = [matrix.dimension_labels.get(dimension, dimension)]
        for paper in papers:
            cell = by_key.get((paper, dimension))
            if not cell or cell.status == "unknown" or cell.evidence_count <= 0:
                row.append("暂无")
            elif cell.status == "weak":
                cite = citations_for_ids(cell.evidence_ids, citation_numbers, limit=2)
                row.append(
                    f"weak · {cell.evidence_count}条 · {cell.confidence:.2f}<br/>"
                    f"仅作审计线索，需同轴多论文证据{cite}"
                )
            else:
                summary = compact_fact(cell.summary, 72)
                cite = citations_for_ids(cell.evidence_ids, citation_numbers, limit=2)
                row.append(f"{cell.status} · {cell.evidence_count}条 · {cell.confidence:.2f}<br/>{summary}{cite}")
        rows.append("|" + "|".join(csv_safe_markdown(value) for value in row) + "|")
    if len(papers) < len(matrix.papers):
        rows.extend(
            [
                "",
                f"> 为保证报告可读性，此处仅展示 {len(papers)} 篇与 report-ready evidence 最相关的论文；完整 matrix 请查看 `exports/matrix.json` 或 `exports/matrix.csv`。",
            ]
        )
    return "\n".join(rows)


def select_report_matrix_papers(
    matrix: PaperPatternMatrix,
    focus_evidence_ids: set[str],
    max_papers: int = 6,
) -> list[str]:
    if len(matrix.papers) <= max_papers:
        return matrix.papers

    paper_scores: dict[str, tuple[int, int, float, float]] = {}
    for paper in matrix.papers:
        cells = [cell for cell in matrix.cells if cell.paper == paper]
        focus_hits = sum(len(set(cell.evidence_ids) & focus_evidence_ids) for cell in cells)
        evidence_count = sum(cell.evidence_count for cell in cells)
        coverage = sum(coverage_points(cell) for cell in cells)
        confidence = average([cell.confidence for cell in cells])
        paper_scores[paper] = (focus_hits, evidence_count, coverage, confidence)

    selected = sorted(
        matrix.papers,
        key=lambda paper: paper_scores.get(paper, (0, 0, 0, 0)),
        reverse=True,
    )[:max_papers]
    return selected or matrix.papers[:max_papers]


def csv_safe_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def build_matrix_csv(matrix: PaperPatternMatrix) -> str:
    rows = ["paper,dimension,dimension_label,status,confidence,evidence_count,summary,evidence_ids"]
    for cell in matrix.cells:
        rows.append(
            ",".join(
                csv_escape(str(value))
                for value in [
                    cell.paper,
                    cell.dimension,
                    cell.dimension_label,
                    cell.status,
                    f"{cell.confidence:.3f}",
                    str(cell.evidence_count),
                    cell.summary,
                    ";".join(cell.evidence_ids),
                ]
            )
        )
    return "\n".join(rows) + "\n"


def build_recommendations_markdown(recommendations: list[OpportunityRecommendation]) -> str:
    if not recommendations:
        return "# Research Recommendations\n\n当前没有足够证据生成建议。\n"
    lines = ["# Research Recommendations", ""]
    for item in recommendations:
        lines.extend(
            [
                f"## {item.title}",
                "",
                f"- Priority: {item.priority}",
                f"- Audience: {item.target_audience}",
                f"- Confidence: {item.confidence:.2f}",
                f"- Recommendation: {item.recommendation}",
                f"- Rationale: {item.rationale}",
                f"- Evidence: {', '.join(item.evidence_ids) or 'None'}",
                "",
            ]
        )
    return "\n".join(lines)


def build_executive_summary(
    request: ResearchRequest,
    metrics: RunMetrics,
    observability: ObservabilitySnapshot,
    recommendations: list[OpportunityRecommendation],
    claims: list[Claim] | None = None,
) -> str:
    ready_findings = report_ready_finding_lines(claims or [], limit=3, max_len=300) if claims is not None else []
    lines = [
        f"# {request.project_name} · Executive Summary",
        "",
        f"研究方向：{request.target_topic}",
        f"本次运行获得 {metrics.sources_fetched} 篇可用内容，抽取 {metrics.evidence_count} 条 Evidence，生成 {metrics.claim_count} 条 Claim。",
        f"综合覆盖度：{observability.evidence_coverage_score:.0%}；报告可信度：{observability.report_confidence:.0%}。",
        "",
        "## Report-ready Findings",
        "",
        *(ready_findings or ["- 本轮尚未形成可进入报告主体的范围限定综合结论；应继续补证或降低结论强度。"]),
        "",
        "## Top Recommendations",
        "",
    ]
    for item in recommendations[:3]:
        lines.append(f"- {item.title}：{item.recommendation}（{item.priority}）")
    if not recommendations:
        lines.append("- 当前证据不足，建议先补充来源。")
    return "\n".join(lines) + "\n"


def build_methodology(request: ResearchRequest, observability: ObservabilitySnapshot) -> str:
    gates = "\n".join(
        f"- {gate.name}: {gate.status}, score={gate.score:.2f}, {gate.message}"
        for gate in observability.quality_gates
    )
    return (
        f"# Methodology\n\n"
        f"ScholarInsight 对「{request.project_name}」执行了 5 Agent deep research 流程："
        "ResearchPlanningAgent 规划研究范围与质量规则，SourceResearchAgent 检索本地论文库并补充来源，"
        "EvidenceStructuringAgent 抽取可溯源事实，AnalysisAndReviewAgent 生成结论、反方审查并整理矩阵/建议，"
        "ReportComposerAgent 生成本地 Markdown、JSON 与 CSV 交付文件。\n\n"
        "所有产物均保存在本 Run 目录下，可直接审计 JSON/JSONL/Markdown/CSV 文件。\n\n"
        "## Quality Gates\n\n"
        f"{gates}\n"
    )
