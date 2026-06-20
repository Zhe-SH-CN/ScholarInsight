"""Core schemas for ScholarInsight academic paper reasoning analysis."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


DEFAULT_DIMENSIONS = [
    "gap_driven_reframing",
    "cross_domain_synthesis",
    "representation_shift",
    "modular_pipeline_composition",
    "data_evaluation_engineering",
    "principled_probabilistic_modeling",
    "formal_experimental_tightening",
    "approximation_engineering",
    "inference_time_control",
    "structural_inductive_bias",
    "multiscale_hierarchical_modeling",
    "mechanistic_decomposition",
    "adversary_modeling",
    "numerics_systems_codesign",
    "data_centric_optimization",
]

TOPIC_FAMILY_DIMENSIONS: dict[str, list[str]] = {
    "rag_with_knowledge_graphs": [
        "kg_rag_architecture",
        "graph_retrieval_grounding",
        "kg_rag_evaluation",
        "multi_hop_kg_reasoning",
        "kg_construction_for_rag",
        "rag_kg_boundary_analysis",
    ],
    "scientific_reasoning_llms": [
        "scientific_problem_benchmarking",
        "tool_augmented_scientific_reasoning",
        "domain_grounding_verification",
        "multimodal_scientific_reasoning",
        "scientific_error_analysis",
        "lab_workflow_reasoning",
    ],
    "mathematical_reasoning": [
        "math_benchmark_evaluation",
        "formal_proof_symbolic_reasoning",
        "program_tool_augmented_solving",
        "self_consistency_search_verification",
        "natural_language_to_formal_math",
        "math_error_diagnosis",
    ],
    "causal_reasoning_llms": [
        "llm_causal_benchmarking",
        "causal_intervention_counterfactual",
        "causal_explanation_mechanism",
        "causal_pitfall_robustness",
        "causal_tool_symbolic_integration",
        "causal_reasoning_evaluation_protocol",
    ],
    "counterfactual_inference": [
        "treatment_effect_estimation",
        "core_counterfactual_inference",
        "counterfactual_explanation_fairness",
        "temporal_counterfactual_estimation",
        "identifiability_assumption_sensitivity",
        "counterfactual_benchmarking",
    ],
    "multi_hop_graph_reasoning": [
        "graph_reasoning_benchmark",
        "kgqa_or_graph_reasoning",
        "core_multi_hop_graph_reasoning",
        "semantic_parsing_grounding",
        "path_composition_reasoning",
        "graph_retrieval_boundary",
    ],
}

DIMENSION_LABELS: dict[str, str] = {
    "gap_driven_reframing": "痛点驱动重构",
    "cross_domain_synthesis": "跨领域综合",
    "representation_shift": "表征转换",
    "modular_pipeline_composition": "模块化管线",
    "data_evaluation_engineering": "数据评估工程",
    "principled_probabilistic_modeling": "概率建模",
    "formal_experimental_tightening": "理论实验迭代",
    "approximation_engineering": "近似工程",
    "inference_time_control": "推理时控制",
    "structural_inductive_bias": "结构归纳偏置",
    "multiscale_hierarchical_modeling": "多尺度分层",
    "mechanistic_decomposition": "机制分解",
    "adversary_modeling": "对抗建模",
    "numerics_systems_codesign": "数值系统协同",
    "data_centric_optimization": "数据中心优化",
    "kg_rag_architecture": "KG-RAG 架构机制",
    "graph_retrieval_grounding": "图检索与 grounding",
    "kg_rag_evaluation": "KG-RAG 评测协议",
    "multi_hop_kg_reasoning": "多跳 KG 推理",
    "kg_construction_for_rag": "RAG 用 KG 构建",
    "rag_kg_boundary_analysis": "RAG/KG 边界分析",
    "application_boundary_cases": "应用边界案例",
    "scientific_problem_benchmarking": "科学问题基准",
    "tool_augmented_scientific_reasoning": "工具增强科学推理",
    "domain_grounding_verification": "领域 grounding 与验证",
    "multimodal_scientific_reasoning": "多模态科学推理",
    "scientific_error_analysis": "科学推理错误分析",
    "lab_workflow_reasoning": "实验流程推理",
    "math_benchmark_evaluation": "数学基准评测",
    "formal_proof_symbolic_reasoning": "形式证明与符号推理",
    "program_tool_augmented_solving": "程序/工具增强求解",
    "self_consistency_search_verification": "搜索验证与自一致性",
    "natural_language_to_formal_math": "自然语言到形式数学",
    "math_error_diagnosis": "数学错误诊断",
    "llm_causal_benchmarking": "LLM 因果基准",
    "causal_intervention_counterfactual": "干预与反事实推理",
    "causal_explanation_mechanism": "因果解释机制",
    "causal_pitfall_robustness": "因果陷阱与鲁棒性",
    "causal_tool_symbolic_integration": "因果工具/符号集成",
    "causal_reasoning_evaluation_protocol": "因果评测协议",
    "treatment_effect_estimation": "处理效应估计",
    "core_counterfactual_inference": "核心反事实推断",
    "counterfactual_explanation_fairness": "反事实解释与公平性",
    "temporal_counterfactual_estimation": "时序反事实估计",
    "identifiability_assumption_sensitivity": "可识别性与假设敏感性",
    "counterfactual_benchmarking": "反事实基准评测",
    "graph_reasoning_benchmark": "图推理基准",
    "kgqa_or_graph_reasoning": "KGQA/图推理",
    "core_multi_hop_graph_reasoning": "核心多跳图推理",
    "semantic_parsing_grounding": "语义解析 grounding",
    "path_composition_reasoning": "路径组合推理",
    "graph_retrieval_boundary": "图检索边界",
    "other": "其他",
}


def topic_family_for_dimensions(target_topic: str) -> str:
    """Return a coarse topic family used only to choose analysis dimensions."""
    topic = target_topic.lower()
    if "rag" in topic and ("knowledge graph" in topic or "kg" in topic):
        return "rag_with_knowledge_graphs"
    if "scientific reasoning" in topic:
        return "scientific_reasoning_llms"
    if "mathematical reasoning" in topic or topic.strip() == "math reasoning":
        return "mathematical_reasoning"
    if "causal reasoning" in topic and ("llm" in topic or "language model" in topic):
        return "causal_reasoning_llms"
    if "counterfactual inference" in topic:
        return "counterfactual_inference"
    if "multi-hop" in topic and "graph" in topic:
        return "multi_hop_graph_reasoning"
    return "generic"


def dimensions_for_topic(target_topic: str) -> list[str]:
    family = topic_family_for_dimensions(target_topic)
    return TOPIC_FAMILY_DIMENSIONS.get(family, DEFAULT_DIMENSIONS).copy()


class ResearchRequest(BaseModel):
    """User-facing request for a real research run."""

    project_name: str = Field(default="论文推理模式分析", min_length=1, max_length=120)
    target_topic: str = Field(default="Retrieval-Augmented Generation", min_length=1, max_length=80)
    topic_description: str = Field(default="", max_length=300)
    seed_papers: list[str] = Field(default_factory=lambda: [])
    analysis_dimensions: list[str] = Field(default_factory=lambda: DEFAULT_DIMENSIONS.copy())
    research_goal: str = Field(
        default="分析该研究方向中论文之间的创新关系和推理模式分布", max_length=1000
    )
    seed_urls: list[str] = Field(default_factory=list)
    max_sources: int = Field(default=150, ge=1, le=1000)
    max_sources_per_query: int = Field(default=3, ge=1, le=8)
    auto_discover_sources: bool = True
    max_search_rounds: int = Field(default=3, ge=1, le=8)
    max_research_loops: int | None = Field(default=None, ge=1, le=8)

    @model_validator(mode="before")
    @classmethod
    def migrate_competeinsight_fields(cls, value: Any) -> Any:
        """Accept legacy payloads without losing the user's topic."""
        if isinstance(value, dict):
            data = dict(value)
            if not data.get("target_topic") and data.get("target_product"):
                data["target_topic"] = data["target_product"]
            if not data.get("topic_description") and data.get("product_description"):
                data["topic_description"] = data["product_description"]
            if not data.get("seed_papers") and data.get("competitors"):
                data["seed_papers"] = data["competitors"]
            if not data.get("analysis_dimensions"):
                data["analysis_dimensions"] = dimensions_for_topic(str(data.get("target_topic") or ""))
            return data
        return value

    @field_validator("seed_papers", "analysis_dimensions", mode="before")
    @classmethod
    def split_string_list(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("seed_urls", mode="before")
    @classmethod
    def split_urls(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [item.strip() for item in value.splitlines() if item.strip()]
        return value

    @property
    def target_product(self) -> str:
        """Compatibility for old tests/manifests."""
        return self.target_topic

    @property
    def product_description(self) -> str:
        """Compatibility for old tests/manifests."""
        return self.topic_description

    @property
    def competitors(self) -> list[str]:
        """Compatibility for old tests/manifests."""
        return self.seed_papers


class QuickExtractRequest(BaseModel):
    url: HttpUrl
    run_id: str | None = None
    source_type: str = "other"


class SourceCandidate(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""
    content: str = ""
    content_source: str = ""
    source_type: str = "other"
    query: str = ""
    score: float = Field(default=0.5, ge=0, le=1)
    embedding_score: float | None = None
    lexical_score: float | None = None
    reranker_score: float | None = None
    relevance_score: float = Field(default=0.5, ge=0, le=1)
    relevance_label: str = "unscored"
    rejection_reason: str = ""
    source_subtype: str = "unclassified"
    source_subtype_reason: str = ""
    source_provider: str = ""
    published_at: datetime | None = None
    date_source: str = "unknown"  # exa / tavily / zhihu_edit_time / unknown


class SourceTask(BaseModel):
    task_id: str
    entity: str
    dimension: str = "other"
    intent: Literal["official", "docs", "review", "news", "comparison", "survey", "benchmark"] = "official"
    query: str
    expected_source_types: list[str] = Field(default_factory=list)
    rationale: str = ""

    @field_validator("intent", mode="before")
    @classmethod
    def migrate_legacy_intent(cls, value: Any) -> str:
        legacy_map = {
            "pricing": "comparison",
            "enterprise": "official",
            "changelog": "news",
        }
        return legacy_map.get(str(value or "official"), str(value or "official"))


class ResearchPlan(BaseModel):
    research_goal: str
    papers: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    source_tasks: list[SourceTask] = Field(default_factory=list)
    required_agents: list[str] = Field(default_factory=list)
    quality_rules: list[str] = Field(default_factory=list)
    notes: str = ""
    planned_by: str = "ChiefResearchPlanner"

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_plan_fields(cls, value: Any) -> Any:
        if isinstance(value, dict) and not value.get("papers") and value.get("competitors"):
            data = dict(value)
            data["papers"] = data["competitors"]
            return data
        return value


class SourceDocument(BaseModel):
    source_id: str
    run_id: str
    url: str
    title: str
    content: str
    excerpt: str
    source_type: str
    http_status: int | None = None
    content_hash: str
    fetched_at: datetime
    parser: str
    provider: str = ""
    query: str = ""
    content_source: str = ""
    source_score: float = Field(default=0.5, ge=0, le=1)
    relevance_score: float = Field(default=0.5, ge=0, le=1)
    relevance_label: str = "unscored"
    rejection_reason: str = ""
    source_subtype: str = "unclassified"
    source_subtype_reason: str = ""
    embedding_score: float | None = None
    lexical_score: float | None = None
    reranker_score: float | None = None
    ok: bool = True
    error: str | None = None
    published_at: datetime | None = None
    date_source: str = "unknown"


class Evidence(BaseModel):
    evidence_id: str
    run_id: str
    url: str
    title: str
    content: str
    fetched_at: datetime
    source_type: str
    dimension: str = "other"
    dimension_label: str = "其他"
    paper: str | None = None
    fact: str
    quote: str
    source_title: str
    source_url: str
    source_id: str | None = None
    confidence: float = Field(default=0.5, ge=0, le=1)
    reasoning_pattern: str = ""
    bottleneck: str = ""
    mechanism: str = ""
    source_subtype: str = "unclassified"
    source_subtype_reason: str = ""
    authority_score: float = Field(default=0.5, ge=0, le=1)
    freshness_score: float = Field(default=0.5, ge=0, le=1)
    relevance_score: float = Field(default=0.5, ge=0, le=1)
    extracted_by_agent: str = "EvidenceStructuringAgent"
    extracted_by_skill: str = "skill.evidence_extraction.v1"
    status: Literal["draft", "verified", "rejected"] = "verified"
    published_at: datetime | None = None


class EvidenceSummary(BaseModel):
    evidence_id: str
    dimension: str
    dimension_label: str
    paper: str | None = None
    fact: str
    quote_preview: str
    source_title: str
    source_url: str
    source_type: str
    confidence: float
    fetched_at: datetime
    reasoning_pattern: str = ""
    bottleneck: str = ""
    mechanism: str = ""
    source_subtype: str = "unclassified"
    source_subtype_reason: str = ""


class RedTeamNote(BaseModel):
    risk_type: str
    comment: str
    suggested_action: str
    severity: Literal["low", "medium", "high"] = "medium"

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity(cls, v: Any) -> str:
        if v == "critical":
            return "high"
        return v


class Claim(BaseModel):
    claim_id: str
    run_id: str
    dimension: str
    dimension_label: str
    claim: str
    supporting_evidence_ids: list[str]
    counter_evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)
    risk_level: Literal["low", "medium", "high"] = "medium"
    reasoning_summary: str
    claim_type: str = "descriptive"
    source_paper_count: int = 0
    evidence_support_level: str = "unknown"
    supporting_source_subtypes: list[str] = Field(default_factory=list)
    supporting_source_subtype_counts: dict[str, int] = Field(default_factory=dict)
    supporting_source_subtype_paper_counts: dict[str, int] = Field(default_factory=dict)
    evidence_cluster_id: str = ""
    evidence_cluster_label: str = ""
    backlog_reason: str = ""
    verification_status: Literal[
        "draft",
        "verified",
        "needs_evidence",
        "challenged",
        "rejected",
        "included_in_report",
    ] = "draft"
    red_team_notes: list[RedTeamNote] = Field(default_factory=list)
    generated_by_agent: str = "AnalysisAndReviewAgent"
    generated_by_skill: str = "skill.claim_generation.v1"
    final_wording: str | None = None


class CounterexampleAuditRow(BaseModel):
    audit_id: str
    target_claim_id: str
    target_dimension: str
    target_dimension_label: str
    target_axis: str = ""
    source_title: str = ""
    source_url: str = ""
    source_subtype: str = "unclassified"
    source_subtype_reason: str = ""
    relevance_score: float = Field(default=0, ge=0, le=1)
    relevance_label: str = "unscored"
    rejection_reason: str = ""
    query: str = ""
    audit_role: str = "boundary_challenge"
    counterexample_type: str = "manual_boundary"
    semantic_quality: float = Field(default=1.0, ge=0, le=1)
    quality_reason: str = ""
    report_visible: bool = True
    boundary_challenge: str = ""
    audit_only_reason: str = ""


class TraceEvent(BaseModel):
    event_id: str
    run_id: str
    timestamp: datetime
    node: str
    phase: Literal["start", "progress", "complete", "error"]
    status: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)


class RunMetrics(BaseModel):
    source_candidates: int = 0
    sources_rejected: int = 0
    sources_fetched: int = 0
    sources_failed: int = 0
    evidence_count: int = 0
    claim_count: int = 0
    verified_claim_count: int = 0
    challenged_claim_count: int = 0
    matrix_cell_count: int = 0
    recommendation_count: int = 0
    average_evidence_confidence: float = 0
    coverage_score: float = 0


class RunStatus(BaseModel):
    run_id: str
    project_name: str
    target_topic: str
    owner: str | None = None
    status: Literal["queued", "running", "completed", "failed", "stopped"]
    current_stage: str = "queued"
    started_at: datetime
    finished_at: datetime | None = None
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    node_status: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_status_fields(cls, value: Any) -> Any:
        if isinstance(value, dict) and not value.get("target_topic") and value.get("target_product"):
            data = dict(value)
            data["target_topic"] = data["target_product"]
            return data
        return value


class EvidenceLink(BaseModel):
    evidence_id: str
    source_title: str
    source_url: str
    quote_preview: str
    confidence: float = Field(default=0.5, ge=0, le=1)


class EvidenceCluster(BaseModel):
    cluster_id: str
    dimension: str
    dimension_label: str
    label: str
    summary: str = ""
    mechanism: str = ""
    bottleneck: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    papers: list[str] = Field(default_factory=list)
    evidence_count: int = 0
    independent_paper_count: int = 0
    average_confidence: float = Field(default=0, ge=0, le=1)
    verified_claim_ids: list[str] = Field(default_factory=list)
    challenged_claim_ids: list[str] = Field(default_factory=list)
    status: Literal["verified", "candidate", "single_paper", "backlog"] = "candidate"


class PaperProfile(BaseModel):
    paper: str
    summary: str
    evidence_count: int = 0
    source_count: int = 0
    average_confidence: float = Field(default=0, ge=0, le=1)
    strongest_dimensions: list[str] = Field(default_factory=list)
    weak_or_unknown_dimensions: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class MatrixCell(BaseModel):
    paper: str
    dimension: str
    dimension_label: str
    summary: str
    evidence_count: int = 0
    confidence: float = Field(default=0, ge=0, le=1)
    source_types: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    status: Literal["strong", "partial", "weak", "unknown"] = "unknown"


class PaperPatternMatrix(BaseModel):
    generated_at: datetime
    papers: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    dimension_labels: dict[str, str] = Field(default_factory=dict)
    cells: list[MatrixCell] = Field(default_factory=list)
    profiles: list[PaperProfile] = Field(default_factory=list)
    coverage_by_paper: dict[str, float] = Field(default_factory=dict)
    coverage_by_dimension: dict[str, float] = Field(default_factory=dict)


class OpportunityRecommendation(BaseModel):
    recommendation_id: str
    title: str
    recommendation: str
    priority: Literal["low", "medium", "high"] = "medium"
    target_audience: Literal["researcher", "advisor", "student", "general"] = "researcher"
    rationale: str
    expected_value: str
    based_on_claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)

    @field_validator("target_audience", mode="before")
    @classmethod
    def migrate_legacy_audience(cls, value: Any) -> str:
        if value in {"executive", "pm", "sales", "investor"}:
            return "researcher"
        return value or "researcher"


class QualityGate(BaseModel):
    gate_id: str
    name: str
    status: Literal["pass", "warn", "fail"] = "warn"
    score: float = Field(default=0, ge=0, le=1)
    message: str
    suggested_action: str = ""


class ObservabilitySnapshot(BaseModel):
    generated_at: datetime
    total_duration_seconds: float = 0
    agent_count: int = 0
    skill_calls: int = 0
    tool_calls: int = 0
    source_mix: dict[str, int] = Field(default_factory=dict)
    dimension_coverage: dict[str, float] = Field(default_factory=dict)
    paper_coverage: dict[str, float] = Field(default_factory=dict)
    claim_pass_rate: float = Field(default=0, ge=0, le=1)
    red_team_challenge_rate: float = Field(default=0, ge=0, le=1)
    evidence_coverage_score: float = Field(default=0, ge=0, le=1)
    report_confidence: float = Field(default=0, ge=0, le=1)
    quality_gates: list[QualityGate] = Field(default_factory=list)
    export_files: dict[str, str] = Field(default_factory=dict)


class EvidenceGraphNode(BaseModel):
    id: str
    label: str
    node_type: Literal["claim", "evidence", "source", "paper", "dimension"]
    score: float = Field(default=0.5, ge=0, le=1)
    meta: dict[str, Any] = Field(default_factory=dict)


class EvidenceGraphEdge(BaseModel):
    source: str
    target: str
    edge_type: str
    weight: float = Field(default=0.5, ge=0, le=1)


class EvidenceGraph(BaseModel):
    generated_at: datetime
    nodes: list[EvidenceGraphNode] = Field(default_factory=list)
    edges: list[EvidenceGraphEdge] = Field(default_factory=list)


class RunDetail(BaseModel):
    status: RunStatus
    request: ResearchRequest
    plan: ResearchPlan | None = None
    dag: dict[str, Any] | None = None
    sources: list[SourceCandidate] = Field(default_factory=list)
    documents: list[SourceDocument] = Field(default_factory=list)
    evidence: list[EvidenceSummary] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    matrix: PaperPatternMatrix | None = None
    recommendations: list[OpportunityRecommendation] = Field(default_factory=list)
    observability: ObservabilitySnapshot | None = None
    evidence_graph: EvidenceGraph | None = None
    evidence_clusters: list[EvidenceCluster] = Field(default_factory=list)
    report_markdown: str = ""
    executive_summary_markdown: str = ""
    methodology_markdown: str = ""
    report_path: str | None = None


class RunStarted(BaseModel):
    run_id: str
    status: str


class QuickExtractResponse(BaseModel):
    status: RunStatus
    evidence: list[Evidence]


class ResearchGap(BaseModel):
    """Analysis Agent 反馈给 Planning Agent 的信息缺口。"""
    dimension: str
    paper: str
    reason: str
    priority: Literal["high", "medium", "low"] = "medium"
    suggested_queries: list[str] = Field(default_factory=list)


class ResearchFeedback(BaseModel):
    """Analysis Agent 向 Planning Agent 的整体反馈。"""
    needs_more_research: bool = False
    gaps: list[ResearchGap] = Field(default_factory=list)
    loop_round: int = 1
    coverage_score: float = 0.0
    message: str = ""


class SearchMemory(BaseModel):
    """跨 Search 调用保留的发现状态，供外层补充研究继续去重。"""

    seen_urls: list[str] = Field(default_factory=list)
    seen_queries: list[str] = Field(default_factory=list)
    rejected_urls: list[str] = Field(default_factory=list)
    remaining_source_slots: int = 0
    resource_limit_reached: bool = False
    loop_round: int = 1
    feedback_message: str = ""
