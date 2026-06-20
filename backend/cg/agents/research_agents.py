"""Five focused agents for the runnable deep-research workflow."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

from cg.agents.runtime import BaseAgent
from cg.repositories.base import append_jsonl
from cg.schemas.research import (
    Claim,
    PaperPatternMatrix,
    DEFAULT_DIMENSIONS,
    DIMENSION_LABELS,
    Evidence,
    ObservabilitySnapshot,
    OpportunityRecommendation,
    RedTeamNote,
    ResearchFeedback,
    ResearchGap,
    ResearchPlan,
    ResearchRequest,
    RunMetrics,
    SearchMemory,
    SourceCandidate,
    SourceDocument,
    SourceTask,
)
from cg.tools.search import SearchTool, classify_source, normalize_url


RESEARCH_AGENT_FLOW = [
    "ResearchPlanningAgent",
    "SourceResearchAgent",
    "EvidenceStructuringAgent",
    "AnalysisAndReviewAgent",
    "ReportComposerAgent",
]

AGENT_SKILLS: dict[str, list[str]] = {
    "ResearchPlanningAgent": [
        "skill.scope_planning.v1",
        "skill.query_design.v1",
        "skill.quality_contract.v1",
    ],
    "SourceResearchAgent": [
        "skill.source_discovery.v1",
        "skill.source_quality_scoring.v1",
        "skill.web_collection.v1",
    ],
    "EvidenceStructuringAgent": [
        "skill.evidence_extraction.v1",
        "skill.quote_grounding.v1",
        "skill.source_attribution.v1",
    ],
    "AnalysisAndReviewAgent": [
        "skill.claim_generation.v1",
        "skill.red_team_review.v1",
        "skill.asset_synthesis.v1",
    ],
    "ReportComposerAgent": [
        "skill.report_writing.v1",
        "skill.citation_rendering.v1",
        "skill.local_export.v1",
    ],
}

AGENT_ALIASES = {
    "ChiefResearchPlanner": "ResearchPlanningAgent",
    "ChiefResearchPlannerAgent": "ResearchPlanningAgent",
    "Planner": "ResearchPlanningAgent",
    "SourceScout": "SourceResearchAgent",
    "SourceScoutAgent": "SourceResearchAgent",
    "CollectionAgent": "SourceResearchAgent",
    "EvidenceExtractorAgent": "EvidenceStructuringAgent",
    "EvidenceExtractionAgent": "EvidenceStructuringAgent",
    "ClaimGenerationAgent": "AnalysisAndReviewAgent",
    "RedTeamAgent": "AnalysisAndReviewAgent",
    "InsightSynthesisAgent": "AnalysisAndReviewAgent",
    "AnalystAgent": "AnalysisAndReviewAgent",
    "InsightSynthesizer": "AnalysisAndReviewAgent",
    "ReportWriterAgent": "ReportComposerAgent",
    "FinalQAAgent": "ReportComposerAgent",
}

SUPPLEMENTAL_APPLICATION_REJECTION_REASON = (
    "supplemental RAG+KG gap search excludes application-case papers for generic topics"
)

APPLICATION_TOPIC_MARKERS = (
    "application",
    "recommendation",
    "recommender",
    "medical",
    "clinical",
    "healthcare",
    "pathology",
    "radiology",
    "biomedical",
    "robot",
    "robotic",
    "planning",
    "low-resource",
    "low resource",
    "low-resourced",
    "multilingual",
)


def target_topic_allows_application_sources(target_topic: str) -> bool:
    topic = target_topic.lower()
    return any(marker in topic for marker in APPLICATION_TOPIC_MARKERS)


def should_reject_supplemental_application_source(
    candidate: SourceCandidate,
    *,
    target_topic: str,
    loop_round: int,
) -> bool:
    return (
        loop_round > 1
        and candidate.source_subtype == "application_case"
        and not target_topic_allows_application_sources(target_topic)
    )


class ResearchPlanningAgent(BaseAgent):
    name = "ResearchPlanningAgent"
    skill_ids = AGENT_SKILLS[name]
    skill_id = skill_ids[0]

    async def plan(
        self,
        request: ResearchRequest,
        feedback: ResearchFeedback | None = None,
        loop_round: int = 1,
    ) -> ResearchPlan:
        """生成研究计划。第 2 轮起可接受 Analysis 的反馈，聚焦补充缺口。"""
        deterministic = deterministic_plan(request)
        dimension_catalog = format_dimension_catalog(request.analysis_dimensions or DEFAULT_DIMENSIONS)

        # 构建反馈上下文
        feedback_context = ""
        if feedback and feedback.gaps:
            gaps_str = "\n".join(
                f"- 论文【{g.paper}】的【{DIMENSION_LABELS.get(g.dimension, g.dimension)}】"
                f"维度：{g.reason}（优先级：{g.priority}）"
                + (f"\n  建议查询：{', '.join(g.suggested_queries)}" if g.suggested_queries else "")
                for g in feedback.gaps
            )
            feedback_context = (
                f"\n\n【第 {loop_round} 轮补充搜索反馈】：\n"
                f"上一轮搜索后分析发现以下信息缺口，请专门针对这些缺口设计补充搜索任务：\n{gaps_str}"
            )

        data = await self.invoke_json(
            system=(
                "你是论文推理模式分析规划智能体（ResearchPlanningAgent），是整个研究流程的战略大脑。\n"
                "\n"
                "【核心职责】\n"
                "把用户的研究方向拆解为覆盖全面、来源多元、可执行的论文检索计划。\n"
                "你需要像一位资深学术研究者一样思考：要证明一个推理模式判断，需要哪些类型的证据？\n"
                "\n"
                "【检索策略框架】\n"
                "对每个研究方向，必须覆盖以下信息层次：\n"
                "  A. 核心论文层：该方向的代表性论文、奠基性工作、最新进展\n"
                "  B. 方法创新层：新提出的模型架构、训练方法、推理技术\n"
                "  C. 应用落地层：实际应用场景、系统实现、工程优化\n"
                "  D. 评测对比层：基准测试、消融实验、与现有方法的对比\n"
                "\n"
                "【查询设计规则】\n"
                "1. 每个 source_task 提供英文查询，用于本地论文库 embedding 检索\n"
                "2. 查询应聚焦于具体的技术方法、模型架构、训练策略\n"
                "3. dimension 只能从本次请求的分析维度中选择，不要创造列表外的新 key\n"
                f"\n【本次允许的分析维度】\n{dimension_catalog}\n"
                "\n"
                "【输出格式】纯 JSON，不加 Markdown 代码块：\n"
                "{\n"
                '  "research_goal": "精炼后的研究目标（一句话）",\n'
                '  "dimensions": ["推理模式1","推理模式2"],\n'
                '  "source_tasks": [\n'
                '    {\n'
                '      "task_id": "task_01",\n'
                '      "entity": "研究方向",\n'
                '      "dimension": "推理模式英文key",\n'
                '      "intent": "official",\n'
                '      "query": "具体英文检索查询",\n'
                '      "expected_source_types": ["academic_paper"],\n'
                '      "rationale": "这条查询能找到什么类型的证据，用于支撑哪个推理模式判断"\n'
                '    }\n'
                "  ],\n"
                '  "quality_rules": [\n'
                '    "每条关键结论至少绑定 2 条不同论文的 Evidence",\n'
                '    "推理模式判断必须有方法创新层来源支撑",\n'
                '    "机制分析必须来自论文原文，不接受推测"\n'
                "  ],\n"
                '  "notes": "研究难点或注意事项"\n'
                "}\n"
                "\n"
                "【数量要求】每个分析维度至少 1 条 source_task，总量 8-20 个；优先保证 query 精确相关。\n"
                "intent 只能是：official / docs / review / comparison"
            ),
            user=(
                f"项目名：{request.project_name}\n"
                f"研究方向：{request.target_topic}"
                + (f"（方向描述：{request.topic_description}）" if request.topic_description else "")
                + f"\n分析维度：{', '.join(request.analysis_dimensions)}\n"
                f"研究目标：{request.research_goal}"
                + feedback_context
            ),
        )

        if not data:
            return deterministic

        # 解析 LLM 返回的 source_tasks，支持 query_en / query_zh 双字段
        raw_tasks = data.get("source_tasks") or []
        parsed_tasks: list[SourceTask] = []
        allowed_intents = {"official", "docs", "review", "news", "comparison"}
        for i, row in enumerate(raw_tasks[:32]):
            if not isinstance(row, dict):
                continue
            # 优先用 query_en 作为英文 query，再回退到 query
            query_en = str(row.get("query_en") or row.get("query") or "").strip()
            query_zh = str(row.get("query_zh") or "").strip()
            use_zhihu = bool(row.get("use_zhihu", False))
            if not query_en and not query_zh:
                continue
            intent = str(row.get("intent") or "official").lower()
            if intent not in allowed_intents:
                intent = "official"
            dimension = str(row.get("dimension") or "other")
            if dimension not in DIMENSION_LABELS:
                dimension = "other"
            entity = str(row.get("entity") or "unknown").strip() or "unknown"
            # 生成英文任务
            if query_en:
                parsed_tasks.append(SourceTask(
                    task_id=str(row.get("task_id") or f"task_{i + 1:02d}_en"),
                    entity=entity,
                    dimension=dimension,
                    intent=intent,  # type: ignore[arg-type]
                    query=query_en,
                    expected_source_types=coerce_str_list(row.get("expected_source_types")),
                    rationale=str(row.get("rationale") or ""),
                ))
            # 生成中文任务（标注为知乎查询）
            if query_zh and use_zhihu:
                parsed_tasks.append(SourceTask(
                    task_id=f"task_{i + 1:02d}_zh",
                    entity=entity,
                    dimension=dimension,
                    intent=intent,  # type: ignore[arg-type]
                    query=query_zh,
                    expected_source_types=["user_review"],
                    rationale=f"中文用户声音搜索（知乎）: {query_zh}",
                ))

        if not parsed_tasks:
            parsed_tasks = deterministic.source_tasks

        queries = [task.query for task in parsed_tasks]
        return ResearchPlan(
            research_goal=str(data.get("research_goal") or deterministic.research_goal),
            papers=coerce_str_list(data.get("papers")) or deterministic.papers,
            dimensions=normalize_dimensions(coerce_str_list(data.get("dimensions"))) or deterministic.dimensions,
            queries=queries[:32],
            source_tasks=parsed_tasks[:32],
            required_agents=normalize_agent_flow(
                coerce_str_list(data.get("required_agents")) or deterministic.required_agents
            ),
            quality_rules=coerce_str_list(data.get("quality_rules")) or deterministic.quality_rules,
            notes=str(data.get("notes") or ""),
            planned_by=self.name if self.llm_enabled else f"{self.name}:deterministic",
        )


class SourceResearchAgent(BaseAgent):
    """ReAct 自主搜索智能体：自主决定搜什么、何时停止，而非执行预定查询列表。"""
    name = "SourceResearchAgent"
    skill_ids = AGENT_SKILLS[name]
    skill_id = skill_ids[0]

    async def discover(
        self,
        request: ResearchRequest,
        plan: ResearchPlan,
        search: Any,
        memory: SearchMemory | None = None,
        feedback: ResearchFeedback | None = None,
        loop_round: int = 1,
    ) -> list[SourceCandidate]:
        """两层搜索：先执行 Planning 查询铺底，再由 Search LLM ReAct 补洞。"""
        candidates: list[SourceCandidate] = []
        seen_urls: set[str] = set(memory.seen_urls if memory else [])
        seen_queries: set[str] = set(memory.seen_queries if memory else [])
        remaining_slots = memory.remaining_source_slots if memory and memory.remaining_source_slots > 0 else request.max_sources
        if memory and memory.resource_limit_reached:
            await self.record_llm_event(
                "progress",
                "resource_limit_reached",
                f"来源资源上限 {request.max_sources} 已达到，Search Agent 不再发起新搜索",
                {"max_sources": request.max_sources, "seen_urls": len(seen_urls)},
            )
            return []

        # 种子 URL 直接加入
        for url in request.seed_urls:
            normalized = normalize_url(url)
            if normalized and normalized not in seen_urls:
                seen_urls.add(normalized)
                candidates.append(SourceCandidate(
                    url=normalized,
                    title=urlparse(normalized).netloc or url,
                    source_type=classify_source(url),
                    query="seed_url",
                    score=0.88,
                ))

        if not request.auto_discover_sources:
            return dedupe_source_candidates(candidates)

        max_rounds = getattr(request, "max_search_rounds", 3)
        use_llm_reasoning = isinstance(search, SearchTool)
        no_new_consecutive = 0  # 连续无新增轮次计数
        per_provider_count = max(
            1,
            min(
                request.max_sources_per_query or self.ctx.settings.cg_search_default_results_per_provider,
                self.ctx.settings.cg_search_max_results_per_provider,
            ),
        )

        for round_num in range(1, max_rounds + 1):
            if remaining_slots <= 0:
                await self.record_llm_event(
                    "progress",
                    "resource_limit_reached",
                    f"来源资源上限 {request.max_sources} 已达到，停止后续搜索",
                    {"max_sources": request.max_sources, "seen_urls": len(seen_urls)},
                )
                break
            # 第一层：第 1 轮只执行 Planning 的几十条 query，不调用 Search LLM。
            if round_num == 1:
                next_tasks = unique_tasks_by_query(plan.source_tasks)
                await self.record_llm_event(
                    "progress", "planned_search",
                    f"执行 {len(next_tasks)} 条 Query 快速获取相关信息",
                    {
                        "round": round_num,
                        "outer_loop_round": loop_round,
                        "task_count": len(next_tasks),
                        "queries": [task.query for task in next_tasks],
                        "feedback": feedback.message if feedback else "",
                    },
                )
            else:
                # 第二层：Search LLM 推理当前内容是否充分，并自主决定下一批 query 与返回数量。
                decision = await self._reason_search_decision(
                    request,
                    plan,
                    candidates,
                    seen_urls,
                    seen_queries,
                    round_num,
                    per_provider_count,
                    feedback=feedback,
                    loop_round=loop_round,
                    use_llm_reasoning=use_llm_reasoning,
                )
                per_provider_count = max(
                    1,
                    min(
                        int(decision.get("max_results_per_provider") or per_provider_count),
                        self.ctx.settings.cg_search_max_results_per_provider,
                    ),
                )
                if decision.get("should_stop"):
                    await self.record_llm_event(
                        "progress", "converged",
                        f"第 {round_num} 轮推理：{str(decision.get('stop_reason') or 'Search Agent 判断信息已满足要求')}",
                        {
                            "round": round_num,
                            "outer_loop_round": loop_round,
                            "coverage": estimate_coverage(
                                candidates,
                                [request.target_topic],
                                request.analysis_dimensions,
                            ),
                            "candidate_count": len(candidates),
                        },
                    )
                    break
                next_tasks = decision.get("tasks") or []

            if not next_tasks:
                coverage = estimate_coverage(
                    candidates,
                    [request.target_topic],
                    request.analysis_dimensions,
                )
                if coverage < self.ctx.settings.cg_min_coverage_to_stop:
                    next_tasks = self._fallback_gap_tasks(
                        request, plan, candidates, seen_urls, round_num
                    )
                if not next_tasks:
                    await self.record_llm_event(
                        "progress", "converged",
                        f"第 {round_num} 轮：没有发现新的高价值查询，停止搜索",
                        {"round": round_num, "total_candidates": len(candidates), "coverage": coverage},
                    )
                    break

            # ── Act 阶段：执行搜索（英文通用 + 中文知乎）──
            round_new = 0
            for task in next_tasks:
                if remaining_slots <= 0:
                    break
                if task.query.lower() in seen_queries and round_num > 1:
                    continue
                seen_queries.add(task.query.lower())
                results = await self.search_task(
                    search,
                    task,
                    per_provider_count,
                    f"outer_{loop_round}_search_{round_num}",
                    round_num,
                    loop_round,
                    request.target_topic,
                )

                # ── Observe 阶段：判断结果是否带来新信息 ──
                for c in results:
                    norm = normalize_url(c.url)
                    if norm and norm not in seen_urls:
                        seen_urls.add(norm)
                        candidates.append(c)
                        round_new += 1
                        remaining_slots -= 1
                        if remaining_slots <= 0:
                            break

            await self.record_llm_event(
                "progress", "round_complete",
                f"第 {round_num} 轮搜索完成，新增 {round_new} 个来源，累计 {len(candidates)} 个",
                {"round": round_num, "new_this_round": round_new, "total": len(candidates)},
            )

            # 收敛判断：连续 2 轮无新增 → 停止
            if round_new == 0:
                no_new_consecutive += 1
                if no_new_consecutive >= 2:
                    await self.record_llm_event(
                        "progress", "converged",
                        f"连续 2 轮无新来源，提前收敛（共 {round_num} 轮）",
                        {"round": round_num},
                    )
                    break
            else:
                no_new_consecutive = 0

            # 覆盖度充足提前停止
            coverage = estimate_coverage(
                candidates,
                [request.target_topic],
                request.analysis_dimensions,
            )
            dimensions = request.analysis_dimensions or DEFAULT_DIMENSIONS
            entities = [request.target_topic]
            covered_pairs = infer_candidate_pairs(candidates, plan.source_tasks, entities, dimensions)
            empty_pairs = [
                (entity, dimension)
                for entity in entities
                for dimension in dimensions
                if (entity, dimension) not in covered_pairs
            ]
            minimum_candidates = min(
                max(request.max_sources, len(entities) * len(dimensions)),
                len(entities) * len(dimensions) * request.max_sources_per_query,
            )
            if (
                coverage >= self.ctx.settings.cg_min_coverage_to_stop
                and not empty_pairs
                and len(candidates) >= minimum_candidates
            ):
                await self.record_llm_event(
                    "progress", "converged",
                    f"覆盖度 {coverage:.0%} 已达标，提前停止（{round_num} 轮）",
                    {"round": round_num, "coverage": coverage, "candidate_count": len(candidates)},
                )
                break

        return dedupe_source_candidates(candidates)

    async def _reason_search_decision(
        self,
        request: ResearchRequest,
        plan: ResearchPlan,
        current_candidates: list[SourceCandidate],
        seen_urls: set[str],
        seen_queries: set[str],
        round_num: int,
        current_max_results: int,
        *,
        feedback: ResearchFeedback | None,
        loop_round: int,
        use_llm_reasoning: bool = True,
    ) -> dict[str, Any]:
        if not self.llm_enabled or not use_llm_reasoning:
            return {
                "tasks": self._fallback_gap_tasks(request, plan, current_candidates, seen_urls, round_num),
                "max_results_per_provider": current_max_results,
                "should_stop": False,
            }

        entities = [request.target_topic]
        dimensions = request.analysis_dimensions or DEFAULT_DIMENSIONS
        pair_counts = candidate_pair_counts(current_candidates, plan.source_tasks, entities, dimensions)
        provider_counts = Counter(c.source_provider for c in current_candidates if c.source_provider)
        source_type_counts = Counter(c.source_type for c in current_candidates if c.source_type)
        coverage_lines = []
        for entity in entities:
            cell_parts = []
            for dim in dimensions:
                cell_parts.append(f"{DIMENSION_LABELS.get(dim, dim)}={pair_counts.get((entity, dim), 0)}")
            coverage_lines.append(f"- {entity}: " + "，".join(cell_parts))

        data = await self.invoke_json(
            system=(
                "你是 SourceResearchAgent 的 ReAct 推理模块。\n"
                "你要判断当前搜索内容是否足够支撑后续 Evidence 和 Analysis。\n\n"
                "【停止条件，必须同时满足】\n"
                "1. 每个研究方向×每个推理模式至少有 3 篇不同论文的候选内容\n"
                "2. 核心方向必须有代表性论文和最新进展\n"
                "3. 核心推理模式维度必须有多篇论文交叉验证\n"
                "4. Analysis Agent 反馈的缺口已经被补上\n"
                "5. 最近搜索仍有新增内容；若连续两轮没有新增，系统会自动停止\n\n"
                "【行动空间】\n"
                "- 若不满足，生成下一批 query，最多 8 条\n"
                "- query 使用英文技术检索语句，面向本地论文库 embedding 检索\n"
                "- 你可以调整 max_results_per_provider，默认 5；持续找不到好内容时提高到 8-12\n\n"
                "输出 JSON：{\n"
                '  "reasoning": "逐研究方向×推理模式说明哪里够、哪里不够",\n'
                '  "should_stop": false,\n'
                '  "stop_reason": "",\n'
                '  "max_results_per_provider": 5,\n'
                '  "next_tasks": [{"entity":"研究方向","dimension":"推理模式","intent":"official","query":"...","rationale":"..."}]\n'
                "}"
            ),
            user=(
                f"外层 Research Loop：第 {loop_round} 轮\n"
                f"Search 内部轮次：第 {round_num} 轮\n"
                f"研究方向：{request.target_topic}\n"
                f"维度：{', '.join(dimensions)}\n"
                f"Analysis 反馈：{feedback.message if feedback else '无'}\n"
                f"Feedback gaps：{[gap.model_dump(mode='json') for gap in (feedback.gaps if feedback else [])]}\n\n"
                "当前覆盖：\n" + "\n".join(coverage_lines) + "\n\n"
                f"总来源数：{len(current_candidates)}\n"
                f"provider 分布：{dict(provider_counts)}\n"
                f"source type 分布：{dict(source_type_counts)}\n"
                f"已搜 query（避免重复）：{list(seen_queries)[-40:]}\n"
                f"已见 URL（截断）：{list(seen_urls)[-40:]}\n"
                f"当前每引擎返回数：{current_max_results}"
            ),
        )
        if not data:
            return {
                "tasks": self._fallback_gap_tasks(request, plan, current_candidates, seen_urls, round_num),
                "max_results_per_provider": current_max_results,
                "should_stop": False,
            }

        reasoning = str(data.get("reasoning") or "")
        should_stop = bool(data.get("should_stop", False))
        await self.record_llm_event(
            "progress", "reasoning",
            f"第 {round_num} 轮推理：{reasoning[:220]}",
            {
                "round": round_num,
                "outer_loop_round": loop_round,
                "reasoning": reasoning,
                "should_stop": should_stop,
                "max_results_per_provider": data.get("max_results_per_provider") or current_max_results,
            },
        )

        tasks: list[SourceTask] = []
        allowed_intents = {"official", "docs", "review", "comparison", "news"}
        for i, row in enumerate((data.get("next_tasks") or [])[:8]):
            if not isinstance(row, dict):
                continue
            query = str(row.get("query") or "").strip()
            if len(query) < 4:
                continue
            intent = str(row.get("intent") or "official").lower()
            if intent not in allowed_intents:
                intent = "official"
            dimension = str(row.get("dimension") or "other")
            if dimension not in DIMENSION_LABELS:
                dimension = "other"
            tasks.append(SourceTask(
                task_id=f"react_r{round_num}_{i + 1:02d}",
                entity=str(row.get("entity") or request.target_topic).strip(),
                dimension=dimension,
                intent=intent,  # type: ignore[arg-type]
                query=query,
                expected_source_types=["academic_paper"],
                rationale=str(row.get("rationale") or ""),
            ))
        return {
            "tasks": tasks,
            "max_results_per_provider": data.get("max_results_per_provider") or current_max_results,
            "should_stop": should_stop,
            "stop_reason": data.get("stop_reason") or "",
        }

    async def _reason_next_tasks(
        self,
        request: ResearchRequest,
        plan: ResearchPlan,
        current_candidates: list[SourceCandidate],
        seen_urls: set[str],
        round_num: int,
        *,
        use_llm_reasoning: bool = True,
    ) -> list[SourceTask]:
        """ReAct Reason 步骤：LLM 分析当前已搜内容，决定下一批搜索任务。"""

        # 构造当前状态摘要给 LLM
        entity_url_counts = Counter[str]()
        dimension_url_counts = Counter[str]()
        for c in current_candidates:
            text = f"{c.url} {c.title} {c.snippet}".lower()
            for entity in [request.target_topic]:
                if entity.lower() in text:
                    entity_url_counts[entity] += 1
            for dim in request.analysis_dimensions:
                keywords = dimension_search_keywords(dim)
                if any(kw.lower() in text for kw in keywords):
                    dimension_url_counts[dim] += 1

        coverage_summary = {
            "每个研究方向的已有来源数": dict(entity_url_counts),
            "每个维度的已有来源数": {
                DIMENSION_LABELS.get(d, d): dimension_url_counts[d]
                for d in request.analysis_dimensions
            },
            "总来源数": len(current_candidates),
            "已知URL列表（用于去重）": list(seen_urls)[:40],
        }

        # 第一轮直接用 plan 里的任务，不调 LLM（节省 token）
        if round_num == 1 and plan.source_tasks:
            tasks = plan.source_tasks[:16]
            await self.record_llm_event(
                "progress", "tool_call",
                f"执行 {len(tasks)} 条 Query 快速获取相关信息",
                {"round": 1, "task_count": len(tasks)},
            )
            return tasks

        if not self.llm_enabled or not use_llm_reasoning:
            # 无 LLM：基于缺口规则生成
            return self._fallback_gap_tasks(request, plan, current_candidates, seen_urls, round_num)

        data = await self.invoke_json(
            system=(
                "你是论文检索智能体（SourceResearchAgent），负责自主决定检索策略。\n"
                "\n"
                "【当前任务】\n"
                "分析已检索到的论文覆盖情况，判断哪些推理模式证据还不够，决定下一步检索什么。\n"
                "\n"
                "【判断信息充足的标准】\n"
                "满足以下【所有】条件时才可以停止（should_stop=true）：\n"
                "  1. 每个推理模式维度都有至少 3 条不同论文的候选内容\n"
                "  2. 核心方向（如知识图谱、NLP）有代表性论文和最新进展\n"
                "  3. 总来源数超过 30 条\n"
                "只要有任何一个推理模式维度的来源数为 0，就不能停止。\n"
                "\n"
                "【下一批查询的设计原则】\n"
                "  A. 优先补充完全没有来源的推理模式维度\n"
                "  B. 其次补充只有经典论文、缺少最新进展的维度\n"
                "  C. 避免重复检索已覆盖充分的内容\n"
                "  D. 技术对比类查询举例：'RAG vs long-context LLM performance comparison'\n"
                "  E. 方法创新类查询举例：'knowledge graph reasoning with large language models'\n"
                "\n"
                "【输出格式】JSON：\n"
                "{\n"
                '  "reasoning": "逐推理模式分析：已有X条，还缺Y，原因Z",\n'
                '  "should_stop": false,\n'
                '  "stop_reason": "（should_stop=true 时填写）",\n'
                '  "next_tasks": [\n'
                '    {\n'
                '      "entity": "研究方向",\n'
                '      "dimension": "推理模式英文key",\n'
                '      "intent": "official",\n'
                '      "query": "具体英文检索查询",\n'
                '      "rationale": "这条查询能补充什么具体信息"\n'
                '    }\n'
                "  ]\n"
                "}\n"
                "\n"
                "next_tasks 每次最多 8 条，intent 只能是：official / docs / review / comparison"
            ),
            user=(
                f"研究方向：{request.target_topic}"
                + (f"（{request.topic_description}）" if request.topic_description else "")
                + f"\n分析维度：{[DIMENSION_LABELS.get(d, d) for d in request.analysis_dimensions]}\n"
                f"当前第 {round_num} 轮（共最多 {getattr(request, 'max_search_rounds', 3)} 轮）\n\n"
                f"当前搜索覆盖状态：\n"
                f"- 各维度已有来源数：{ {DIMENSION_LABELS.get(d, d): dimension_url_counts[d] for d in request.analysis_dimensions} }\n"
                f"- 总来源数：{len(current_candidates)}\n"
                f"- 已知URL（用于去重，勿重复）：{list(seen_urls)[:30]}"
            ),
        )

        if not data:
            return self._fallback_gap_tasks(request, plan, current_candidates, seen_urls, round_num)

        reasoning = str(data.get("reasoning") or "")
        should_stop = bool(data.get("should_stop", False))

        await self.record_llm_event(
            "progress", "reasoning",
            f"第 {round_num} 轮推理：{reasoning[:120]}",
            {"round": round_num, "reasoning": reasoning, "should_stop": should_stop},
        )

        if should_stop:
            return []

        raw_tasks = data.get("next_tasks") or []
        tasks: list[SourceTask] = []
        allowed_intents = {"official", "docs", "review", "comparison", "news"}
        for i, row in enumerate(raw_tasks[:8]):
            if not isinstance(row, dict):
                continue
            query = str(row.get("query") or "").strip()
            if len(query) < 4:
                continue
            intent = str(row.get("intent") or "official").lower()
            if intent not in allowed_intents:
                intent = "official"
            dimension = str(row.get("dimension") or "other")
            if dimension not in DIMENSION_LABELS:
                dimension = "other"
            use_zhihu = bool(row.get("use_zhihu", False))
            tasks.append(SourceTask(
                task_id=f"react_r{round_num}_{i + 1:02d}",
                entity=str(row.get("entity") or request.target_topic).strip(),
                dimension=dimension,
                intent=intent,  # type: ignore[arg-type]
                query=query,
                expected_source_types=["academic_paper"],
                rationale=str(row.get("rationale") or ""),
            ))
        return tasks

    def _fallback_gap_tasks(
        self,
        request: ResearchRequest,
        plan: ResearchPlan,
        candidates: list[SourceCandidate],
        seen_urls: set[str],
        round_num: int,
    ) -> list[SourceTask]:
        """无 LLM 时基于覆盖缺口规则生成补充任务。"""
        entities = [request.target_topic]
        dimensions = request.analysis_dimensions or DEFAULT_DIMENSIONS
        covered = infer_candidate_pairs(candidates, plan.source_tasks, entities, dimensions)
        tasks: list[SourceTask] = []
        used_queries: set[str] = {normalize_url(c.url) for c in candidates}
        used_queries.update(c.query.lower() for c in candidates if c.query)

        deduped = dedupe_source_candidates(candidates)
        domains = [source_domain(candidate.url) for candidate in deduped]
        domain_counts = Counter(domain for domain in domains if domain)
        dominant_share = max(domain_counts.values(), default=0) / max(1, len(deduped))
        if len(deduped) >= 3 and dominant_share > 0.6:
            for entity in entities:
                diversity_dimension = dimensions[0] if dimensions else "other"
                diversity_task = SourceTask(
                    task_id=f"gap_diversity_{round_num:02d}_{len(tasks) + 1:02d}",
                    entity=entity,
                    dimension=diversity_dimension,
                    intent="comparison",
                    query=f"{entity} survey benchmark recent advances",
                    expected_source_types=["academic_paper"],
                    rationale="Diversify paper candidates and add broad survey or benchmark coverage.",
                )
                if diversity_task.query.lower() not in used_queries:
                    tasks.append(diversity_task)
                    used_queries.add(diversity_task.query.lower())
                if len(tasks) >= 2:
                    break

        for entity in entities:
            for dim in dimensions:
                if (entity, dim) in covered:
                    continue
                task = make_gap_source_task(entity, dim, len(tasks) + 1)
                if task.query.lower() not in used_queries:
                    tasks.append(task)
                    used_queries.add(task.query.lower())
                if len(tasks) >= 6:
                    return tasks
        return tasks

    async def _search_zhihu_task(
        self,
        search: SearchTool,
        task: SourceTask,
        max_results: int,
        round_num: int,
    ) -> list[SourceCandidate]:
        """调用知乎搜索并记录 trace。"""
        await self.record_llm_event(
            "progress", "tool_call",
            f"知乎搜索：{task.query}",
            {
                "tool": "zhihu_search",
                "query": task.query,
                "entity": task.entity,
                "dimension": task.dimension,
                "round": round_num,
            },
        )
        try:
            results = await search.search_zhihu(task.query, max_results=max_results)
        except Exception as exc:
            await self.record_llm_event(
                "error", "failed",
                f"知乎搜索失败：{exc!s:.200}",
                {"tool": "zhihu_search", "query": task.query},
            )
            return []
        await self.record_llm_event(
            "progress", "tool_result",
            f"知乎搜索返回 {len(results)} 条结果",
            {"tool": "zhihu_search", "query": task.query, "count": len(results)},
        )
        for r in results:
            r.query = task.query
        return results

    async def search_task(
        self,
        search: Any,
        task: SourceTask,
        max_results: int,
        search_round: str,
        round_num: int = 1,
        loop_round: int = 1,
        target_topic: str = "",
    ) -> list[SourceCandidate]:
        await self.record_llm_event(
            "progress", "tool_call",
            f"内容搜索：{task.query}",
            {
                "tool": "content_search",
                "task_id": task.task_id,
                "entity": task.entity,
                "dimension": task.dimension,
                "intent": task.intent,
                "query": task.query,
                "expected_source_types": task.expected_source_types,
                "search_round": search_round,
                "round": round_num,
                "outer_loop_round": loop_round,
                "max_results_per_provider": max_results,
            },
        )
        try:
            if hasattr(search, "search_content_providers"):
                provider_batches = await search.search_content_providers(task.query, max_results_per_provider=max_results)
            elif hasattr(search, "search_for_topic"):
                topic_results = await search.search_for_topic(
                    task.query,
                    target_topic,
                    max_results=max_results,
                )
                provider_batches = {"local_papers": topic_results}
            else:
                legacy_results = await search.search(task.query, max_results=max_results)
                provider_batches = {"search": legacy_results}
        except Exception as exc:
            await self.record_llm_event(
                "error", "failed",
                f"搜索失败：{exc!s:.200}",
                {"tool": "search", "task_id": task.task_id, "query": task.query},
            )
            return []
        results: list[SourceCandidate] = []
        rejected_results: list[SourceCandidate] = []
        for provider, provider_results in provider_batches.items():
            for result in provider_results:
                result.query = task.query
                if result.source_type == "other" and task.expected_source_types:
                    result.source_type = task.expected_source_types[0]
                task_score = score_candidate_for_task(result, task, search_round)
                if result.relevance_label == "unscored":
                    result.score = task_score
                    result.relevance_score = max(result.relevance_score, task_score)
                else:
                    result.score = round(
                        min(1.0, max(result.score, result.relevance_score * 0.85 + task_score * 0.15)),
                        4,
                    )
                if should_reject_supplemental_application_source(
                    result,
                    target_topic=target_topic,
                    loop_round=loop_round,
                ):
                    result.relevance_label = "reject"
                    result.relevance_score = 0.0
                    result.score = 0.0
                    result.rejection_reason = SUPPLEMENTAL_APPLICATION_REJECTION_REASON
                if result.relevance_label == "reject":
                    rejected_results.append(result)
                    continue
                results.append(result)
            await self.record_llm_event(
                "progress", "tool_result",
                f"{provider_display_name(provider)} 返回 {len(provider_results)} 条结果，接受 {len([item for item in provider_results if item.relevance_label != 'reject'])} 条",
                {
                    "tool": provider,
                    "task_id": task.task_id,
                    "query": task.query,
                    "count": len(provider_results),
                    "accepted_count": len([item for item in provider_results if item.relevance_label != "reject"]),
                    "rejected_count": len([item for item in provider_results if item.relevance_label == "reject"]),
                    "search_round": search_round,
                    "results": [
                        {
                            "title": item.title,
                            "url": item.url,
                            "snippet": item.snippet,
                            "content_source": item.content_source,
                            "relevance_score": item.relevance_score,
                            "relevance_label": item.relevance_label,
                            "rejection_reason": item.rejection_reason,
                            "source_subtype": item.source_subtype,
                            "source_subtype_reason": item.source_subtype_reason,
                        }
                        for item in provider_results
                    ],
                },
            )
        if rejected_results:
            path = self.ctx.run_dir / "sources" / "rejected_sources.jsonl"
            for item in rejected_results:
                await append_jsonl(
                    path,
                    {
                        "url": item.url,
                        "title": item.title,
                        "query": item.query,
                        "task_id": task.task_id,
                        "dimension": task.dimension,
                        "source_provider": item.source_provider,
                        "score": item.score,
                        "embedding_score": item.embedding_score,
                        "lexical_score": item.lexical_score,
                        "reranker_score": item.reranker_score,
                        "relevance_score": item.relevance_score,
                        "relevance_label": item.relevance_label,
                        "rejection_reason": item.rejection_reason,
                        "source_subtype": item.source_subtype,
                        "source_subtype_reason": item.source_subtype_reason,
                    },
                )
            await self.record_llm_event(
                "progress",
                "sources_rejected",
                f"过滤 {len(rejected_results)} 条低相关来源，未进入 Evidence 抽取",
                {
                    "task_id": task.task_id,
                    "query": task.query,
                    "rejected_count": len(rejected_results),
                    "examples": [
                        {
                            "title": item.title,
                            "url": item.url,
                            "relevance_score": item.relevance_score,
                            "reason": item.rejection_reason,
                            "source_subtype": item.source_subtype,
                            "source_subtype_reason": item.source_subtype_reason,
                        }
                        for item in rejected_results[:8]
                    ],
                },
            )
        return results

    async def collect(
        self,
        candidates: list[SourceCandidate],
        fetcher,
        save_document: Callable[[SourceCandidate, Any], Awaitable[SourceDocument]],
        trace: Callable[[str, str, str, dict[str, Any] | None], Awaitable[None]],
    ) -> list[SourceDocument]:
        documents: list[SourceDocument] = []
        for candidate in candidates:
            page = await fetcher.fetch(candidate.url)
            document = await save_document(candidate, page)
            documents.append(document)
            await trace(
                "progress",
                "fetched" if document.ok else "failed",
                document.title,
                {"url": document.url, "ok": document.ok, "error": document.error},
            )
        return documents


class EvidenceStructuringAgent(BaseAgent):
    name = "EvidenceStructuringAgent"
    skill_ids = AGENT_SKILLS[name]
    skill_id = skill_ids[0]

    async def extract(
        self,
        request: ResearchRequest,
        document: SourceDocument,
        dimensions: list[str],
        entities: list[str],
        deterministic_items: list[Evidence],
        stable_id_fn,
        source_weight_fn,
    ) -> list[Evidence]:
        if not self.llm_enabled or not document.content:
            return deterministic_items

        allow_relaxed_quote_match = document.source_type in {"academic_paper", "local_paper"}
        system_prompt = (
            "你是论文创新证据提取智能体（EvidenceStructuringAgent），专门从学术论文中提取结构化创新证据。\n"
            "\n"
            "【当前来源类型】学术论文\n"
            "\n"
            "【提取规则】\n"
            "1. 只提取可验证的创新事实，禁止提取推测、空洞评价或无法核实的声明\n"
            "2. 每条证据必须包含：reasoning_pattern（推理模式）、bottleneck（瓶颈）、mechanism（机制）\n"
            "3. reasoning_pattern 必须是 15 种之一：gap_driven_reframing, cross_domain_synthesis, representation_shift,\n"
            "   modular_pipeline_composition, data_evaluation_engineering, principled_probabilistic_modeling,\n"
            "   formal_experimental_tightening, approximation_engineering, inference_time_control,\n"
            "   structural_inductive_bias, multiscale_hierarchical_modeling, mechanistic_decomposition,\n"
            "   adversary_modeling, numerics_systems_codesign, data_centric_optimization\n"
            "4. bottleneck：该论文要解决的具体技术瓶颈是什么\n"
            "5. mechanism：论文提出的具体解决机制是什么\n"
            "6. fact 字段：用中文写一句完整的创新事实，包含：论文名 + 具体创新内容\n"
            "7. quote 必须是论文原文中的连续片段，字数 20-300，不能改写\n"
            "8. confidence 按证据强度：有充分实验验证 → 0.80-0.92，初步探索 → 0.55-0.70\n"
            "9. 同一创新点只提取一次，不要重复\n"
            "\n"
            f"dimension 只能从以下值选择：{', '.join(dimensions)}\n"
            '输出 JSON：{"evidence":[{"paper": "论文名", "dimension": "维度", "reasoning_pattern": "推理模式", "bottleneck": "瓶颈", "mechanism": "机制", "fact": "创新事实", "quote": "原文片段", "confidence": 0.8}]}\n'
            "每篇论文最多提取 12 条最重要的创新证据。"
        )

        data = await self.invoke_json(
            system=system_prompt,
            user=(
                f"研究方向：{request.target_topic}\n"
                f"来源类型：{document.source_type}\n"
                f"网页标题：{document.title}\n"
                f"URL：{document.url}\n\n"
                f"正文内容：\n{document.content[:8000]}"
            ),
        )
        rows = data.get("evidence") if data else None
        if not isinstance(rows, list):
            return deterministic_items
        items: list[Evidence] = []
        content_lower = document.content.lower()
        max_items = 12
        for row in rows[:max_items]:
            if not isinstance(row, dict):
                continue
            dimension = str(row.get("dimension") or "other")
            if dimension not in dimensions:
                dimension = "other"
            quote = str(row.get("quote") or "").strip()
            fact = str(row.get("fact") or "").strip()
            if len(quote) < 15 or not fact:
                continue
            # 学术论文：LLM 可能从论文任意部分提取 quote，不要求严格匹配 content
            # 只要求 quote 包含一些 content 中的单词即可
            if allow_relaxed_quote_match and quote.lower() not in content_lower:
                # 放宽检查：只要 quote 中有 30% 以上的单词出现在 content 中就接受
                quote_words = set(quote.lower().split())
                content_words = set(content_lower.split())
                overlap = len(quote_words & content_words) / max(len(quote_words), 1)
                if overlap < 0.3:
                    continue
            elif not allow_relaxed_quote_match and quote.lower() not in content_lower:
                continue
            confidence = clamp_float(row.get("confidence"), 0.45, 0.95)
            evidence_id = stable_id_fn("ev", f"{self.ctx.run_id}:{document.url}:{dimension}:{quote}")
            paper = str(row.get("paper") or "").strip() or None
            freshness = compute_freshness_score(document.published_at, dimension)
            reasoning_pattern = str(row.get("reasoning_pattern") or "").strip()
            bottleneck = str(row.get("bottleneck") or "").strip()
            mechanism = str(row.get("mechanism") or "").strip()
            items.append(
                Evidence(
                    evidence_id=evidence_id,
                    run_id=self.ctx.run_id,
                    url=document.url,
                    title=document.title,
                    content=quote[:500],
                    fetched_at=document.fetched_at,
                    source_type=document.source_type,
                    dimension=dimension,
                    dimension_label=DIMENSION_LABELS.get(dimension, dimension),
                    paper=paper,
                    fact=fact[:500],
                    quote=quote[:500],
                    source_title=document.title,
                    source_url=document.url,
                    source_id=document.source_id,
                    source_subtype=document.source_subtype,
                    source_subtype_reason=document.source_subtype_reason,
                    confidence=confidence,
                    reasoning_pattern=reasoning_pattern,
                    bottleneck=bottleneck,
                    mechanism=mechanism,
                    authority_score=round(source_weight_fn(document.source_type) + 0.3, 3),
                    freshness_score=freshness,
                    relevance_score=confidence,
                    extracted_by_agent=self.name,
                    extracted_by_skill=self.skill_id,
                    published_at=document.published_at,
                )
            )
        return items or deterministic_items


class AnalysisAndReviewAgent(BaseAgent):
    name = "AnalysisAndReviewAgent"
    skill_ids = AGENT_SKILLS[name]
    skill_id = skill_ids[0]

    async def generate(self, evidence: list[Evidence], deterministic_claims: list[Claim], request: "ResearchRequest | None" = None) -> list[Claim]:
        if not self.llm_enabled or not evidence:
            return deterministic_claims

        # 从 evidence 或 request 里推断研究方向
        topic = request.target_topic if request else "研究方向"

        # 按维度聚合 Evidence，便于 LLM 做横向对比
        by_dim: dict[str, list[dict]] = {}
        for ev in evidence[:100]:
            dim = ev.dimension
            by_dim.setdefault(dim, []).append({
                "evidence_id": ev.evidence_id,
                "paper": ev.paper,
                "fact": ev.fact,
                "quote": ev.quote[:200],
                "source_type": ev.source_type,
                "source_subtype": ev.source_subtype,
                "source_url": ev.source_url,
                "confidence": ev.confidence,
            })

        # 构建 evidence 映射，用于后续检查
        evidence_by_id = {ev.evidence_id: ev for ev in evidence}
        evidence_ids = set(evidence_by_id.keys())

        data = await self.invoke_json(
            system=(
                "你是推理模式分析智能体（AnalysisAndReviewAgent）的结论生成技能。\n"
                "\n"
                "【核心原则】\n"
                "你的工作是从论文 Evidence 中提炼真正有价值的推理模式洞察，而不是简单复述证据。\n"
                "好的 Claim 要回答：'在这个推理模式上，这些论文之间的关键创新差异是什么？这对该领域意味着什么？'\n"
                "\n"
                "【结论生成要求 - 严格执行】\n"
                "1. 先生成 atomic observation，再生成 synthesis claim；不要直接从大证据池跳到领域趋势。\n"
                "2. 【推理模式对比型】结论：必须引用 >=2 篇不同论文，且来源角色/source_subtype 必须一致\n"
                "   ✓ '在因果评测协议上，论文A构造抽象变量基准，而论文B评估统计因果陷阱'\n"
                "   ✗ 如果只有1篇论文有证据，禁止生成对比型结论\n"
                "3. 【跨 source-role 对照型】结论：如果 supporting_evidence_ids 来自不同 source_subtype，禁止写领域趋势；只能显式写成 benchmark/core/testbed 等角色差异对照\n"
                "   ✓ '作为跨 source-role 对照，benchmark 论文侧重评测协议，core method 论文侧重机制设计'\n"
                "   ✗ '该领域整体形成某种趋势'（不同 source role 混合时禁止）\n"
                "4. 【单论文洞察型】结论：必须引用 >=2 条来自同一论文的证据，并显式写成'作为单论文观察'\n"
                "   ✓ '论文C的表征转换创新在于将图结构转化为序列表示，有2条证据支持'\n"
                "   ✗ 如果只有1条证据，只能生成描述性结论，措辞必须保守\n"
                "5. 禁止生成【无实质内容的废话型】结论：\n"
                "   '各论文均在积极探索'（×）'方法较为新颖'（×）\n"
                "6. 每条 Claim 必须绑定 supporting_evidence_ids，且 evidence_id 必须真实存在\n"
                "7. 关注推理模式的分布：哪些模式被大量论文使用（热点），哪些模式被忽视（研究空白）\n"
                "8. confidence 反映证据充分程度：\n"
                "   - 多篇论文交叉验证 → 0.75-0.85\n"
                "   - 单篇论文多条证据 → 0.60-0.75\n"
                "   - 仅1条证据 → 0.45-0.55（措辞必须非常保守）\n"
                "\n"
                "【证据充分性检查 - 必须遵守】\n"
                "- 对比型结论：supporting_evidence_ids 必须来自 >=2 篇不同论文\n"
                "- 普通 comparative 结论：supporting_evidence_ids 必须来自同一 source_subtype\n"
                "- 不同 source_subtype 的结论：只能写成 cross_role_contrast，不得写成领域整体趋势\n"
                "- 单论文结论：supporting_evidence_ids 必须有 >=2 条证据\n"
                "- 如果证据不足，降低 confidence 或跳过该结论\n"
                "\n"
                "【输出格式】JSON：\n"
                '{"claims":[{\n'
                '  "dimension": "推理模式英文key",\n'
                '  "claim_type": "comparative|cross_role_contrast|single_paper|descriptive",\n'
                '  "claim": "完整的分析结论（中文，50-200字）",\n'
                '  "supporting_evidence_ids": ["ev_xxx", "ev_yyy"],\n'
                '  "confidence": 0.75,\n'
                '  "reasoning_summary": "基于哪些证据得出此结论，推理链条是什么（2-3句）"\n'
                "}]}"
            ),
            user=(
                f"研究方向：{topic}\n\n"
                "按维度聚合的 Evidence（请逐维度分析，生成推理模式结论）：\n"
                + "\n\n".join(
                    f"=== {DIMENSION_LABELS.get(dim, dim)} 维度（{len(items)} 条证据，来自 {len(set(ev['paper'] for ev in items if ev['paper']))} 篇论文）===\n"
                    + "\n".join(
                        f"[{ev['evidence_id']}] {ev['paper'] or '?'} | {ev['source_type']} | {ev.get('source_subtype') or 'unclassified'} | "
                        f"置信度{ev['confidence']:.2f}\n事实：{ev['fact']}\n摘要：{ev['quote'][:120]}"
                        for ev in items[:8]
                    )
                    for dim, items in by_dim.items()
                )
            ),
        )
        rows = data.get("claims") if data else None
        if not isinstance(rows, list):
            return deterministic_claims
        claims: list[Claim] = []
        for row in rows[:24]:
            if not isinstance(row, dict):
                continue
            supporting = [ev_id for ev_id in coerce_str_list(row.get("supporting_evidence_ids")) if ev_id in evidence_ids]
            if not supporting:
                continue

            # 推断 claim 类型（不依赖 LLM 输出）
            supporting_papers = set()
            supporting_source_subtypes = []
            supporting_source_subtype_papers: dict[str, set[str]] = {}
            supporting_dimensions = []
            for ev_id in supporting:
                ev = evidence_by_id.get(ev_id)
                if not ev:
                    continue
                paper_key = ev.paper or ev.source_title or ev.source_url or ev.url or "unknown"
                if paper_key:
                    supporting_papers.add(paper_key)
                if ev.source_subtype:
                    supporting_source_subtypes.append(ev.source_subtype)
                    supporting_source_subtype_papers.setdefault(ev.source_subtype, set()).add(paper_key)
                if ev.dimension:
                    supporting_dimensions.append(ev.dimension)
            source_subtype_counts = dict(sorted(Counter(supporting_source_subtypes).items()))
            source_subtype_paper_counts = {
                subtype: len(papers)
                for subtype, papers in sorted(supporting_source_subtype_papers.items())
            }

            # 自动推断 claim_type
            if len(supporting_papers) >= 2:
                claim_type = "comparative"  # 多篇论文 → 对比型
            elif len(supporting) >= 2:
                claim_type = "single_paper_observation"  # 同一篇论文多条证据 → 单论文观察
            else:
                claim_type = "backlog"  # 仅1条证据 → 进入补证队列

            claim_text = str(row.get("claim") or "").strip()
            if len(claim_text) < 12:
                continue
            reasoning_summary = str(row.get("reasoning_summary") or "由 LLM 基于 Evidence 聚合生成。")
            dimension = normalize_claim_dimension(
                row.get("dimension"),
                f"{claim_text} {reasoning_summary}",
                supporting_dimensions,
                set(by_dim),
            )
            if not dimension:
                continue
            confidence = clamp_float(row.get("confidence"), 0.45, 0.95)
            if claim_type == "backlog":
                confidence = min(confidence, 0.55)
                if not re.match(r"^(根据|基于|现有|从|该证据)", claim_text):
                    claim_text = f"现有单篇论文证据显示，{claim_text}"
            elif claim_type == "single_paper_observation":
                confidence = min(confidence, 0.75)
            support_level = (
                "strong" if len(supporting_papers) >= 2 and len(supporting) >= 2
                else "single_paper" if len(supporting) >= 2
                else "insufficient"
            )
            claim_id = make_stable_id("claim", f"{self.ctx.run_id}:{dimension}:{claim_text}:{supporting}")
            claims.append(
                Claim(
                    claim_id=claim_id,
                    run_id=self.ctx.run_id,
                    dimension=dimension,
                    dimension_label=DIMENSION_LABELS.get(dimension, dimension),
                    claim=claim_text,
                    supporting_evidence_ids=supporting,
                    confidence=confidence,
                    risk_level="medium",
                    reasoning_summary=reasoning_summary,
                    claim_type=claim_type,
                    source_paper_count=len(supporting_papers),
                    evidence_support_level=support_level,
                    supporting_source_subtypes=sorted(source_subtype_counts),
                    supporting_source_subtype_counts=source_subtype_counts,
                    supporting_source_subtype_paper_counts=source_subtype_paper_counts,
                    backlog_reason="single_evidence_claim" if claim_type == "backlog" else "",
                    verification_status="draft",
                    generated_by_agent=self.name,
                    generated_by_skill=self.skill_id,
                )
            )
        final_claims = merge_claims(deterministic_claims, claims)
        await self.record_llm_event(
            "progress",
            "claims_generated",
            f"基于 {len(evidence)} 条 Evidence 生成 {len(final_claims)} 条分析 Claim",
            {
                "evidence_count": len(evidence),
                "claim_count": len(final_claims),
                "llm_claim_count": len(claims),
                "deterministic_claim_count": len(deterministic_claims),
            },
        )
        return final_claims

    async def review(self, claims: list[Claim], evidence: list[Evidence]) -> list[Claim]:
        reviewed = deterministic_red_team(claims, evidence)
        if not self.llm_enabled or not reviewed:
            return reviewed

        by_id = {ev.evidence_id: ev for ev in evidence}
        system_prompt = (
            "你是推理模式审查智能体（AnalysisAndReviewAgent）的红队技能，\n"
            "专门从反方视角审查论文推理模式分析结论的可信度和风险。\n"
            "\n"
            "【审查维度】（每条 Claim 逐一检查）\n"
            "1. 【来源单一风险】：跨论文 synthesis 少于 2 篇独立论文 → severity=high；"
            "single_paper_observation 如果明确限定为单论文观察，不因来源单一被挑战\n"
            "2. 【证据不足风险】：仅 1 条证据支撑强结论 → severity=medium/high\n"
            "3. 【过度推断风险】：结论超出证据直接支持的范围 → severity=medium\n"
            "4. 【时效风险】：证据可能过时（快速演进方向中，较旧论文需标注时间边界）→ severity=low\n"
            "5. 【措辞风险】：使用了'最好'、'唯一'、'绝对'等绝对化表述而证据不支撑 → severity=medium\n"
            "6. 【实验验证缺失】：理论分析充分但缺乏实验结果佐证 → severity=low\n"
            "7. 【source role 混杂】：把 benchmark、应用案例、通用图学习、核心方法混成同一结论 → severity=medium/high\n"
            "\n"
            "【学术证据边界】\n"
            "- 学术 survey 级 observation 可以由独立论文证据支撑；不要默认要求 code repo、社区复现或 leaderboard。\n"
            "- 只有当 claim 声称可复现性、工程采用、社区影响或 SOTA 排名时，才要求代码/leaderboard/社区证据。\n"
            "- single_paper_observation 可以 verified，但 final_wording 必须保留'作为单论文观察'或等价限定。\n"
            "\n"
            "【验证状态标准】\n"
            "- verified：多源交叉验证，结论措辞保守，无明显风险\n"
            "- needs_evidence：证据不足，结论需要更多来源支撑\n"
            "- challenged：有明显风险，结论需要大幅修改措辞\n"
            "- rejected：结论无法被证据支撑，或存在明显事实错误\n"
            "\n"
            "【final_wording 要求】\n"
            "对 challenged/needs_evidence 的结论，必须给出修改后的保守表述：\n"
            "  - 加限定语：'根据现有论文...'、'据目前可见的证据...'、'部分实验结果显示...'\n"
            "  - 降低确定性：把'是'改成'似乎是'，把'最好'改成'表现较好'\n"
            "\n"
            "【输出格式】JSON：\n"
            '{"reviews":[{\n'
            '  "claim_id": "...",\n'
            '  "verification_status": "verified|needs_evidence|challenged|rejected",\n'
            '  "final_wording": "（修改后的措辞，verified 时可留空）",\n'
            '  "notes": [{"risk_type": "...", "comment": "...", "suggested_action": "...", "severity": "low|medium|high"}]\n'
            "}]}"
        )

        # 分批处理，每批 15 个 claims，避免 payload 过大导致 TTFT 过长
        BATCH_SIZE = 15
        for batch_start in range(0, len(reviewed), BATCH_SIZE):
            batch = reviewed[batch_start:batch_start + BATCH_SIZE]
            payload = []
            for claim in batch:
                supporting_evs = [by_id[eid] for eid in claim.supporting_evidence_ids if eid in by_id]
                source_types = list({ev.source_type for ev in supporting_evs})
                source_count = len({ev.source_url for ev in supporting_evs})
                payload.append({
                    "claim_id": claim.claim_id,
                    "claim": claim.claim,
                    "dimension": claim.dimension,
                    "claim_type": claim.claim_type,
                    "source_paper_count": claim.source_paper_count,
                    "supporting_evidence_ids": claim.supporting_evidence_ids[:5],  # 限制数量
                    "confidence": claim.confidence,
                    "evidence_source_types": source_types,
                    "evidence_source_papers": list({ev.paper or ev.source_title or ev.source_url for ev in supporting_evs})[:5],
                    "evidence_source_subtypes": list({ev.source_subtype for ev in supporting_evs if ev.source_subtype})[:5],
                    "unique_source_count": source_count,
                    "current_notes": [note.model_dump(mode="json") for note in claim.red_team_notes[:2]],
                })

            data = await self.invoke_json(
                system=system_prompt,
                user=f"待审查 Claims（含证据元信息）：\n{payload}",
            )
            rows = data.get("reviews") if data else None
            if not isinstance(rows, list):
                continue
            by_claim_id = {claim.claim_id: claim for claim in batch}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                claim = by_claim_id.get(str(row.get("claim_id") or ""))
                if not claim:
                    continue
                notes: list[RedTeamNote] = []
                for note in row.get("notes") or []:
                    if isinstance(note, dict):
                        notes.append(
                            RedTeamNote(
                                risk_type=str(note.get("risk_type") or "llm_review"),
                                comment=str(note.get("comment") or ""),
                                suggested_action=str(note.get("suggested_action") or ""),
                                severity=str(note.get("severity") or "medium"),  # type: ignore[arg-type]
                            )
                        )
                if notes:
                    claim.red_team_notes = notes
                final_wording = str(row.get("final_wording") or "").strip()
                if final_wording:
                    claim.final_wording = final_wording
                status = str(row.get("verification_status") or claim.verification_status)
                if claim.verification_status == "verified" and status != "verified" and not notes:
                    status = "verified"
                if status in {"verified", "needs_evidence", "challenged", "rejected", "draft", "included_in_report"}:
                    claim.verification_status = status  # type: ignore[assignment]
                if claim.verification_status == "verified":
                    claim.red_team_notes = []
                    claim.risk_level = "low"
                    claim.final_wording = claim.final_wording or claim.claim
                    continue
                claim.risk_level = "high" if any(note.severity == "high" for note in claim.red_team_notes) else (
                    "medium" if claim.red_team_notes else "low"
                )

        await self.record_llm_event(
            "progress",
            "claims_reviewed",
            f"完成 {len(reviewed)} 条 Claim 的 Red Team 审查",
            {
                "claim_count": len(reviewed),
                "verified": len([claim for claim in reviewed if claim.verification_status == "verified"]),
                "challenged": len([claim for claim in reviewed if claim.verification_status != "verified"]),
            },
        )
        return reviewed

    async def synthesize(
        self,
        request: ResearchRequest,
        plan: ResearchPlan,
        evidence: list[Evidence],
        claims: list[Claim],
        metrics: RunMetrics,
        started_at: datetime,
        *,
        matrix_builder: Callable[[list[Evidence], list[str], list[str]], Any],
        recommendations_builder: Callable[[str, ResearchRequest, list[Claim], list[Evidence], Any], list[Any]],
        graph_builder: Callable[[list[Evidence], list[Claim]], Any],
        observability_builder: Callable[..., Any],
        average_fn: Callable[[list[float]], float],
        unique_strings_fn: Callable[[list[str | None]], list[str]],
    ) -> dict[str, Any]:
        dimensions = plan.dimensions or request.analysis_dimensions or DEFAULT_DIMENSIONS
        paper_counts = Counter(ev.paper for ev in evidence if ev.paper)
        papers = unique_strings_fn(
            [
                *request.seed_papers,
                *[paper for paper, _ in paper_counts.most_common()],
            ]
        )[:12]
        if not papers:
            papers = [request.target_topic]
        matrix = matrix_builder(evidence, papers, dimensions)
        recommendations = recommendations_builder(self.ctx.run_id, request, claims, evidence, matrix)
        evidence_graph = graph_builder(evidence, claims)
        observability = observability_builder(
            started_at=started_at,
            plan=plan,
            metrics=metrics,
            evidence=evidence,
            claims=claims,
            matrix=matrix,
            recommendations=recommendations,
        )
        return {
            "matrix": matrix,
            "recommendations": recommendations,
            "evidence_graph": evidence_graph,
            "observability": observability,
            "average_evidence_confidence": round(average_fn([ev.confidence for ev in evidence]), 3),
        }

    async def assess_gaps(
        self,
        request: ResearchRequest,
        evidence: list[Evidence],
        claims: list[Claim],
        loop_round: int,
        coverage_score: float,
    ) -> ResearchFeedback:
        """评估当前证据缺口，决定是否需要继续搜索以及补充什么。"""
        dimensions = request.analysis_dimensions or DEFAULT_DIMENSIONS
        evidence_by_id = {ev.evidence_id: ev for ev in evidence}
        evidence_count_by_dim: Counter[str] = Counter()
        papers_by_dim: dict[str, set[str]] = {dim: set() for dim in dimensions}
        verified_claims_by_dim: Counter[str] = Counter()
        challenged_claims_by_dim: Counter[str] = Counter()

        for ev in evidence:
            if ev.dimension not in dimensions:
                continue
            evidence_count_by_dim[ev.dimension] += 1
            paper = ev.paper or ev.source_title or ev.source_url or ev.url
            if paper:
                papers_by_dim.setdefault(ev.dimension, set()).add(paper)

        for claim in claims:
            dim = claim.dimension if claim.dimension in dimensions else "other"
            if dim not in dimensions:
                continue
            supporting = [evidence_by_id[ev_id] for ev_id in claim.supporting_evidence_ids if ev_id in evidence_by_id]
            if not supporting:
                continue
            if claim.verification_status == "verified" and claim.risk_level != "high":
                verified_claims_by_dim[dim] += 1
            else:
                challenged_claims_by_dim[dim] += 1

        empty_cells = [
            (request.target_topic, dim)
            for dim in dimensions
            if evidence_count_by_dim.get(dim, 0) == 0
        ]
        weak_cells = [
            (request.target_topic, dim)
            for dim in dimensions
            if evidence_count_by_dim.get(dim, 0) > 0
            and (len(papers_by_dim.get(dim, set())) < 2 or verified_claims_by_dim.get(dim, 0) == 0)
        ]

        high_priority_weak_cells = [
            (entity, dim)
            for entity, dim in weak_cells
            if verified_claims_by_dim.get(dim, 0) == 0
        ]

        # 覆盖度足够且没有关键空白/弱覆盖时才停止，避免 50% 这种刚过线的状态过早收敛。
        if (
            coverage_score >= self.ctx.settings.cg_min_coverage_to_stop
            and not empty_cells
            and not high_priority_weak_cells
        ):
            return ResearchFeedback(
                needs_more_research=False,
                gaps=[],
                loop_round=loop_round,
                coverage_score=coverage_score,
                message=f"覆盖度 {coverage_score:.0%} 已达标，无需补充搜索。",
            )

        if not self.llm_enabled:
            # 直接用规则生成 gap
            lower_priority_weak_cells = [cell for cell in weak_cells if cell not in high_priority_weak_cells]
            gap_cells = empty_cells[:6]
            remaining_slots = max(0, 6 - len(gap_cells))
            gap_cells = [*gap_cells, *high_priority_weak_cells[:remaining_slots]]
            remaining_slots = max(0, 6 - len(gap_cells))
            gap_cells = [*gap_cells, *lower_priority_weak_cells[:remaining_slots]]
            gaps = [
                ResearchGap(
                    dimension=dim,
                    paper=entity,
                    reason=(
                        "本轮未采集到任何证据"
                        if (entity, dim) in empty_cells
                        else "当前 evidence 尚未形成足够的独立论文支撑或 Red Team verified claim"
                    ),
                    priority="high" if (entity, dim) in high_priority_weak_cells or (entity, dim) in empty_cells else "medium",
                    suggested_queries=[
                        f"{entity} {DIMENSION_LABELS.get(dim, dim)} 评测",
                        f"{entity} {dim} official",
                    ],
                )
                for entity, dim in gap_cells
            ]
            return ResearchFeedback(
                needs_more_research=bool(gaps),
                gaps=gaps,
                loop_round=loop_round,
                coverage_score=coverage_score,
                message=f"发现 {len(empty_cells)} 个空白格，{len(weak_cells)} 个弱覆盖格，建议补充搜索。",
            )

        # 用 LLM 生成更智能的补充建议
        payload = {
            "dimension_stats": {
                dim: {
                    "label": DIMENSION_LABELS.get(dim, dim),
                    "evidence_count": evidence_count_by_dim.get(dim, 0),
                    "independent_paper_count": len(papers_by_dim.get(dim, set())),
                    "verified_claim_count": verified_claims_by_dim.get(dim, 0),
                    "challenged_claim_count": challenged_claims_by_dim.get(dim, 0),
                }
                for dim in dimensions
            },
            "empty_cells": [{"entity": e, "dimension": DIMENSION_LABELS.get(d, d)} for e, d in empty_cells[:12]],
            "weak_cells": [{"entity": e, "dimension": DIMENSION_LABELS.get(d, d)} for e, d in weak_cells[:8]],
            "coverage_score": f"{coverage_score:.0%}",
            "loop_round": loop_round,
        }
        data = await self.invoke_json(
            system=(
                "你是推理模式分析审查智能体（AnalysisAndReviewAgent）的证据缺口评估器。\n"
                "\n"
                "【任务】\n"
                "基于当前证据覆盖矩阵，判断是否需要启动下一轮搜索，以及优先补充哪些信息缺口。\n"
                "\n"
                "【停止搜索的硬条件（设 needs_more_research=false）】\n"
                "只有同时满足以下条件时才停止：\n"
                f"  1. 覆盖度 ≥ {self.ctx.settings.cg_min_coverage_to_stop:.0%}\n"
                "  2. 没有关键维度的空白格/弱覆盖格\n"
                "  3. 核心研究方向已有多个推理模式的 Red Team verified claim\n"
                "  4. 若仍存在高优先级缺口，即使已到第 2 轮也继续建议补充搜索\n"
                "\n"
                "【缺口优先级判断】\n"
                "  high（必须补）：核心推理模式在任意维度完全没有证据\n"
                "  high（必须补）：已有 evidence 但没有 Red Team verified claim，无法进入主报告\n"
                "  high（必须补）：关键维度只覆盖 1 篇论文，无法交叉验证\n"
                "  medium（建议补）：次要维度没有证据或证据较弱\n"
                "  low（可选补）：非核心维度证据较弱\n"
                "\n"
                "【suggested_queries 要求】\n"
                "  - 必须具体，包含研究方向 + 推理模式关键词\n"
                "  - 查询写英文技术检索语句，适配本地论文库 embedding 检索\n"
                "\n"
                "【输出格式】JSON：\n"
                "{\n"
                '  "needs_more_research": true/false,\n'
                '  "reason": "判断原因（2-3句，说明主要缺口是什么）",\n'
                '  "gaps": [\n'
                '    {\n'
                '      "paper": "论文名",\n'
                '      "dimension": "维度英文key",\n'
                '      "reason": "为什么这个信息现在缺失，对分析有什么影响",\n'
                '      "priority": "high/medium/low",\n'
                '      "suggested_queries": ["具体查询1", "具体查询2"]\n'
                '    }\n'
                "  ]\n"
                "}"
            ),
            user=(
                f"研究方向：{request.target_topic}\n"
                f"当前轮次：第 {loop_round} 轮  当前覆盖度：{coverage_score:.0%}\n\n"
                f"按推理模式聚合的证据与 verified claim 覆盖：\n"
                + "\n".join(
                    f"  {DIMENSION_LABELS.get(d, d)}: "
                    f"{evidence_count_by_dim.get(d, 0)} 条 evidence，"
                    f"{len(papers_by_dim.get(d, set()))} 篇独立论文，"
                    f"{verified_claims_by_dim.get(d, 0)} 条 verified claim，"
                    f"{challenged_claims_by_dim.get(d, 0)} 条 challenged/backlog claim"
                    for d in dimensions
                )
                + f"\n\n空白格（0条证据）：{len(empty_cells)} 个\n"
                + "\n".join(f"  - {e} × {DIMENSION_LABELS.get(d, d)}" for e, d in empty_cells[:10])
                + f"\n\n弱覆盖格（无 verified claim 或独立论文不足）：{len(weak_cells)} 个"
            ),
        )

        if not data:
            lower_priority_weak_cells = [cell for cell in weak_cells if cell not in high_priority_weak_cells]
            gap_cells = empty_cells[:4]
            remaining_slots = max(0, 4 - len(gap_cells))
            gap_cells = [*gap_cells, *high_priority_weak_cells[:remaining_slots]]
            remaining_slots = max(0, 4 - len(gap_cells))
            gap_cells = [*gap_cells, *lower_priority_weak_cells[:remaining_slots]]
            gaps = [
                ResearchGap(
                    dimension=dim,
                    paper=entity,
                    reason=(
                        "证据为空"
                        if (entity, dim) in empty_cells
                        else "当前维度缺少足够的独立论文支撑或 Red Team verified claim"
                    ),
                    priority="high" if (entity, dim) in high_priority_weak_cells or (entity, dim) in empty_cells else "medium",
                    suggested_queries=[f"{entity} {dim}"],
                )
                for entity, dim in gap_cells
            ]
            return ResearchFeedback(
                needs_more_research=bool(gaps),
                gaps=gaps,
                loop_round=loop_round,
                coverage_score=coverage_score,
            )

        needs_more = bool(data.get("needs_more_research", False))
        raw_gaps = data.get("gaps") or []
        gaps: list[ResearchGap] = []
        for row in raw_gaps[:8]:
            if not isinstance(row, dict):
                continue
            suggested_queries = coerce_str_list(row.get("suggested_queries"))[:4]
            dim = normalize_dimension_key(
                row.get("dimension"),
                " ".join([str(row.get("reason") or ""), *suggested_queries]),
            )
            priority = str(row.get("priority") or "medium")
            if priority not in {"high", "medium", "low"}:
                priority = "medium"
            gaps.append(ResearchGap(
                dimension=dim,
                paper=str(row.get("paper") or request.target_topic),
                reason=str(row.get("reason") or ""),
                priority=priority,  # type: ignore[arg-type]
                suggested_queries=suggested_queries,
            ))

        await self.record_llm_event(
            "progress", "gap_assessment",
            f"缺口评估：需要补充={needs_more}，发现 {len(gaps)} 个缺口",
            {"needs_more": needs_more, "gap_count": len(gaps), "coverage_score": coverage_score},
        )

        return ResearchFeedback(
            needs_more_research=needs_more,
            gaps=gaps,
            loop_round=loop_round,
            coverage_score=coverage_score,
            message=str(data.get("reason") or ""),
        )


class ReportComposerAgent(BaseAgent):
    name = "ReportComposerAgent"
    skill_ids = AGENT_SKILLS[name]
    skill_id = skill_ids[0]

    async def write(
        self,
        request: ResearchRequest,
        evidence: list[Evidence],
        claims: list[Claim],
        metrics: RunMetrics,
        artifacts: dict[str, Any] | None = None,
    ) -> str:
        if not self.llm_enabled:
            raise RuntimeError("LLM is not configured; report generation cannot continue")

        artifacts = artifacts or {}
        citation_map, references = self._build_citation_index(
            evidence,
            claims,
            artifacts.get("recommendations") or [],
            max_references=150,
        )
        brief = self._build_report_brief(request, evidence, claims, metrics, artifacts, citation_map)
        ref_text = "\n".join(references)

        markdown = await self.invoke_text_strict(
            system=(
                "你是一名专业的学术研究分析师，正在撰写一份供研究者阅读的论文推理模式分析报告。\n"
                "你只能基于用户提供的 briefing notes 写作，不得编造 briefing notes 之外的事实、数据或来源。\n"
                "\n"
                "【硬性禁止】报告正文中不得出现以下任何内容：\n"
                "- '维度覆盖度'、'覆盖度约X%'、'该维度共关联N条Evidence'\n"
                "- 'ev_'开头的evidence_id\n"
                "- '置信度0.XX'、'strong·0.76'等置信度分数\n"
                "- '该来源 在...相关公开资料中出现了可核验表述'这类系统内部描述\n"
                "- '相关证据集中在X，说明该方向值得持续关注'这类无信息量套话\n"
                "- 任何'X在Y维度有Z条证据'的表述\n"
                "\n"
                "【报告质量要求】\n"
                "1. 若 briefing notes 中 quality_context.report_mode 为 exploratory_pilot，标题或摘要第一句必须明确标注"
                "这是探索性 pilot / 低置信报告；不得把待验证观察写成正式结论\n"
                "2. 摘要中的强结论只能来自 top_claims；若 top_claims 为空，必须写明本轮未形成足够稳健的主结论\n"
                "3. tentative_claims 是补证 backlog，只能放在'下一轮补证'或'置信度说明'语境，不得作为报告主体分析材料\n"
                "4. 被 rejected 或 high risk 的 claim 不得作为结论或建议依据\n"
                "5. 摘要要有真正的学术判断：哪些推理模式最突出、哪些论文机制最有启发、研究空白在哪里\n"
                "6. 推理模式对比矩阵中每个格子只写简洁的文字判断（1句话），不写分数或证据条数\n"
                "7. 每个模式分析要写对比性段落：说清楚论文间的瓶颈、机制、实验支撑和适用边界差异\n"
                "8. 给出具体的、有据可查的结论——不要写'各有优劣'、'建议持续关注'这类无信息量的话\n"
                "9. 研究建议要给出能直接指导后续研究的差异化方向，针对具体场景，避免模板化\n"
                "10. 研究建议要可操作，结合推理模式分布说明为什么这么建议\n"
                "11. 每个关键判断后用方括号标注来源编号，如[1][3]，不要暴露内部ID\n"
                "12. 证据条目后附有（YYYY-MM）格式的发布月份标注；时效性敏感维度（资源需求、方法更新、研究动态）"
                "优先引用较新的证据，若只有旧证据可用，应在报告中注明'截至YYYY年MM月'并提示信息可能已更新\n"
                "13. 不要机械复述 briefing notes；要把证据转化成有判断、有取舍的学术分析。\n"
                "14. 如果 briefing notes 对某个判断支持不足，请直接写'现有公开证据不足以判断'，不要补充想象。\n"
                "\n"
                "【报告结构】（Markdown格式）\n"
                "# [报告标题]\n"
                "## 执行摘要\n"
                "3-5条核心结论，每条直接给出判断\n"
                "## 推理模式格局概览\n"
                "简短段落描述主要论文、机制簇和研究脉络\n"
                "## 推理模式能力对比矩阵\n"
                "Markdown表格，每格写简洁文字判断\n"
                "## 各推理模式深度分析\n"
                "每个推理模式一节，写比较性分析段落\n"
                "## 代表性论文：机制与启发\n"
                "## 研究空白与风险\n"
                "## 后续研究路线\n"
                "## 研究建议\n"
                "## 参考来源\n"
                "在此处输出占位符 <<REFERENCES>>，系统会自动替换为完整的参考来源列表\n"
                "\n"
                "只输出Markdown正文，不要用代码块包裹。"
            ),
            user=json.dumps(brief, ensure_ascii=False, default=str),
        )
        markdown = (markdown or "").strip()
        if markdown.startswith("```"):
            markdown = markdown.strip("`").removeprefix("markdown").strip()
        if len(markdown) < 800:
            raise ValueError("LLM report output was too short; report generation failed")
        ref_text = self._render_cited_references(markdown, references)
        if "<<REFERENCES>>" in markdown:
            markdown = markdown.replace("<<REFERENCES>>", ref_text)
        elif "## 参考来源" not in markdown:
            markdown = markdown + "\n\n## 参考来源\n\n" + ref_text
        return markdown

    def _render_cited_references(self, markdown: str, references: list[str]) -> str:
        cited_numbers = {
            int(match)
            for match in re.findall(r"\[(\d+)\]", markdown)
            if match.isdigit()
        }
        if not cited_numbers:
            return "\n".join(references)
        return "\n".join(
            reference
            for index, reference in enumerate(references, start=1)
            if index in cited_numbers
        )

    def _build_citation_index(
        self,
        evidence: list[Evidence],
        claims: list[Claim],
        recommendations: list[OpportunityRecommendation],
        *,
        max_references: int,
    ) -> tuple[dict[str, int], list[str]]:
        evidence_by_id = {item.evidence_id: item for item in evidence}
        ordered_ids: list[str] = []

        def add_id(evidence_id: str) -> None:
            if evidence_id in evidence_by_id and evidence_id not in ordered_ids:
                ordered_ids.append(evidence_id)

        for claim in sorted(claims, key=lambda item: item.confidence, reverse=True)[:30]:
            for evidence_id in claim.supporting_evidence_ids[:3]:
                add_id(evidence_id)
        for recommendation in recommendations[:8]:
            for evidence_id in recommendation.evidence_ids[:3]:
                add_id(evidence_id)
        for item in sorted(
            evidence,
            key=lambda ev: (ev.confidence * 0.65 + ev.freshness_score * 0.35),
            reverse=True,
        ):
            add_id(item.evidence_id)
            if len(ordered_ids) >= max_references:
                break

        citation_map: dict[str, int] = {}
        references: list[str] = []
        seen_urls: dict[str, int] = {}
        for evidence_id in ordered_ids:
            ev = evidence_by_id[evidence_id]
            url_key = ev.source_url.rstrip("/").lower()
            if url_key in seen_urls:
                citation_map[evidence_id] = seen_urls[url_key]
                continue
            number = len(references) + 1
            citation_map[evidence_id] = number
            seen_urls[url_key] = number
            title = ev.source_title or ev.title or ev.source_url
            references.append(f"[{number}] {title}. {ev.source_url}")
            if len(references) >= max_references:
                break
        return citation_map, references

    def _build_report_brief(
        self,
        request: ResearchRequest,
        evidence: list[Evidence],
        claims: list[Claim],
        metrics: RunMetrics,
        artifacts: dict[str, Any],
        citation_map: dict[str, int],
    ) -> dict[str, Any]:
        matrix: PaperPatternMatrix | None = artifacts.get("matrix")
        recommendations: list[OpportunityRecommendation] = artifacts.get("recommendations") or []
        observability: ObservabilitySnapshot | None = artifacts.get("observability")
        dimensions = request.analysis_dimensions or DEFAULT_DIMENSIONS
        entities = matrix.papers[:12] if matrix and matrix.papers else [request.target_topic]

        def cite(evidence_ids: list[str], limit: int = 4) -> str:
            seen: list[int] = []
            for evidence_id in evidence_ids:
                number = citation_map.get(evidence_id)
                if number and number not in seen:
                    seen.append(number)
                if len(seen) >= limit:
                    break
            return "".join(f"[{number}]" for number in seen)

        def short(text: str | None, limit: int = 220) -> str:
            clean = " ".join((text or "").split())
            if len(clean) <= limit:
                return clean
            return clean[: limit - 1].rstrip() + "..."

        matrix_cells: list[dict[str, Any]] = []
        if matrix:
            for cell in matrix.cells:
                if cell.dimension not in dimensions or cell.paper not in entities:
                    continue
                matrix_cells.append(
                    {
                        "dimension": DIMENSION_LABELS.get(cell.dimension, cell.dimension),
                        "paper": cell.paper,
                        "summary": short(cell.summary, 170),
                        "citations": cite(cell.evidence_ids, 3),
                    }
                )

        claim_status_counts = Counter(claim.verification_status for claim in claims)
        claim_risk_counts = Counter(note.risk_type for claim in claims for note in claim.red_team_notes)
        failed_gates = [
            {
                "gate_id": gate.gate_id,
                "name": gate.name,
                "status": gate.status,
                "message": gate.message,
                "suggested_action": gate.suggested_action,
            }
            for gate in (observability.quality_gates if observability else [])
            if gate.status == "fail"
        ]
        report_ready = bool(observability) and not failed_gates and (observability.claim_pass_rate >= 0.2)
        if observability and observability.evidence_coverage_score < 0.25:
            report_ready = False
        report_mode = "publication_ready" if report_ready else "exploratory_pilot"

        def claim_payload(claim: Claim, limit: int = 260) -> dict[str, Any] | None:
            citations = cite(claim.supporting_evidence_ids, 4)
            if not citations:
                return None
            return {
                "dimension": DIMENSION_LABELS.get(claim.dimension, claim.dimension),
                "claim": short(claim.final_wording or claim.claim, limit),
                "reasoning": short(claim.reasoning_summary, 180),
                "risk_level": claim.risk_level,
                "verification_status": claim.verification_status,
                "supporting_evidence_count": len(claim.supporting_evidence_ids),
                "red_team_risks": [note.risk_type for note in claim.red_team_notes[:4]],
                "citations": citations,
            }

        top_claims: list[dict[str, Any]] = []
        tentative_claims: list[dict[str, Any]] = []
        do_not_use_as_conclusions: list[dict[str, Any]] = []
        for claim in sorted(claims, key=lambda item: item.confidence, reverse=True):
            payload = claim_payload(claim)
            if not payload:
                continue
            if claim.verification_status == "verified" and claim.risk_level != "high":
                top_claims.append(payload)
            elif claim.verification_status in {"needs_evidence", "challenged"} and claim.risk_level != "high":
                tentative_claims.append(payload)
            else:
                do_not_use_as_conclusions.append(payload)
            if len(top_claims) >= 12 and len(tentative_claims) >= 18 and len(do_not_use_as_conclusions) >= 8:
                break

        from collections import defaultdict

        def evidence_score(ev: Evidence) -> float:
            return ev.confidence * 0.6 + ev.freshness_score * 0.25 + ev.authority_score * 0.15

        grouped: dict[str, dict[str, list[Evidence]]] = defaultdict(lambda: defaultdict(list))
        for item in sorted(evidence, key=evidence_score, reverse=True):
            if item.dimension not in dimensions:
                continue
            paper = item.paper or "其他"
            if paper not in entities:
                continue
            if not citation_map.get(item.evidence_id):
                continue
            bucket = grouped[item.dimension][paper]
            if len(bucket) < 2:
                bucket.append(item)

        evidence_brief: list[dict[str, Any]] = []
        for dimension in dimensions:
            dimension_rows: list[dict[str, Any]] = []
            for entity in entities:
                for item in grouped.get(dimension, {}).get(entity, []):
                    dimension_rows.append(
                        {
                            "paper": entity,
                            "fact": short(item.fact, 220),
                            "quote": short(item.quote, 220),
                            "source_title": short(item.source_title or item.title, 110),
                            "source_type": item.source_type,
                            "published_at": item.published_at.strftime("%Y-%m") if item.published_at else None,
                            "citation": cite([item.evidence_id], 1),
                        }
                    )
            if dimension_rows:
                evidence_brief.append(
                    {
                        "dimension": DIMENSION_LABELS.get(dimension, dimension),
                        "evidence": dimension_rows[:10],
                    }
                )

        return {
            "project": {
                "name": request.project_name,
                "target_topic": request.target_topic,
                "topic_description": request.topic_description,
                "papers": request.seed_papers,
                "research_goal": request.research_goal,
                "analysis_dimensions": [DIMENSION_LABELS.get(item, item) for item in dimensions],
            },
            "run_metrics": {
                "sources_fetched": metrics.sources_fetched,
                "evidence_count": metrics.evidence_count,
                "claim_count": metrics.claim_count,
                "verified_claim_count": metrics.verified_claim_count,
                "challenged_claim_count": metrics.challenged_claim_count,
            },
            "quality_context": {
                "report_mode": report_mode,
                "source_mix": observability.source_mix if observability else {},
                "report_confidence": observability.report_confidence if observability else None,
                "claim_pass_rate": observability.claim_pass_rate if observability else 0,
                "red_team_challenge_rate": observability.red_team_challenge_rate if observability else 0,
                "evidence_coverage_score": observability.evidence_coverage_score if observability else 0,
                "claim_status_counts": dict(claim_status_counts),
                "claim_risk_counts": dict(claim_risk_counts),
                "failed_gates": failed_gates,
                "claim_use_policy": (
                    "Use top_claims as main conclusions. Treat tentative_claims as claim backlog for future evidence "
                    "collection, not as report-body conclusions. Never use do_not_use_as_conclusions as report conclusions."
                ),
            },
            "matrix_cells": matrix_cells,
            "top_claims": top_claims,
            "tentative_claims": tentative_claims[:18],
            "claim_backlog": tentative_claims[:18],
            "do_not_use_as_conclusions": do_not_use_as_conclusions[:8],
            "representative_evidence": evidence_brief,
            "recommendations": [
                {
                    "title": item.title,
                    "recommendation": short(item.recommendation, 240),
                    "rationale": short(item.rationale, 220),
                    "expected_value": short(item.expected_value, 180),
                    "next_steps": [short(step, 120) for step in item.next_steps[:3]],
                    "citations": cite(item.evidence_ids, 4),
                }
                for item in recommendations[:6]
            ],
            "citation_instruction": "Use citation numbers exactly as provided, for example [1][3]. End the report with <<REFERENCES>> under ## 参考来源.",
        }

    async def write_executive_summary(
        self,
        request: ResearchRequest,
        evidence: list[Evidence],
        claims: list[Claim],
        metrics: RunMetrics,
        matrix: PaperPatternMatrix,
        recommendations: list[OpportunityRecommendation],
        observability: ObservabilitySnapshot,
    ) -> str:
        if not self.llm_enabled:
            raise RuntimeError("LLM is not configured; executive summary generation cannot continue")

        failed_gates = [
            gate.model_dump(mode="json")
            for gate in observability.quality_gates
            if gate.status == "fail"
        ]
        report_ready = not failed_gates and observability.claim_pass_rate >= 0.2
        if observability.evidence_coverage_score < 0.25:
            report_ready = False

        def summary_claim(claim: Claim) -> dict[str, Any]:
            return {
                "dimension": DIMENSION_LABELS.get(claim.dimension, claim.dimension),
                "claim": claim.final_wording or claim.claim,
                "confidence": claim.confidence,
                "risk_level": claim.risk_level,
                "verification_status": claim.verification_status,
                "supporting_evidence_count": len(claim.supporting_evidence_ids),
                "red_team_risks": [note.risk_type for note in claim.red_team_notes[:4]],
            }

        verified_claims = [
            summary_claim(claim)
            for claim in sorted(claims, key=lambda item: item.confidence, reverse=True)
            if claim.verification_status == "verified" and claim.risk_level != "high"
        ][:12]
        tentative_claims = [
            summary_claim(claim)
            for claim in sorted(claims, key=lambda item: item.confidence, reverse=True)
            if claim.verification_status in {"needs_evidence", "challenged"} and claim.risk_level != "high"
        ][:12]

        context = {
            "project": request.project_name,
            "target_topic": request.target_topic,
            "papers": request.seed_papers,
            "research_goal": request.research_goal,
            "metrics": metrics.model_dump(mode="json"),
            "coverage_by_dimension": matrix.coverage_by_dimension,
            "coverage_by_paper": matrix.coverage_by_paper,
            "quality_context": {
                "report_mode": "publication_ready" if report_ready else "exploratory_pilot",
                "report_confidence": observability.report_confidence,
                "claim_pass_rate": observability.claim_pass_rate,
                "red_team_challenge_rate": observability.red_team_challenge_rate,
                "evidence_coverage_score": observability.evidence_coverage_score,
                "claim_status_counts": dict(Counter(claim.verification_status for claim in claims)),
                "failed_gates": failed_gates,
            },
            "top_recommendations": [
                item.model_dump(mode="json") for item in recommendations[:6]
            ],
            "verified_claims": verified_claims,
            "tentative_claims": tentative_claims,
            "claim_use_policy": (
                "Key Findings may only use verified_claims. Tentative claims require validation and must be "
                "labeled as observations, not conclusions."
            ),
            "evidence_samples": [
                {
                    "dimension": item.dimension_label,
                    "paper": item.paper,
                    "fact": item.fact,
                    "source_type": item.source_type,
                    "confidence": item.confidence,
                    "published_at": item.published_at.isoformat() if item.published_at else None,
                }
                for item in sorted(evidence, key=lambda ev: ev.confidence, reverse=True)[:36]
            ],
        }

        markdown = await self.invoke_text_strict(
            system=(
                "你是推理模式研究报告的执行摘要写作者。请只基于输入中的证据、Claim、矩阵覆盖和建议来写，"
                "不要编造额外事实，不要暴露 ev_ ID，不要写模板化空话。"
                "输出中文 Markdown，结构为：# Summary、## Key Findings、## Recommended Moves、## Confidence Notes。"
                "Key Findings 只能使用 verified_claims；如果 verified_claims 为空，必须明确写本轮没有足够稳健结论。"
                "tentative_claims 是 claim backlog，只能写入 Confidence Notes 或下一轮补证计划，不能写成发现。"
                "若 quality_context.report_mode 是 exploratory_pilot，Summary 第一段必须标注这是探索性 pilot。"
                "Recommended Moves 写 2-4 条可执行建议；Confidence Notes 简要说明哪些结论较稳、哪些仍需补证。"
            ),
            user=json.dumps(context, ensure_ascii=False, default=str)[:24000],
        )
        markdown = (markdown or "").strip()
        if markdown.startswith("```"):
            markdown = markdown.strip("`").removeprefix("markdown").strip()
        if not markdown:
            raise ValueError("LLM executive summary output was empty")
        return markdown

    async def write_methodology(
        self,
        request: ResearchRequest,
        evidence: list[Evidence],
        claims: list[Claim],
        metrics: RunMetrics,
        matrix: PaperPatternMatrix,
        observability: ObservabilitySnapshot,
    ) -> str:
        if not self.llm_enabled:
            raise RuntimeError("LLM is not configured; methodology generation cannot continue")

        context = {
            "project": request.project_name,
            "target_topic": request.target_topic,
            "papers": request.seed_papers,
            "analysis_dimensions": [
                DIMENSION_LABELS.get(item, item) for item in request.analysis_dimensions
            ],
            "research_goal": request.research_goal,
            "metrics": metrics.model_dump(mode="json"),
            "source_mix": observability.source_mix,
            "dimension_coverage": observability.dimension_coverage,
            "paper_coverage": observability.paper_coverage,
            "quality_gates": [
                gate.model_dump(mode="json") for gate in observability.quality_gates
            ],
            "claim_review": {
                "claim_pass_rate": observability.claim_pass_rate,
                "red_team_challenge_rate": observability.red_team_challenge_rate,
                "verified": metrics.verified_claim_count,
                "challenged": metrics.challenged_claim_count,
            },
            "matrix_coverage": {
                "by_dimension": matrix.coverage_by_dimension,
                "by_paper": matrix.coverage_by_paper,
            },
            "evidence_profile": {
                "total": len(evidence),
                "source_types": dict(Counter(item.source_type for item in evidence)),
                "dimensions": dict(Counter(item.dimension_label for item in evidence)),
            },
            "risk_examples": [
                {
                    "claim": claim.final_wording or claim.claim,
                    "risk_level": claim.risk_level,
                    "notes": [
                        note.model_dump(mode="json") for note in claim.red_team_notes[:3]
                    ],
                }
                for claim in claims
                if claim.red_team_notes
            ][:12],
        }

        markdown = await self.invoke_text_strict(
            system=(
                "你是推理模式研究方法论说明的作者。请基于输入中的真实运行数据说明本次研究如何完成，"
                "包括研究设计、来源采集、Evidence 抽取、Claim 生成与 Red Team 审查、质量门禁和局限性。"
                "不要写成系统介绍，不要暴露内部文件路径，不要把质量门禁简单逐字翻译成表格。"
                "输出中文 Markdown，结构为：# Methodology、## Research Design、## Evidence Pipeline、"
                "## Claim Review、## Quality Controls、## Known Limits。"
            ),
            user=json.dumps(context, ensure_ascii=False, default=str)[:24000],
        )
        markdown = (markdown or "").strip()
        if markdown.startswith("```"):
            markdown = markdown.strip("`").removeprefix("markdown").strip()
        if not markdown:
            raise ValueError("LLM methodology output was empty")
        return markdown


def estimate_coverage(
    candidates: list[SourceCandidate],
    papers: list[str],
    dimensions: list[str],
) -> float:
    """快速估算当前来源对 论文×维度 矩阵的覆盖比例。"""
    if not papers or not dimensions:
        return 0.0
    covered: set[tuple[str, str]] = set()
    for c in candidates:
        text = f"{c.url} {c.title} {c.snippet}".lower()
        for entity in papers:
            if entity.lower() in text:
                for dim in dimensions:
                    kws = dimension_search_keywords(dim)
                    if any(kw.lower() in text for kw in kws):
                        covered.add((entity, dim))
    total = len(papers) * len(dimensions)
    return round(len(covered) / total, 3) if total else 0.0


def deterministic_plan(request: ResearchRequest) -> ResearchPlan:
    entities = [request.target_topic]
    dimensions = request.analysis_dimensions or DEFAULT_DIMENSIONS
    queries: list[str] = []
    source_tasks: list[SourceTask] = []
    for entity in entities:
        if "gap_driven_reframing" in dimensions:
            add_source_task(
                source_tasks,
                entity,
                "gap_driven_reframing",
                "official",
                f"{entity} limitations challenges bottlenecks survey",
                ["academic_paper"],
            )
        for dimension in dimensions:
            if dimension == "gap_driven_reframing":
                continue
            task = make_gap_source_task(entity, dimension, len(source_tasks) + 1)
            source_tasks.append(task)
    queries = [task.query for task in source_tasks]
    return ResearchPlan(
        research_goal=request.research_goal,
        papers=request.seed_papers,
        dimensions=dimensions,
        queries=queries[:18],
        source_tasks=source_tasks[:18],
        required_agents=RESEARCH_AGENT_FLOW.copy(),
        quality_rules=[
            "所有 Evidence 必须绑定论文来源和原文片段",
            "每条关键 Claim 优先绑定 2 条以上 Evidence",
            "来源单一、低置信度或过度推断的 Claim 必须降级措辞",
        ],
        notes="LLM 未配置或调用失败时使用的确定性研究计划；只基于真实抓取内容生成产物。",
        planned_by="ResearchPlanningAgent:deterministic",
    )


def add_source_task(
    tasks: list[SourceTask],
    entity: str,
    dimension: str,
    intent: str,
    query: str,
    expected_source_types: list[str],
) -> None:
    tasks.append(
        SourceTask(
            task_id=f"task_{len(tasks) + 1:02d}",
            entity=entity,
            dimension=dimension,
            intent=intent,  # type: ignore[arg-type]
            query=query,
            expected_source_types=expected_source_types,
            rationale=f"Collect {intent} evidence for {entity} / {dimension}.",
        )
    )


def build_gap_source_tasks(
    request: ResearchRequest,
    plan: ResearchPlan,
    candidates: list[SourceCandidate],
    planned_tasks: list[SourceTask],
    executed_task_ids: set[str],
) -> list[SourceTask]:
    if not request.auto_discover_sources:
        return []

    deduped = dedupe_source_candidates(candidates)
    domains = [source_domain(candidate.url) for candidate in deduped]
    domain_counts = Counter(domain for domain in domains if domain)
    dominant_share = max(domain_counts.values(), default=0) / max(1, len(deduped))
    dimensions = plan.dimensions or request.analysis_dimensions or DEFAULT_DIMENSIONS
    entities = [request.target_topic]
    covered_pairs = infer_candidate_pairs(deduped, planned_tasks, entities, dimensions)

    gap_tasks: list[SourceTask] = []
    used_queries = {task.query.lower() for task in planned_tasks if task.task_id in executed_task_ids}

    for task in planned_tasks:
        if task.task_id in executed_task_ids:
            continue
        if (task.entity, task.dimension) not in covered_pairs:
            append_gap_task(gap_tasks, task, used_queries, "continue planned query for uncovered entity/dimension")
        if len(gap_tasks) >= 4:
            return gap_tasks

    for entity in entities:
        for dimension in dimensions:
            if (entity, dimension) in covered_pairs:
                continue
            task = make_gap_source_task(entity, dimension, len(gap_tasks) + 1)
            append_gap_task(gap_tasks, task, used_queries, "fill entity/dimension coverage gap")
            if len(gap_tasks) >= 4:
                return gap_tasks

    needs_more_sources = len(deduped) < min(request.max_sources, max(4, len(entities) * 2))
    too_concentrated = len(deduped) >= 3 and dominant_share > 0.6
    if needs_more_sources or too_concentrated:
        for entity in entities:
            task = SourceTask(
                task_id=f"gap_diversity_{len(gap_tasks) + 1:02d}",
                entity=entity,
                dimension=dimensions[0] if dimensions else "other",
                intent="comparison",
                query=f"{entity} survey benchmark recent advances",
                expected_source_types=["academic_paper"],
                rationale="Second pass to diversify paper candidates and add survey or benchmark perspective.",
            )
            append_gap_task(gap_tasks, task, used_queries, "diversify source domains")
            if len(gap_tasks) >= 4:
                return gap_tasks

    return gap_tasks


def append_gap_task(
    tasks: list[SourceTask],
    task: SourceTask,
    used_queries: set[str],
    rationale: str,
) -> None:
    query_key = task.query.lower()
    if query_key in used_queries:
        return
    used_queries.add(query_key)
    tasks.append(
        SourceTask(
            task_id=task.task_id if task.task_id.startswith("gap_") else f"gap_{task.task_id}",
            entity=task.entity,
            dimension=task.dimension,
            intent=task.intent,
            query=task.query,
            expected_source_types=task.expected_source_types,
            rationale=f"{rationale}: {task.rationale or task.query}",
        )
    )


def make_gap_source_task(entity: str, dimension: str, index: int) -> SourceTask:
    query_by_dimension = {
        "gap_driven_reframing": f"{entity} limitations challenges bottlenecks problem reframing",
        "cross_domain_synthesis": f"{entity} cross-domain synthesis knowledge graph multimodal integration",
        "representation_shift": f"{entity} representation learning embedding latent structure transformation",
        "modular_pipeline_composition": f"{entity} modular pipeline agent framework tool composition",
        "data_evaluation_engineering": f"{entity} benchmark dataset evaluation metric annotation",
        "principled_probabilistic_modeling": f"{entity} probabilistic modeling uncertainty bayesian inference",
        "formal_experimental_tightening": f"{entity} theorem proof ablation controlled experiment rigorous evaluation",
        "approximation_engineering": f"{entity} approximation heuristic scalable efficient inference sampling",
        "inference_time_control": f"{entity} inference-time control decoding planning self-correction",
        "structural_inductive_bias": f"{entity} structural inductive bias graph constraints hierarchy",
        "multiscale_hierarchical_modeling": f"{entity} multiscale hierarchical modeling coarse-to-fine reasoning",
        "mechanistic_decomposition": f"{entity} mechanism decomposition interpretability causal analysis",
        "adversary_modeling": f"{entity} adversarial robustness attack defense red teaming",
        "numerics_systems_codesign": f"{entity} systems codesign quantization kernel latency throughput",
        "data_centric_optimization": f"{entity} data-centric optimization data selection curation augmentation",
        "kg_rag_architecture": f"{entity} GraphRAG knowledge graph retrieval augmented generation architecture framework",
        "graph_retrieval_grounding": f"{entity} graph retrieval grounding entity linking subgraph retrieval evidence",
        "kg_rag_evaluation": f"{entity} benchmark evaluation knowledge graph RAG retrieval grounded generation",
        "multi_hop_kg_reasoning": f"{entity} multi-hop knowledge graph reasoning retrieval augmented generation",
        "kg_construction_for_rag": f"{entity} knowledge graph construction extraction for retrieval augmented generation",
        "rag_kg_boundary_analysis": f"{entity} GraphRAG KGQA KG construction boundary analysis retrieval augmented generation",
        "application_boundary_cases": f"{entity} application case study domain-specific KG-RAG limitations",
        "scientific_problem_benchmarking": f"{entity} scientific reasoning benchmark dataset problem solving evaluation",
        "tool_augmented_scientific_reasoning": f"{entity} scientific reasoning tool augmented solver code execution simulation",
        "domain_grounding_verification": f"{entity} domain grounding verification factual scientific evidence",
        "multimodal_scientific_reasoning": f"{entity} multimodal scientific reasoning diagrams tables equations",
        "scientific_error_analysis": f"{entity} scientific reasoning error analysis hallucination robustness",
        "lab_workflow_reasoning": f"{entity} scientific discovery lab workflow hypothesis experiment planning",
        "math_benchmark_evaluation": f"{entity} mathematical reasoning benchmark dataset evaluation GSM8K MATH",
        "formal_proof_symbolic_reasoning": f"{entity} formal proof symbolic reasoning theorem proving",
        "program_tool_augmented_solving": f"{entity} program aided mathematical reasoning tool augmented solving code execution",
        "self_consistency_search_verification": f"{entity} self-consistency tree search verifier mathematical reasoning",
        "natural_language_to_formal_math": f"{entity} natural language to formal math equations symbolic representation",
        "math_error_diagnosis": f"{entity} mathematical reasoning error analysis process supervision verification",
        "llm_causal_benchmarking": f"{entity} benchmark large language models causal reasoning statistical pitfalls",
        "causal_intervention_counterfactual": f"{entity} intervention counterfactual causal reasoning large language models",
        "causal_explanation_mechanism": f"{entity} causal explanation mechanism analysis large language models",
        "causal_pitfall_robustness": f"{entity} causal reasoning robustness confounding spurious correlation benchmark",
        "causal_tool_symbolic_integration": f"{entity} causal graph symbolic tool augmented large language model reasoning",
        "causal_reasoning_evaluation_protocol": f"{entity} causal reasoning evaluation protocol datasets metrics LLM",
        "treatment_effect_estimation": f"{entity} treatment effect estimation counterfactual machine learning",
        "core_counterfactual_inference": f"{entity} counterfactual inference potential outcomes structural causal model",
        "counterfactual_explanation_fairness": f"{entity} counterfactual explanation fairness recourse causal inference",
        "temporal_counterfactual_estimation": f"{entity} temporal counterfactual estimation longitudinal treatment effect",
        "identifiability_assumption_sensitivity": f"{entity} counterfactual identifiability assumptions sensitivity analysis",
        "counterfactual_benchmarking": f"{entity} counterfactual inference benchmark dataset evaluation",
        "graph_reasoning_benchmark": f"{entity} graph reasoning benchmark multi-hop question answering knowledge graph",
        "kgqa_or_graph_reasoning": f"{entity} knowledge graph question answering multi-hop graph reasoning",
        "core_multi_hop_graph_reasoning": f"{entity} multi-hop graph reasoning path reasoning knowledge graphs",
        "semantic_parsing_grounding": f"{entity} semantic parsing grounding knowledge graph question answering",
        "path_composition_reasoning": f"{entity} path composition relation chain reasoning knowledge graph",
        "graph_retrieval_boundary": f"{entity} graph retrieval RAG adjacent multi-hop reasoning boundary",
    }
    return SourceTask(
        task_id=f"gap_coverage_{index:02d}",
        entity=entity,
        dimension=dimension,
        intent="comparison",
        query=query_by_dimension.get(dimension, f"{entity} {dimension} academic papers"),
        expected_source_types=["academic_paper"],
        rationale="Generated after first-pass paper search found low coverage.",
    )


def infer_candidate_dimensions(
    candidates: list[SourceCandidate],
    planned_tasks: list[SourceTask],
    dimensions: list[str],
) -> set[str]:
    dimensions_set = set(dimensions or DEFAULT_DIMENSIONS)
    by_query = {task.query.lower(): task.dimension for task in planned_tasks}
    covered: set[str] = set()
    for candidate in candidates:
        task_dimension = by_query.get(candidate.query.lower())
        if task_dimension in dimensions_set:
            covered.add(task_dimension)
        text = f"{candidate.url} {candidate.title} {candidate.snippet} {candidate.source_type}".lower()
        for dimension in dimensions_set:
            if dimension in text or any(keyword.lower() in text for keyword in dimension_search_keywords(dimension)):
                covered.add(dimension)
    return covered


def infer_candidate_entities(candidates: list[SourceCandidate], entities: list[str]) -> set[str]:
    covered: set[str] = set()
    for candidate in candidates:
        text = f"{candidate.url} {candidate.title} {candidate.snippet} {candidate.query}".lower()
        host_key = "".join(part for part in source_domain(candidate.url).split(".") if part)
        for entity in entities:
            entity_key = "".join(ch for ch in entity.lower() if ch.isalnum())
            if entity.lower() in text or (entity_key and entity_key in host_key):
                covered.add(entity)
    return covered


def infer_candidate_pairs(
    candidates: list[SourceCandidate],
    planned_tasks: list[SourceTask],
    entities: list[str],
    dimensions: list[str],
) -> set[tuple[str, str]]:
    by_query = {task.query.lower(): (task.entity, task.dimension) for task in planned_tasks}
    pairs: set[tuple[str, str]] = set()
    dimensions_set = set(dimensions or DEFAULT_DIMENSIONS)
    for candidate in candidates:
        planned_pair = by_query.get(candidate.query.lower())
        if planned_pair and planned_pair[0] in entities and planned_pair[1] in dimensions_set:
            pairs.add(planned_pair)
            continue
        candidate_entities = infer_candidate_entities([candidate], entities)
        candidate_dimensions = infer_candidate_dimensions([candidate], planned_tasks, dimensions)
        for entity in candidate_entities:
            for dimension in candidate_dimensions:
                pairs.add((entity, dimension))
    return pairs


def candidate_pair_counts(
    candidates: list[SourceCandidate],
    planned_tasks: list[SourceTask],
    entities: list[str],
    dimensions: list[str],
) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for candidate in dedupe_source_candidates(candidates):
        pairs = infer_candidate_pairs([candidate], planned_tasks, entities, dimensions)
        for pair in pairs:
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def unique_tasks_by_query(tasks: list[SourceTask]) -> list[SourceTask]:
    seen: set[str] = set()
    unique: list[SourceTask] = []
    for task in tasks:
        key = task.query.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(task)
    return unique


def provider_display_name(provider: str) -> str:
    return {
        "tavily": "Tavily",
        "exa": "Exa",
        "zhihu_inner": "知乎站内",
        "zhihu_global": "知乎全网",
        "search": "Search",
    }.get(provider, provider or "搜索引擎")


def dimension_search_keywords(dimension: str) -> list[str]:
    return {
        "gap_driven_reframing": ["gap", "limitation", "challenge", "bottleneck", "reframe"],
        "cross_domain_synthesis": ["cross-domain", "synthesis", "integrate", "hybrid", "knowledge graph"],
        "representation_shift": ["representation", "embedding", "latent", "encode", "token"],
        "modular_pipeline_composition": ["pipeline", "module", "component", "framework", "agent"],
        "data_evaluation_engineering": ["benchmark", "dataset", "evaluation", "metric", "annotation"],
        "principled_probabilistic_modeling": ["probabilistic", "bayesian", "uncertainty", "distribution"],
        "formal_experimental_tightening": ["theorem", "proof", "ablation", "controlled experiment", "rigorous"],
        "approximation_engineering": ["approximation", "heuristic", "efficient", "scalable", "sampling"],
        "inference_time_control": ["inference-time", "decoding", "test-time", "planning", "search"],
        "structural_inductive_bias": ["inductive bias", "structure", "graph", "hierarchy", "constraint"],
        "multiscale_hierarchical_modeling": ["hierarchical", "multi-scale", "coarse-to-fine", "granularity"],
        "mechanistic_decomposition": ["mechanism", "decompose", "interpretability", "causal"],
        "adversary_modeling": ["adversarial", "robust", "attack", "defense", "red-team"],
        "numerics_systems_codesign": ["system", "hardware", "kernel", "quantization", "latency"],
        "data_centric_optimization": ["data-centric", "data selection", "curation", "augmentation", "quality"],
        "kg_rag_architecture": ["graphrag", "kg-rag", "knowledge graph rag", "architecture", "framework"],
        "graph_retrieval_grounding": ["subgraph retrieval", "entity linking", "grounding", "retrieval"],
        "kg_rag_evaluation": ["benchmark", "evaluation", "metric", "grounded generation", "retrieval quality"],
        "multi_hop_kg_reasoning": ["multi-hop", "multi hop", "knowledge graph reasoning", "kg reasoning"],
        "kg_construction_for_rag": ["knowledge graph construction", "kg construction", "extraction", "ontology"],
        "rag_kg_boundary_analysis": ["boundary", "adjacent", "kgqa", "kg construction", "graphrag"],
        "application_boundary_cases": ["application", "case study", "domain-specific", "medical", "recommendation"],
        "scientific_problem_benchmarking": ["scientific benchmark", "problem solving", "scibench", "scienceqa"],
        "tool_augmented_scientific_reasoning": ["tool", "code execution", "simulation", "solver"],
        "domain_grounding_verification": ["grounding", "verification", "factual", "evidence"],
        "multimodal_scientific_reasoning": ["multimodal", "diagram", "table", "equation"],
        "scientific_error_analysis": ["error analysis", "hallucination", "robustness", "pitfall"],
        "lab_workflow_reasoning": ["scientific discovery", "hypothesis", "experiment", "workflow"],
        "math_benchmark_evaluation": ["gsm8k", "math", "benchmark", "evaluation"],
        "formal_proof_symbolic_reasoning": ["formal proof", "symbolic", "theorem", "proof"],
        "program_tool_augmented_solving": ["program", "code execution", "tool", "python"],
        "self_consistency_search_verification": ["self-consistency", "tree search", "verifier", "verification"],
        "natural_language_to_formal_math": ["formal math", "equation", "symbolic", "latex"],
        "math_error_diagnosis": ["error", "process supervision", "reasoning trace", "diagnosis"],
        "llm_causal_benchmarking": ["causal benchmark", "statistical pitfall", "causal reasoning", "llm"],
        "causal_intervention_counterfactual": ["intervention", "counterfactual", "do-calculus", "causal"],
        "causal_explanation_mechanism": ["causal explanation", "mechanism", "causal graph"],
        "causal_pitfall_robustness": ["confounding", "spurious", "pitfall", "robustness"],
        "causal_tool_symbolic_integration": ["causal graph", "symbolic", "tool", "structural causal"],
        "causal_reasoning_evaluation_protocol": ["evaluation protocol", "dataset", "metric", "causal"],
        "treatment_effect_estimation": ["treatment effect", "cate", "heterogeneous causal effect"],
        "core_counterfactual_inference": ["counterfactual inference", "potential outcome", "structural causal"],
        "counterfactual_explanation_fairness": ["counterfactual explanation", "fairness", "recourse"],
        "temporal_counterfactual_estimation": ["temporal", "longitudinal", "over time", "counterfactual"],
        "identifiability_assumption_sensitivity": ["identifiability", "assumption", "sensitivity"],
        "counterfactual_benchmarking": ["counterfactual benchmark", "dataset", "evaluation"],
        "graph_reasoning_benchmark": ["graph reasoning benchmark", "multi-hop benchmark", "question answering"],
        "kgqa_or_graph_reasoning": ["kgqa", "knowledge graph question answering", "graph reasoning"],
        "core_multi_hop_graph_reasoning": ["multi-hop graph", "path reasoning", "relation chain"],
        "semantic_parsing_grounding": ["semantic parsing", "logical form", "grounding"],
        "path_composition_reasoning": ["path composition", "relation composition", "reasoning path"],
        "graph_retrieval_boundary": ["graph retrieval", "rag", "retrieval boundary"],
    }.get(dimension, [dimension])


def dedupe_source_candidates(candidates: list[SourceCandidate]) -> list[SourceCandidate]:
    seen: set[str] = set()
    deduped: list[SourceCandidate] = []
    for candidate in candidates:
        normalized = normalize_url(candidate.url)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        candidate.url = normalized
        if not candidate.source_type or candidate.source_type == "other":
            candidate.source_type = classify_source(candidate.url, candidate.title)
        candidate.score = round(min(1.0, max(0.1, candidate.score)), 3)
        deduped.append(candidate)
    return deduped


def select_balanced_candidates(candidates: list[SourceCandidate], limit: int) -> list[SourceCandidate]:
    deduped = sorted(dedupe_source_candidates(candidates), key=lambda item: item.score, reverse=True)
    if len(deduped) <= limit:
        return deduped

    selected: list[SourceCandidate] = []
    domain_counts: Counter[str] = Counter()
    for candidate in deduped:
        domain = source_domain(candidate.url)
        if domain and domain_counts[domain] >= 2:
            continue
        selected.append(candidate)
        if domain:
            domain_counts[domain] += 1
        if len(selected) >= limit:
            return selected

    selected_urls = {candidate.url for candidate in selected}
    for candidate in deduped:
        if candidate.url in selected_urls:
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def source_domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def score_candidate_for_task(candidate: SourceCandidate, task: SourceTask, search_round: str) -> float:
    score = candidate.score or 0.5
    if candidate.source_type in task.expected_source_types:
        score += 0.08
    if task.entity and task.entity.lower() in f"{candidate.url} {candidate.title}".lower():
        score += 0.05
    if search_round == "gap":
        score += 0.03
    return round(min(1.0, max(0.1, score)), 3)


def tasks_from_queries(queries: list[str]) -> list[SourceTask]:
    return [
        SourceTask(
            task_id=f"task_{index + 1:02d}",
            entity="unknown",
            dimension="other",
            intent="official",
            query=query,
            expected_source_types=[],
            rationale="Fallback task generated from planner query.",
        )
        for index, query in enumerate(queries)
    ]


def normalize_source_tasks(value: Any, fallback: list[SourceTask]) -> list[SourceTask]:
    if not isinstance(value, list):
        return fallback
    tasks: list[SourceTask] = []
    allowed_intents = {"official", "docs", "review", "news", "comparison"}
    for row in value[:24]:
        if not isinstance(row, dict):
            continue
        query = str(row.get("query") or "").strip()
        if len(query) < 4:
            continue
        intent = str(row.get("intent") or "official").strip().lower()
        if intent not in allowed_intents:
            intent = "official"
        dimension = str(row.get("dimension") or "other").strip()
        if dimension not in DIMENSION_LABELS:
            dimension = "other"
        tasks.append(
            SourceTask(
                task_id=str(row.get("task_id") or f"task_{len(tasks) + 1:02d}"),
                entity=str(row.get("entity") or "unknown").strip() or "unknown",
                dimension=dimension,
                intent=intent,  # type: ignore[arg-type]
                query=query,
                expected_source_types=coerce_str_list(row.get("expected_source_types")),
                rationale=str(row.get("rationale") or ""),
            )
        )
    return tasks or fallback


def deterministic_red_team(claims: list[Claim], evidence: list[Evidence]) -> list[Claim]:
    by_id = {ev.evidence_id: ev for ev in evidence}
    for claim in claims:
        supporting = [by_id[ev_id] for ev_id in claim.supporting_evidence_ids if ev_id in by_id]
        source_domains = {urlparse(ev.source_url).netloc for ev in supporting}
        source_papers = {ev.paper or ev.source_title or ev.source_url for ev in supporting}
        source_types = {ev.source_type for ev in supporting}
        academic_evidence = bool(source_types) and source_types.issubset({"academic_paper", "local_paper"})
        claim.source_paper_count = len(source_papers)
        if claim.source_paper_count >= 2:
            if claim.claim_type != "cross_role_contrast":
                claim.claim_type = "comparative"
            claim.evidence_support_level = "strong" if len(supporting) >= 2 else "weak"
        elif len(supporting) >= 2:
            claim.claim_type = "single_paper_observation"
            claim.evidence_support_level = "single_paper"
        else:
            claim.claim_type = "backlog"
            claim.evidence_support_level = "insufficient"
            claim.backlog_reason = claim.backlog_reason or "single_evidence_claim"
        notes: list[RedTeamNote] = []
        if len(supporting) < 2:
            notes.append(
                RedTeamNote(
                    risk_type="insufficient_evidence",
                    comment="该结论目前少于 2 条支持证据。",
                    suggested_action="补充更多来源后再提升表达强度。",
                    severity="high",
                )
            )
        if claim.claim_type == "comparative" and claim.source_paper_count < 2:
            notes.append(
                RedTeamNote(
                    risk_type="single_source",
                    comment="跨论文 synthesis 未达到 2 篇独立论文支撑。",
                    suggested_action="补充独立论文证据，或降级为 single_paper_observation。",
                    severity="high",
                )
            )
        elif supporting and not academic_evidence and len(source_domains) < 2:
            notes.append(
                RedTeamNote(
                    risk_type="single_domain",
                    comment="非学术来源主要来自单一域名，存在来源单一风险。",
                    suggested_action="补充独立第三方资料后再提升表达强度。",
                    severity="medium",
                )
            )
        if claim.confidence < 0.65:
            notes.append(
                RedTeamNote(
                    risk_type="low_confidence",
                    comment="当前证据质量或相关性不足以支撑强断言。",
                    suggested_action="在报告中使用谨慎措辞，并标记为待验证。",
                    severity="medium",
                )
            )
        claim.red_team_notes = notes
        if notes:
            claim.verification_status = "needs_evidence"
            claim.risk_level = "high" if any(note.severity == "high" for note in notes) else "medium"
            claim.final_wording = claim.claim.replace("显示", "初步显示")
        else:
            claim.verification_status = "verified"
            claim.risk_level = "low"
            claim.final_wording = claim.claim
    return claims


def merge_claims(primary: list[Claim], fallback: list[Claim], limit: int = 40) -> list[Claim]:
    merged: list[Claim] = []
    seen: set[str] = set()
    for claim in [*primary, *fallback]:
        support_key = ",".join(sorted(claim.supporting_evidence_ids[:8]))
        text_key = " ".join(claim.claim.lower().split())[:160]
        key = f"{claim.dimension}:{support_key}:{text_key}"
        if key in seen:
            continue
        seen.add(key)
        merged.append(claim)
        if len(merged) >= limit:
            break
    return merged


def coerce_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def format_dimension_catalog(dimensions: list[str]) -> str:
    lines = []
    for dimension in dimensions:
        lines.append(f"- {dimension}: {DIMENSION_LABELS.get(dimension, dimension)}")
    return "\n".join(lines)


def normalize_dimensions(values: list[str]) -> list[str]:
    reverse = {label: key for key, label in DIMENSION_LABELS.items()}
    normalized: list[str] = []
    for value in values:
        item = value.strip()
        if item in DIMENSION_LABELS:
            normalized.append(item)
        elif item in reverse:
            normalized.append(reverse[item])
    return normalized


DIMENSION_HINTS: dict[str, tuple[str, ...]] = {
    "gap_driven_reframing": ("pain point", "failure analysis", "method reconstruction", "痛点", "失败分析"),
    "cross_domain_synthesis": ("cross-domain", "cross domain", "multimodal integration", "跨领域"),
    "representation_shift": (
        "representation transformation",
        "representation shift",
        "symbolic representation",
        "natural language to symbolic",
        "符号表达",
        "表征",
    ),
    "modular_pipeline_composition": ("modular pipeline", "planner solver verifier", "multi step pipeline", "模块化", "管线"),
    "data_evaluation_engineering": ("benchmark", "dataset", "evaluation", "contamination", "数据", "评测"),
    "principled_probabilistic_modeling": ("probabilistic", "uncertainty", "bayesian", "calibration", "概率", "不确定性"),
    "formal_experimental_tightening": ("controlled experiment", "theory experiment", "process supervision", "ablation", "理论实验", "实验"),
    "approximation_engineering": ("approximation", "approximate", "efficiency", "compression", "近似"),
    "inference_time_control": ("inference time", "test time", "self consistency", "tree search", "verifier guided", "推理时"),
    "structural_inductive_bias": ("structural inductive bias", "expression tree", "proof graph", "结构归纳"),
    "multiscale_hierarchical_modeling": ("hierarchical", "multi-scale", "multiscale", "多尺度", "分层"),
    "mechanistic_decomposition": ("mechanism decomposition", "mechanistic", "decomposition of reasoning", "机制分解"),
    "adversary_modeling": ("adversarial", "robustness", "red team", "对抗"),
    "numerics_systems_codesign": ("numeric", "numerical", "systems codesign", "数值", "系统协同"),
    "data_centric_optimization": ("data-centric", "data centric", "instruction tuning", "synthetic data", "数据中心"),
    "kg_rag_architecture": ("graphrag", "kg-rag", "knowledge graph rag", "architecture", "framework"),
    "graph_retrieval_grounding": ("subgraph retrieval", "entity linking", "grounding", "retrieval grounding"),
    "kg_rag_evaluation": ("kg-rag benchmark", "graph rag benchmark", "grounded generation evaluation"),
    "multi_hop_kg_reasoning": ("multi-hop", "multi hop", "knowledge graph reasoning", "kg reasoning"),
    "kg_construction_for_rag": ("kg construction", "knowledge graph construction", "extraction for rag"),
    "rag_kg_boundary_analysis": ("boundary analysis", "kgqa", "kg construction", "graphrag adjacent"),
    "application_boundary_cases": ("application", "case study", "domain-specific", "boundary case"),
    "scientific_problem_benchmarking": ("scibench", "scienceqa", "scientific benchmark", "problem solving"),
    "tool_augmented_scientific_reasoning": ("tool augmented", "code execution", "simulation", "solver"),
    "domain_grounding_verification": ("domain grounding", "verification", "factual scientific", "evidence"),
    "multimodal_scientific_reasoning": ("diagram", "table", "equation", "multimodal scientific"),
    "scientific_error_analysis": ("error analysis", "hallucination", "robustness", "misleading evidence"),
    "lab_workflow_reasoning": ("scientific discovery", "hypothesis", "experiment planning", "lab workflow"),
    "math_benchmark_evaluation": ("gsm8k", "math benchmark", "mathematical benchmark", "evaluation"),
    "formal_proof_symbolic_reasoning": ("formal proof", "symbolic reasoning", "theorem proving"),
    "program_tool_augmented_solving": ("program aided", "code execution", "tool augmented", "python"),
    "self_consistency_search_verification": ("self-consistency", "tree search", "verifier", "verification"),
    "natural_language_to_formal_math": ("formal math", "equations", "symbolic representation", "latex"),
    "math_error_diagnosis": ("math error", "process supervision", "reasoning trace", "diagnosis"),
    "llm_causal_benchmarking": ("causal benchmark", "statistical pitfalls", "causal reasoning benchmark"),
    "causal_intervention_counterfactual": ("intervention", "counterfactual", "do-calculus", "causal query"),
    "causal_explanation_mechanism": ("causal explanation", "mechanism", "causal graph explanation"),
    "causal_pitfall_robustness": ("confounding", "spurious correlation", "pitfall", "robustness"),
    "causal_tool_symbolic_integration": ("causal graph", "symbolic", "tool augmented", "structural causal"),
    "causal_reasoning_evaluation_protocol": ("evaluation protocol", "causal dataset", "metrics"),
    "treatment_effect_estimation": ("treatment effect", "cate", "heterogeneous causal effect"),
    "core_counterfactual_inference": ("counterfactual inference", "potential outcome", "structural causal model"),
    "counterfactual_explanation_fairness": ("counterfactual explanation", "fairness", "recourse"),
    "temporal_counterfactual_estimation": ("temporal counterfactual", "longitudinal", "over time"),
    "identifiability_assumption_sensitivity": ("identifiability", "assumption", "sensitivity analysis"),
    "counterfactual_benchmarking": ("counterfactual benchmark", "counterfactual dataset", "evaluation"),
    "graph_reasoning_benchmark": ("graph reasoning benchmark", "multi-hop benchmark", "graph question answering benchmark"),
    "kgqa_or_graph_reasoning": ("kgqa", "knowledge graph question answering", "graph reasoning"),
    "core_multi_hop_graph_reasoning": ("multi-hop graph", "path reasoning", "relation chain"),
    "semantic_parsing_grounding": ("semantic parsing", "logical form", "grounding"),
    "path_composition_reasoning": ("path composition", "relation composition", "reasoning path"),
    "graph_retrieval_boundary": ("graph retrieval", "rag adjacent", "retrieval boundary"),
}

DIMENSION_ALIASES: dict[str, str] = {
    "counterfactual_benchmark_evaluation": "counterfactual_benchmarking",
    "counterfactual_explanation_and_fairness": "counterfactual_explanation_fairness",
    "counterfactual_explanation_or_fairness": "counterfactual_explanation_fairness",
    "identifiability_and_assumption_sensitivity": "identifiability_assumption_sensitivity",
}


def normalize_dimension_key(value: Any, context: str = "") -> str:
    raw = str(value or "").strip()
    if raw in DIMENSION_LABELS:
        return raw

    lower_raw = raw.lower()
    alias = DIMENSION_ALIASES.get(lower_raw)
    if alias:
        return alias

    for key, label in DIMENSION_LABELS.items():
        if lower_raw == key.lower() or raw == label:
            return key

    raw_context = f"{raw} {context}"
    combined = raw_context.lower()
    for key, label in DIMENSION_LABELS.items():
        if key.lower() in combined or (label and label in raw_context):
            return key

    best_hint_match: tuple[int, str] | None = None
    for key, hints in DIMENSION_HINTS.items():
        for hint in hints:
            lowered_hint = hint.lower()
            if lowered_hint in combined:
                score = len(lowered_hint)
                if best_hint_match is None or score > best_hint_match[0]:
                    best_hint_match = (score, key)
    if best_hint_match:
        return best_hint_match[1]
    return "other"


def normalize_claim_dimension(
    value: Any,
    context: str,
    supporting_dimensions: list[str],
    valid_dimensions: set[str],
) -> str:
    dimension = normalize_dimension_key(value, context)
    if dimension in valid_dimensions:
        return dimension

    counts = Counter(dim for dim in supporting_dimensions if dim in valid_dimensions)
    if len(counts) == 1:
        return next(iter(counts))

    return ""


def normalize_agent_flow(values: list[str]) -> list[str]:
    requested = [AGENT_ALIASES.get(value, value) for value in values]
    requested_set = set(requested)
    ordered = [agent for agent in RESEARCH_AGENT_FLOW if agent in requested_set or not values]
    for agent in RESEARCH_AGENT_FLOW:
        if agent not in ordered:
            ordered.append(agent)
    return ordered


def clamp_float(value: Any, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = low
    return round(max(low, min(high, parsed)), 3)


# 按推理模式的时效半衰期（天）：系统/推理时控制更新更快，基础建模和理论相对更慢。
_FRESHNESS_HALF_LIFE: dict[str, float] = {
    "inference_time_control": 180,
    "numerics_systems_codesign": 180,
    "data_centric_optimization": 240,
    "data_evaluation_engineering": 300,
    "modular_pipeline_composition": 300,
    "adversary_modeling": 300,
    "cross_domain_synthesis": 365,
    "representation_shift": 365,
    "approximation_engineering": 365,
    "mechanistic_decomposition": 365,
    "structural_inductive_bias": 540,
    "multiscale_hierarchical_modeling": 540,
    "gap_driven_reframing": 730,
    "principled_probabilistic_modeling": 730,
    "formal_experimental_tightening": 730,
}

def compute_freshness_score(published_at: datetime | None, dimension: str) -> float:
    """
    根据发布时间和维度计算时效分数。
    - published_at 未知 → 返回 0.5（中性，不奖不罚）
    - 文章越新分数越高，按维度设定衰减半衰期
    - 公式：score = 0.85 * 0.5^(days / half_life) + 0.15，范围约 [0.15, 1.0]
    """
    if published_at is None:
        return 0.5
    now = datetime.now(timezone.utc)
    pub = published_at if published_at.tzinfo else published_at.replace(tzinfo=timezone.utc)
    days_old = max(0, (now - pub).days)
    half_life = _FRESHNESS_HALF_LIFE.get(dimension, 365)
    raw = 0.85 * (0.5 ** (days_old / half_life)) + 0.15
    return round(min(1.0, max(0.15, raw)), 3)


def make_stable_id(prefix: str, text: str) -> str:
    import hashlib

    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{prefix}_{digest}"


# Compatibility aliases for old run manifests and stray imports.
ChiefResearchPlannerAgent = ResearchPlanningAgent
SourceScoutAgent = SourceResearchAgent
CollectionAgent = SourceResearchAgent
EvidenceExtractionAgent = EvidenceStructuringAgent
ClaimGenerationAgent = AnalysisAndReviewAgent
RedTeamAgent = AnalysisAndReviewAgent
InsightSynthesisAgent = AnalysisAndReviewAgent
ReportWriterAgent = ReportComposerAgent
