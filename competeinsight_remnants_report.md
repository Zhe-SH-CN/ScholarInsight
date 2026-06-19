# CompeteInsight 残留扫描报告

扫描时间: 2026-06-19
扫描范围: backend/cg/**/*.py, skills/*.yaml, frontend/src/App.tsx

---

## 一、后端 Python 文件

### 1. `cg/agents/research_agents.py` (最严重，20+ 处)

**System Prompt 残留:**
- L109: `f"- 竞品【{g.competitor}】的【{DIMENSION_LABELS.get(g.dimension, g.dimension)}】"` — feedback context
- L171: `f"目标产品：{request.target_product}"` — ResearchPlanningAgent user prompt
- L173: `f"\n竞品列表：{', '.join(request.competitors)}\n"` — ResearchPlanningAgent user prompt
- L500: `"2. 定价维度必须尽量有官方定价页或可信 pricing 来源\n"` — SourceResearchAgent
- L513: `'  "next_tasks": [{"entity":"产品","dimension":"pricing","intent":"pricing","query":"...","rationale":"..."}]\n'` — SourceResearchAgent
- L519: `f"目标产品：{request.target_product}\n"` — SourceResearchAgent
- L520: `f"竞品：{', '.join(request.competitors)}\n"` — SourceResearchAgent
- L673: `f"目标产品：{request.target_product}"` — SourceResearchAgent
- L675: `f"\n竞品：{', '.join(request.competitors)}\n"` — SourceResearchAgent
- L983: `f"目标产品：{request.target_product}\n"` — EvidenceStructuringAgent
- L984: `f"竞品：{', '.join(request.competitors)}\n"` — EvidenceStructuringAgent
- L1053: `target = request.target_product if request else "目标产品"` — AnalysisAndReviewAgent
- L1054: `competitors_str = ", ".join(request.competitors) if request else "竞品"` — AnalysisAndReviewAgent
- L1100: `f"竞品：{competitors_str}\n\n"` — AnalysisAndReviewAgent
- L1404: `"你是竞品分析审查智能体（AnalysisAndReviewAgent）的证据缺口评估器。\n"` — assess_gaps
- L1413: `"  3. 每个核心产品（目标产品 + 主要竞品）在至少 3 个维度有证据\n"` — assess_gaps
- L1414: `"  4. 若包含 user_voice 维度，必须已有来自真实用户或第三方社区的内容\n"` — assess_gaps
- L1421: `"  medium（建议补）：主要竞品在关键维度（定价/功能/用户声音）没有证据或证据较弱\n"` — assess_gaps

**代码逻辑残留:**
- L186: `allowed_intents = {"official", "pricing", "docs", "changelog", "review", "enterprise", "news", "comparison"}`
- L555: 同上
- L703: 同上
- L574: `expected_source_types=["user_review"] if dimension == "user_voice" else []`
- L750: `if "user_voice" in dimensions:`
- L928: `is_user_voice = document.source_type in {"user_review", "review_platform"}`
- L1347: `or dim in {"positioning", "feature", "pricing", "user_voice", "enterprise"}`

### 2. `cg/orchestrator/pipeline.py` (15+ 处)

- L31: `BattlecardItem` import
- L33-34: `CompetitorMatrix, CompetitorProfile` imports
- L41: `MatrixCell` import
- L43: `OpportunityRecommendation` import
- L185: `"⚠️ LLM API Key 未配置。CompeteGraph 需要 LLM 才能运行智能体搜索与分析。"`
- L330: `competitors = [request.target_product, *request.competitors]`
- L331: `interim_matrix = build_competitor_matrix(...)`
- L411: `matrix_builder=build_competitor_matrix`
- L413: `battlecards_builder=build_battlecards`
- L422: `metrics.battlecard_count = len(artifacts["battlecards"])`
- L468: `target_product=urlparse(str(request.url)).netloc`
- L649: `competitors=request.competitors`
- L659: `entities = [request.target_product, *request.competitors]`
- L794: `f"跳过 {skipped_documents} 篇文档：对应竞品×维度 Evidence 已较充分"`
- L831: 同上
- L847-875: `generate_claims` 中大量 competitor 相关逻辑
- L904: `battlecards = artifacts["battlecards"]`
- L911-912: battlecards 导出
- L923-924: `CompetitorMatrix, OpportunityRecommendation` 类型注解

### 3. `cg/schemas/research.py` (20+ 处)

- L1: `"""Core schemas for the first end-to-end CompeteGraph research loop."""`
- L53: `target_product: str = Field(default="Retrieval-Augmented Generation")`
- L55: `competitors: list[str] = Field(default_factory=lambda: [])`
- L113: `competitors: list[str]` (ResearchPlan)
- L154: `competitor: str | None = None` (Evidence)
- L177: `competitor: str | None = None` (EvidenceSummary)
- L247: `battlecard_count: int = 0` (RunMetrics)
- L255: `target_product: str` (RunStatus)
- L275-276: `class CompetitorProfile`, `competitor: str`
- L286-287: `class MatrixCell`, `competitor: str`
- L298-305: `class CompetitorMatrix` (整个类)
- L309: `class OpportunityRecommendation` (整个类)
- L324-328: `class BattlecardItem` (整个类)
- L353: `competitor_coverage: dict[str, float]`
- L365: `node_type: Literal["claim", "evidence", "source", "competitor", "dimension"]`
- L393-395: `matrix, recommendations, battlecards` 字段
- L417: `competitor: str` (ResearchGap)

### 4. `cg/settings.py` (1 处)

- L88: `cg_user_agent: str = "CompeteGraphBot/0.1"`

### 5. `cg/api/runs.py` (10+ 处)

- L32: `competitors: list[dict]` (SuggestCompetitorsResponse)
- L36: `# ── LLM 必要性校验规则候选竞品（按行业关键词）──`
- L179-180: `"目标产品"`, `"竞品"` — chat context
- L201: `"你是一位竞品情报分析师。请基于提供的研究上下文简洁地回答问题。"`
- L211-262: 整个 `suggest_competitors` 端点（竞品推荐）

### 6. `cg/main.py` (3 处)

- L27: `"""StaticFiles variant that requires a valid CompeteGraph session cookie."""`
- L85: `title="CompeteGraph API"`
- L87: `description="AI 驱动的可溯源竞品分析 Agent 协作系统"`

### 7. `cg/agents/runtime.py` (1 处)

- L1: `"""Agent runtime primitives for CompeteGraph."""`

### 8. `cg/__init__.py` (1 处)

- L1: `"""CompeteGraph 后端包。"""`

### 9. `cg/tools/local_paper_search.py` (2 处)

- L1: `"""本地论文检索工具，替代 CompeteInsight 的 Web SearchTool。`
- L104: `"""与 CompeteInsight SearchTool 接口兼容的本地论文检索。"""`

### 10. `cg/repositories/run.py` (3 处)

- L52: `target_product=request.target_product`
- L199: `battlecards_data = await read_json(run_dir / "exports" / "battlecards.json", [])`
- L221: `battlecards=[BattlecardItem(**row) for row in battlecards_data]`

### 11. `cg/repositories/evidence.py` (1 处)

- L36: `competitor=evidence.competitor`

---

## 二、Skills YAML 文件

### 1. `skills/research_planning.yaml` (8 处)

- L9: `target product and competitors`
- L19: `competitors`
- L27: `competitors are explicit`
- L30: `Missing competitor aliases`
- L36: `target product, competitors, dimensions`
- L41-42: `target_product: string`, `competitors: string[]`
- L47: `competitors: string[]`
- L52: `target_product is not duplicated in competitors`
- L62: `competitors: string[]`

### 2. `skills/evidence_structuring.yaml` (3 处)

- L10: `Unknown competitor is acceptable`
- L64: `competitor, and confidence metadata`
- L68-69: `Detect competitor`, `Keep unknown competitor`

### 3. `skills/analysis_and_review.yaml` (2 处)

- L21-22: `battlecards.json`, `battlecards.md`

### 4. `skills/report_composer.yaml` (4 处)

- L33: `competitor profiles and matrix`
- L34: `battlecards`
- L42: `battlecards: BattlecardItem[]`
- L53: `competitor, source title`

---

## 三、前端 `frontend/src/App.tsx` (15+ 处)

- L586: `<strong>CompeteInsight</strong>`
- L650-651: `competitor question`, `CompeteInsight plans the search`
- L773: `aria-label="CompeteInsight overview"`
- L776: `<h1>CompeteInsight</h1>`
- L777: `competitor questions`
- L790: `CompeteInsight server`
- L832: `输入目标产品、竞品和研究目标，一次启动完整的竞品研究链路。`
- L852: `生成带引用的竞品报告、矩阵和方法说明`
- L914: `https://github.com/SHYTHU49/CompeteInsight`
- L311-318: `dimensionOptions` 仍是旧的 6 个竞品维度
- L2193-2206: `defaultDraft` 仍是 Trae/Cursor/Copilot 默认值
- L2283-2289: `dimensionLabel` 仍是旧维度映射

---

## 四、统计

| 文件 | 残留数 | 严重程度 |
|------|--------|----------|
| research_agents.py | 20+ | 🔴 高 |
| pipeline.py | 15+ | 🔴 高 |
| schemas/research.py | 20+ | 🔴 高 |
| api/runs.py | 10+ | 🟡 中 |
| App.tsx | 15+ | 🔴 高 |
| skills/*.yaml | 17 | 🟡 中 |
| main.py | 3 | 🟢 低 |
| settings.py | 1 | 🟢 低 |
| runtime.py | 1 | 🟢 低 |
| __init__.py | 1 | 🟢 低 |
| local_paper_search.py | 2 | 🟢 低 |
| repositories/*.py | 4 | 🟡 中 |

**总计: ~110 处残留**
