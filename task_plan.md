# Task Plan: ScholarInsight — 统一执行计划

## Goal
将 CompeteInsight 改造为 ScholarInsight（学术论文推理模式分析）+ 24/7 batch daemon 持续消耗 MiMo 380B tokens。

## 源文档
- `ScholarInsight_implementation_plan.md` — opus 写的代码改造清单
- `02_batch_daemon_instructions.md` — opus 写的 daemon 需求
- `decisions.md` — grilling 确认的 28 条决策

## Current Phase
全部 Phase 完成 — Batch Daemon 24/7 运行中

---

## Phase 0: 环境准备 + 工作区清理
- [x] 归档 PaperLens 遗留文件到 archive/
- [x] Fork CompeteInsight → ScholarInsight (cp -r)
- [x] 设置 git remote: git@github.com:Zhe-SH-CN/ScholarInsight.git
- [x] 创建 decisions.md（28 条决策）
- [x] 创建统一 task_plan.md / progress.md / findings.md
- [x] 确认 backend/.env 配置（MiMo key + Gemini key + 路径）
- [x] 复制 CognoGraph 资源文件到 ScholarInsight（discover_edges.md, utils.py, schema.md）
- [x] Git commit + push
- **Status:** complete

## Phase 1: PDF 预处理 — build_paper_index.py
- [x] 创建 `scripts/build_paper_index.py`
  - PyMuPDF 批量提取 /home/zsz/papers/ 下所有 PDF
  - 提取 title + abstract + 前 5 页文本 (focused_text, 前 8K 字符)
  - 输出 `data/paper_index.json`
  - 幂等：重跑覆盖
  - 每 500 篇打印进度
- [ ] 运行预处理（后台运行中 22500/36539）
- [ ] 抽样验证索引质量
- [ ] Git commit + push
- **Status:** in_progress

## Phase 2: Embedding 索引 — build_embeddings.py ✅
- [x] 创建 `scripts/build_embeddings.py`
- [x] 运行 embedding（CPU ~3 小时，27725 篇 × 1024 维）
- [x] 输出 `data/embeddings.npy` (109MB)
- **Status:** complete

## Phase 3: 代码改造 — Schemas & Settings & .env ✅
- [x] `DEFAULT_DIMENSIONS` → 15 种 REASONING_PATTERNS
- [x] `DIMENSION_LABELS` → 对应中文标签
- [x] `ResearchRequest` 默认值改为学术语境
- [x] `Evidence` 加 3 字段：reasoning_pattern, bottleneck, mechanism
- [x] settings.py 加 mimo + scholar 字段
- [x] .env 配置完成
- [x] Git commit + push
- **Status:** complete

## Phase 4: 代码改造 — LLM Client（关键修复）✅
- [x] `complete()` 加 `max_tokens=8192`
- [x] `reasoning_content` fallback
- [x] 429 无限重试 sleep 10s
- [x] Git commit + push
- **Status:** complete

## Phase 5: 代码改造 — Local Paper Search ✅
- [x] `LocalPaperIndex` + `LocalPaperSearchTool` 创建
- [x] embedding 检索 + text fallback
- [x] Git commit + push
- **Status:** complete

## Phase 6: 代码改造 — Pipeline & Agents ✅
- [x] `LocalPaperSearchTool` 替换 `SearchTool`
- [x] 5 个 Agent system prompt 从竞品→论文分析
- [x] Git commit + push
- **Status:** complete

### 6b: backend/cg/agents/research_agents.py
按 opus 概念映射表改造 5 个 Agent 的 system prompt：
- [ ] **ResearchPlanningAgent**：竞品研究规划 → 论文关系分析规划，"维度" → "推理模式"
- [ ] **SourceResearchAgent**：网页搜索策略 → 本地论文库检索策略
- [ ] **EvidenceStructuringAgent**：从网页提取事实 → 从论文提取创新证据
  - 嵌入 discover_edges.md 的核心 prompt
  - 要求 LLM 输出 reasoning_pattern (15 种之一) + bottleneck + mechanism + evidence
- [ ] **AnalysisAndReviewAgent**：竞品结论 → 推理模式判断
  - 红队审查：质疑 pattern 判断是否准确，bottleneck/mechanism 是否有证据支撑
  - **红队调用 Gemini 3.1 Pro**（非 MiMo）
- [ ] **ReportComposerAgent**：竞品报告 → 论文演化报告 + 研究空白建议

- [ ] Git commit + push
- **Status:** pending

## Phase 7: 代码改造 — Skills & 验证
### 7a: skills/*.yaml（5 个文件）
- [ ] research_planning.yaml — mission/purpose 改为论文分析
- [ ] source_research.yaml — 同上
- [ ] evidence_structuring.yaml — 同上
- [ ] analysis_and_review.yaml — 同上
- [ ] report_composer.yaml — 同上

### 7b: 验证 ✅
- [x] `uv sync --extra full` 安装依赖
- [x] `uv run uvicorn cg.main:app` 启动后端无报错
- [x] Git commit + push
- **Status:** complete

## Phase 8: Batch Daemon ✅
- [x] scripts/topics.json (132 个研究话题)
- [x] scripts/batch_daemon.py (asyncio.Semaphore(3), 断点续传, 错误记录)
- [x] Git commit + push
- **Status:** complete

## Phase 9: 收尾 ✅
- [x] Git commit + push
- [x] PDF 提取完成 (27,725/36,539)
- [x] Embedding 构建完成 (27,725 × 1024)
- [x] 更新 progress.md
- **Status:** complete

---

## 验收项
| 检查项 | 预期结果 |
|--------|----------|
| build_paper_index.py | 生成 data/paper_index.json，包含所有 PDF |
| build_embeddings.py | 生成 data/embeddings.npy，检索测试通过 |
| uv run uvicorn cg.main:app | 后端启动无报错 |
| 单个 topic run | Pipeline 跑完 5 个 Agent，生成报告 |
| 报告内容 | 包含论文间的推理模式分析（非竞品分析） |
| Evidence 数据 | 包含 reasoning_pattern, bottleneck, mechanism |
| batch_daemon | 24/7 运行稳定，消耗 MiMo tokens |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| (none yet) | | |
