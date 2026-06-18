# Task Plan: ScholarInsight — 统一执行计划

## Goal
将 CompeteInsight 改造为 ScholarInsight（学术论文推理模式分析）+ 24/7 batch daemon 持续消耗 MiMo 380B tokens。

## 源文档
- `ScholarInsight_implementation_plan.md` — opus 写的代码改造清单
- `02_batch_daemon_instructions.md` — opus 写的 daemon 需求
- `decisions.md` — grilling 确认的 28 条决策

## Current Phase
Phase 0

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
- [ ] 创建 `scripts/build_paper_index.py`
  - PyMuPDF 批量提取 /home/zsz/papers/ 下所有 PDF
  - 提取 title + abstract + 前 5 页文本 (focused_text, 前 8K 字符)
  - 输出 `data/paper_index.json`
  - 幂等：重跑覆盖
  - 每 500 篇打印进度
- [ ] 运行预处理（~1 小时，CPU，可后台）
- [ ] 抽样验证索引质量
- [ ] Git commit + push
- **Status:** pending

## Phase 2: Embedding 索引 — build_embeddings.py
- [ ] 创建 `scripts/build_embeddings.py`
  - 加载 `BAAI/bge-large-en-v1.5` 到 A6000
  - encode 所有论文的 title + abstract
  - 输出 `data/embeddings.npy`（N × 1024 float32）
  - 更新 paper_index.json 加入 embedding 索引位置
- [ ] 运行 embedding（~10 分钟，GPU）
- [ ] 验证：加载 embedding，测试几个 query 的 cosine similarity
- [ ] Git commit + push
- **Status:** pending

## Phase 3: 代码改造 — Schemas & Settings & .env
按 opus 实现计划执行：

### 3a: backend/cg/schemas/research.py
- [ ] `DEFAULT_DIMENSIONS` → 15 种 REASONING_PATTERNS
- [ ] `DIMENSION_LABELS` → 对应中文标签
- [ ] `ResearchRequest` 默认值改为学术语境（target_topic, seed_papers, research_goal）
- [ ] `Evidence` 加 3 字段：reasoning_pattern, bottleneck, mechanism

### 3b: backend/cg/settings.py
- [ ] 加 `mimo_api_key: str = ""`
- [ ] 加 `mimo_base_url: str = "https://token-plan-cn.xiaomimimo.com/v1"`
- [ ] 加 `scholar_paper_index_path: str = "data/paper_index.json"`
- [ ] 加 `scholar_local_papers_dir: str = "/home/zsz/papers/"`
- [ ] `active_llm_api_key` 属性加 `"mimo"` 分支
- [ ] `active_llm_base_url` 属性加 `"mimo"` 分支

### 3c: backend/.env
- [ ] CG_LLM_PROVIDER=mimo
- [ ] CG_LLM_MODEL=mimo-v2.5-pro
- [ ] MIMO_API_KEY=（从 /home/zsz/Mimo/.env 复制）
- [ ] MIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
- [ ] SCHOLAR_PAPER_INDEX_PATH=data/paper_index.json
- [ ] SCHOLAR_LOCAL_PAPERS_DIR=/home/zsz/papers/

- [ ] Git commit + push
- **Status:** pending

## Phase 4: 代码改造 — LLM Client（关键修复）
修改 `backend/cg/llm/client.py`：
- [ ] `complete()` 加 `max_tokens=8192` 参数
- [ ] `content` 为空时 fallback 到 `reasoning_content`
- [ ] 从 reasoning_content 中提取 JSON（正则匹配 `{...}`）
- [ ] 429 重试：sleep 10s，无限重试（替换现有的有限重试）
- [ ] 保持现有 JSON repair 机制不变
- [ ] Git commit + push
- **Status:** pending

## Phase 5: 代码改造 — Local Paper Search
创建 `backend/cg/tools/local_paper_search.py`：
- [ ] `LocalPaperIndex` 类
  - 加载 paper_index.json + embeddings.npy
  - `search(query, max_results)` 方法：sentence-transformers encode query → numpy cosine similarity → top-k
  - 返回 list[SourceCandidate]（接口与 SearchTool 兼容）
- [ ] `LocalPaperSearchTool` 类
  - `__init__(settings)` — 从 settings 读 index 路径
  - `provider_names` / `active_provider_names` 属性
  - `async search(query, max_results)` — 委托给 LocalPaperIndex
- [ ] Git commit + push
- **Status:** pending

## Phase 6: 代码改造 — Pipeline & Agents
### 6a: backend/cg/orchestrator/pipeline.py
- [ ] import `LocalPaperSearchTool`
- [ ] `SearchTool(settings)` → `LocalPaperSearchTool(settings)`

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

### 7b: 验证
- [ ] `uv sync` 安装依赖
- [ ] `uv run uvicorn cg.main:app --reload` 启动后端无报错
- [ ] 手动发起一个 topic run（如 "RAG"），验证 Pipeline 跑通
- [ ] 验证报告包含 reasoning_pattern / bottleneck / mechanism
- [ ] Git commit + push
- **Status:** pending

## Phase 8: Batch Daemon
### 8a: scripts/topics.json
- [ ] 按 02_batch_daemon.md 的 6 大类生成 500+ 研究话题
  - 知识图谱与推理（最高优先级）
  - NLP / 大语言模型
  - 多模态
  - 模型效率与部署
  - 端侧 AI / 嵌入式智能
  - 学习范式
  - 评测与安全
- [ ] 每个话题可组合变体

### 8b: scripts/batch_daemon.py
- [ ] 自动从 topics.json 读取话题
- [ ] 构造 ResearchRequest，直接 import 调用 ResearchPipeline.run()
- [ ] asyncio.Semaphore(3) 并发控制
- [ ] batch_progress.json 断点续传
- [ ] 失败话题记录到 batch_errors.json
- [ ] 每 10 个 run 打印统计（已完成/总数/预估剩余时间/累计 token）
- [ ] MiMo 429 重试：sleep 10s，无限重试
- [ ] LLM 调用全部走 MiMo（红队走 Gemini）

### 8c: 联调 + 启动
- [ ] 前台运行测试 2-3 个 topic
- [ ] 验证 batch_progress.json 正确记录
- [ ] 验证 batch_errors.json 错误记录
- [ ] nohup 后台启动 daemon
- [ ] Git commit + push
- **Status:** pending

## Phase 9: 收尾
- [ ] 全量 git push
- [ ] 更新 progress.md 最终状态
- [ ] 输出验收报告
- **Status:** pending

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
