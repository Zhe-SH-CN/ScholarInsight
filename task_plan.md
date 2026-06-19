# Task Plan: ScholarInsight — 统一执行计划

## Goal
将 CompeteInsight 改造为 ScholarInsight（学术论文推理模式分析）+ 24/7 batch daemon 持续消耗 MiMo 380B tokens。

## Current Phase
后端完成，前端 App.tsx 改造 + batch daemon 重启

---

## Phase 0-2: 数据准备 ✅
- [x] Phase 0: 环境准备 + 工作区清理
- [x] Phase 1: PDF 预处理（35,031/35,039 成功，99.98%）
- [x] Phase 2: Embedding 索引（35,031 × 1024，GPU 加速）

## Phase 3-8: 后端代码改造 ✅
- [x] Phase 3: Schemas（15 种推理模式）+ Settings（mimo + gemini provider）+ .env
- [x] Phase 4: LLM Client（max_tokens=8192, reasoning_content, 429 无限重试）
- [x] Phase 5: LocalPaperSearchTool（embedding 检索 + text fallback）
- [x] Phase 6: Pipeline + 5 个 Agent system prompt 改造
- [x] Phase 7: Skills YAML 更新 + uvicorn 验证
- [x] Phase 8: Batch Daemon 代码（topics.json 132 话题 + batch_daemon.py）
- [x] Gemini 红队审查：MiMo 主分析 + Gemini 红队（模型多样性）

## Phase 9: 后端残留清理 + 环境重建 ✅
- [x] 全面清理 CompeteInsight 残留 (~110 处)
- [x] schemas/research.py: CompetitorMatrix→PaperPatternMatrix, BattlecardItem→删除
- [x] pipeline.py: build_competitor_matrix→build_paper_pattern_matrix, 删除 battlecards
- [x] research_agents.py: competitor→paper, target_product→target_topic
- [x] api/runs.py: 删除 suggest_competitors 端点
- [x] settings.py: user_agent→ScholarInsightBot
- [x] skills/*.yaml: competitor→paper, battlecards→删除
- [x] main.py/runtime.py/__init__.py: 品牌名更新
- **Status:** complete

## Phase 10: 前端 App.tsx 改造 ❌ 待执行
- [ ] 10a: 品牌名 CompeteInsight → ScholarInsight
- [ ] 10b: dimensionOptions 改为 15 种推理模式（中文标签）
- [ ] 10c: defaultDraft 改为学术语境默认值
- [ ] 10d: IntroCarousel 内容改为论文分析流程
- [ ] 10e: EvidenceView 加 reasoning_pattern / bottleneck / mechanism 显示
- [ ] 10f: BriefView 维度显示改为推理模式
- [ ] 10g: ResearchComposer 表单标签改为学术语境
- [ ] 10h: 前端构建验证

## Phase 11: Batch Daemon 重启 ❌ 待执行
- [ ] 11a: 清理旧 runs（基于旧 index 的）
- [ ] 11b: 重启 batch daemon（新 index + 新 embedding）
- [ ] 11c: 监控直到全部 132 话题完成

## Phase 12: 最终收尾 ❌ 待执行
- [ ] 12a: Git commit + push 所有改动
- [ ] 12b: 更新 progress.md / findings.md
- [ ] 12c: 输出验收报告

---

## 验收项
| 检查项 | 当前状态 | 预期结果 |
|--------|----------|----------|
| paper_index.json | ✅ 35,031 成功 | 包含所有 PDF |
| embeddings.npy | ✅ 35,031 × 1024 | GPU 生成 |
| 后端代码 | ✅ 全部改造完成 | 5 Agent 论文分析 |
| 前端 UI | ❌ 仍是竞品分析 UI | 改为学术论文分析 |
| batch daemon | ⏸ 已停止 | 24/7 运行 |

## Errors Encountered
| Error | Resolution |
|-------|------------|
| PyTorch cu130 vs driver cu128 | 降级 cu128 + Python 3.10 |
| torchvision 版本不匹配 | torch 2.11 + torchvision 0.26 共存 |
| normalize_url 不支持本地路径 | 添加 `/` 开头判断 |
| macOS ._{'`'} 文件污染 index | 清理后重跑 |
