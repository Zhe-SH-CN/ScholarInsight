# Task Plan: ScholarInsight — 统一执行计划

## Goal
将 CompeteInsight 改造为 ScholarInsight（学术论文推理模式分析）+ 24/7 batch daemon 持续消耗 MiMo 380B tokens。

## Current Phase
环境重建 (Python 3.10 + torch cu128) + Embedding 重建

---

## Phase 0: 环境准备 + 工作区清理 ✅
- [x] 归档 PaperLens / Fork CompeteInsight / git remote / decisions.md
- [x] .env 配置 / CognoGraph 资源复制
- **Status:** complete

## Phase 1: PDF 预处理 — build_paper_index.py ✅
- [x] 创建 `scripts/build_paper_index.py`（支持断点续传）
- [x] 全量运行：54,430 PDF → 35,031 成功 / 19,399 失败
- [x] 输出 `data/paper_index.json` (333MB)
- **Status:** complete

## Phase 2: Embedding 索引 ❌ 需重建
- [x] 创建 `scripts/build_embeddings.py`
- [ ] **重建 embedding**（旧 27,725 与新 index 54,430 不兼容）
- [ ] GPU 加速（需 Python 3.10 + torch cu128）
- **Status:** pending — 等环境重建后执行

## Phase 3: 代码改造 — Schemas & Settings & .env ✅
- [x] 15 种推理模式 / Evidence 新字段 / mimo provider / .env
- **Status:** complete

## Phase 4: LLM Client 修复 ✅
- [x] max_tokens=8192 / reasoning_content fallback / 429 无限重试
- **Status:** complete

## Phase 5: Local Paper Search ✅
- [x] LocalPaperSearchTool（embedding 检索 + text fallback）
- [x] normalize_url 修复（支持本地文件路径）
- **Status:** complete

## Phase 6: Pipeline & Agents ✅
- [x] LocalPaperSearchTool 替换 SearchTool
- [x] 5 个 Agent system prompt 从竞品→论文分析
- **Status:** complete

## Phase 7: Skills & 验证 ✅
- [x] 5 个 YAML mission 更新
- [x] uvicorn 启动验证通过
- **Status:** complete

## Phase 8: Batch Daemon ✅
- [x] topics.json (132 话题)
- [x] batch_daemon.py (Semaphore(3), 断点续传)
- [x] 已完成 45/132 话题（旧 index），需用新 index 重跑剩余 87 个
- **Status:** complete（代码），待重启

## Phase 9: 环境重建 + 收尾
- [ ] uv 环境重建为 Python 3.10
- [ ] 安装 torch cu128（GPU 支持）
- [ ] 重建 embedding（35,031 篇 × 1024，GPU ~10 分钟）
- [ ] 重启 batch daemon（新 index + 新 embedding）
- [ ] Git commit + push
- **Status:** pending

---

## 验收项
| 检查项 | 当前状态 | 预期结果 |
|--------|----------|----------|
| paper_index.json | ✅ 35,031/54,430 成功 | 包含所有 PDF |
| embeddings.npy | ❌ 旧 27,725 与 index 不兼容 | 35,031 × 1024 |
| GPU 支持 | ❌ PyTorch CUDA 版本不匹配 | torch cu128 + A6000 |
| uv 环境 | ❌ Python 3.13 + torch 版本冲突 | Python 3.10 + torch cu128 |
| batch daemon | ⏸ 已停止 | 24/7 运行 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| PyTorch cu130 vs driver cu128 | 多次 | 降级到 cu128，但 torchvision 版本不匹配 |
| torchvision 0.26 需要 torch 2.12 | 2 | cu128 index 只有 torch 2.11，无解 |
| normalize_url 不支持本地路径 | 1 | 添加 `/` 开头路径判断 |
| paper_index 路径重复 (data/data/) | 1 | 改用绝对路径 |
