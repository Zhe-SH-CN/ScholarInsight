# Progress Log

## Session: 2026-06-19

### Phase 0: 环境准备 + 工作区清理 ✅
- **Status:** complete
- Actions: 归档PaperLens, Fork CompeteInsight, 设置git remote, 创建decisions.md, 统一计划

### Phase 1: PDF 预处理 ✅
- **Status:** complete
- 36,539 PDF 扫描, 27,725 成功提取
- 输出: backend/data/paper_index.json (272MB)

### Phase 2: Embedding 索引 ✅
- **Status:** complete
- bge-large-en-v1.5 (CPU, ~3 小时)
- 输出: data/embeddings.npy (27,725 × 1024, 109MB)

### Phase 3: Schemas & Settings & .env ✅
- **Status:** complete
- 15种推理模式, Evidence新字段, mimo provider

### Phase 4: LLM Client 修复 ✅
- **Status:** complete
- max_tokens=8192, reasoning_content fallback, 429无限重试

### Phase 5: Local Paper Search ✅
- **Status:** complete
- embedding检索 + text fallback

### Phase 6: Pipeline & Agents ✅
- **Status:** complete
- LocalPaperSearchTool替换SearchTool, 5个Agent prompt改造

### Phase 7: Skills & 验证 ✅
- **Status:** complete
- 5个skills YAML更新, uvicorn启动验证通过

### Phase 8: Batch Daemon ✅
- **Status:** complete
- topics.json (132话题), batch_daemon.py

### Phase 9: 收尾 ✅
- **Status:** complete
- normalize_url 修复 (本地文件路径支持)
- Batch Daemon 启动运行中

## 当前状态
- Batch Daemon 在 tmux session `scholar-batch` 中运行
- 3 个 topic 并行处理中 (Semaphore=3)
- 每个 topic: 39-50 sources, 10+ evidence
- 132 个话题待处理

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | 全部 Phase 完成, Daemon 运行中 |
| Where am I going? | Daemon 24/7 运行, 消耗 MiMo tokens |
| What's goal? | 论文推理模式分析 + 24/7烧MiMo tokens |
| What have I learned? | normalize_url需要处理本地路径, PyTorch CUDA版本匹配 |
| What have I done? | ScholarInsight完整改造, 36K PDF处理, daemon启动 |
