# Findings & Decisions

## Requirements
- 将 CompeteInsight 改造为 ScholarInsight（学术论文推理模式分析）
- 24/7 batch daemon 消耗 MiMo 380B tokens
- 本地 30K PDF 库作为数据源
- embedding 检索替代 Tavily/Exa
- MiMo 做主分析，Gemini 做红队审查

## Research Findings
- CompeteInsight 代码未改造，全部原始竞品分析逻辑
- LLMClient.complete() 不兼容 MiMo thinking model（content 为空 + 无 max_tokens）
- CompeteInsight 已有 JSON repair 机制（complete_json 重试）
- 30K PDF × 1024 维 embedding ≈ 120MB，numpy 存储即可
- 论文仍在上传中，先建 index 后续重建

## 执行原则
**最大化利用 opus 的代码。** opus 在 `ScholarInsight_implementation_plan.md` 和 `02_batch_daemon_instructions.md` 中已经写好了：
- 完整的 `build_paper_index.py` 代码（~80 行）
- 完整的 `local_paper_search.py` 代码（~200 行，SequenceMatcher 版本）
- `schemas/research.py` 的 diff（替换 DEFAULT_DIMENSIONS、DIMENSION_LABELS、ResearchRequest、Evidence）
- `settings.py` 的 diff（加 mimo 字段）
- `pipeline.py` 的 diff（替换 SearchTool）
- `.env` 模板
- 5 个 Agent prompt 改造的具体指令
- batch_daemon 的完整需求 + topics 种子列表

**原则：opus 写了什么就直接用什么，不自己重写。** 只有 opus 没给完整代码的部分才自己写。

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Fork 到 ScholarInsight/ | 保留原始 CompeteInsight，Git 历史清晰 |
| bge-large-en-v1.5 | 335M 模型，A6000 48GB 随便放，检索质量好 |
| numpy .npy 存 embedding | 30K 规模暴力搜索 < 10ms，零额外依赖 |
| MiMo max_tokens=8192 | thinking model 需要大 token 空间 |
| Gemini 做红队 | 模型多样性，捕获 MiMo 盲点 |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| LLMClient 不兼容 MiMo thinking | 已修复：reasoning_content fallback + max_tokens=8192 |
| PyTorch cu130 vs driver cu128 | 降级到 cu128 + Python 3.10 |
| torchvision 版本不匹配 | torch 2.11 + torchvision 0.26 共存 |
| normalize_url 不支持本地路径 | 添加 `/` 开头路径判断 |
| macOS ._{'`'} 文件污染 index | 清理后重跑 |
| 红队未使用 Gemini | 已实现：独立 Gemini LLMClient，pipeline.red_team() 使用 |

## Resources
- CompeteInsight 源码: /home/zsz/Mimo/CompeteInsight/
- ScholarInsight 工作目录: /home/zsz/Mimo/ScholarInsight/
- CognoGraph patterns: /home/zsz/Mimo/CognoGraph/.agents/skills/cognograph/scripts/utils.py
- Papers: /home/zsz/papers/ (接近 30K PDF)
- 实现计划: /home/zsz/Mimo/ScholarInsight_implementation_plan.md
