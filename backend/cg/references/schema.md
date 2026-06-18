# CognoGraph Schema

CognoGraph stores one graph at `.cognograph/graph.json`.

## Top-Level Graph

```json
{
  "metadata": {
    "name": "CognoGraph",
    "version": "0.1.0",
    "created_at": "YYYY-MM-DD_HH_MM",
    "updated_at": "YYYY-MM-DD_HH_MM",
    "description": "user-defined description",
    "node_count": 0,
    "edge_count": 0,
    "topic": "research field"
  },
  "schema": {
    "reasoning_patterns": ["pattern_id"],
    "reasoning_pattern_descriptions": {"pattern_id": "description"}
  },
  "nodes": [],
  "edges": []
}
```

Use `.agents/skills/cognograph/scripts/utils.py` as the executable source of truth for constants and graph I/O.

## Node

One paper is one node.

```json
{
  "id": "paper_0000",
  "title": "paper title",
  "authors": ["author 1", "author 2"],
  "year": 2024,
  "venue": "NeurIPS",
  "abstract": "abstract text",
  "methods": [
    {
      "name": "method acronym",
      "full_name": "method full name",
      "description": "method summary"
    }
  ],
  "pdf_path": "/path/to/user/papers/paper.pdf",
  "source": "user | seed",
  "added_at": "YYYY-MM-DD_HH_MM"
}
```

## Edge

Edges point from successor paper to predecessor paper.

```json
{
  "id": "edge_0000",
  "source": "paper_0001",
  "target": "paper_0000",
  "reasoning_pattern": "gap_driven_reframing",
  "bottleneck": "specific technical bottleneck",
  "mechanism": "specific technical mechanism",
  "evidence": "brief evidence",
  "confidence": 0.85
}
```

## Reasoning Patterns

| Pattern ID | Name | Description |
| --- | --- | --- |
| `gap_driven_reframing` | 痛点驱动重构 | 将失败/局限转化为显式设计约束 |
| `cross_domain_synthesis` | 跨领域综合 | 从邻近领域移植方案并添加兼容层 |
| `representation_shift` | 表征转换 | 替换基础表征原语 |
| `modular_pipeline_composition` | 模块化管线 | 分解为可组合模块 |
| `data_evaluation_engineering` | 数据评估工程 | 新数据集/基准/指标 |
| `principled_probabilistic_modeling` | 概率建模 | 概率图模型、贝叶斯推理 |
| `formal_experimental_tightening` | 理论实验迭代 | 证明/界限与实验验证迭代 |
| `approximation_engineering` | 近似工程 | 有原则的近似（量化、低秩） |
| `inference_time_control` | 推理时控制 | 运行时引导采样 |
| `structural_inductive_bias` | 结构归纳偏置 | 硬编码领域结构 |
| `multiscale_hierarchical_modeling` | 多尺度分层 | 多层级粒度交互 |
| `mechanistic_decomposition` | 机制分解 | 分解为可解释机制 |
| `adversary_modeling` | 对抗建模 | 建模对手，设计防御 |
| `numerics_systems_codesign` | 数值系统协同 | 算法+硬件协同设计 |
| `data_centric_optimization` | 数据中心优化 | 优化数据分布 |

## Validation Rules

- Paper IDs must use `paper_XXXX`.
- Edge IDs must use `edge_XXXX`.
- `source` on an edge is the newer or innovating paper.
- `target` on an edge is the predecessor or paper being improved.
- `reasoning_pattern` must be one of the 15 pattern IDs.
- `confidence` must be between 0.0 and 1.0.
- Titles are de-duplicated with case-folding and punctuation normalization.
