# ScholarInsight 交接报告

日期：2026-06-21

## 当前一句话状态

ScholarInsight 已经从 CompeteInsight 改造成一个本地论文库驱动的、source-role aware、claim-gated、falsification-aware 的 research ideation pipeline。当前最强结论是：

> 系统可以稳定生成可审计、可证伪、适合导师检查的 research-ideation artifacts；但还不能声称自动生成 novel/useful/publishable ideas。

## 当前代码与文档状态

主分支：`main`

当前核心文档：

- `README.md`：项目入口、架构图、运行方式。
- `docs/advisor_idea_brief.md`：给导师看的内部 idea brief。
- `docs/project_handoff_2026-06-21.md`：本交接报告。
- `docs/embedding_granularity_mineru_plan.md`：embedding 粒度分析与 MinerU 改造建议。
- `docs/paper/aaai_draft.md`：内部 AAAI-style draft，不可直接投稿。
- `docs/paper/references.bib`：第一批 reference seed。

当前核心代码：

- `backend/cg/tools/local_paper_search.py`
  - 本地 paper index 检索。
  - paper-level embedding recall。
  - lexical score。
  - CUDA reranker。
  - topic-family source-role gate。
- `backend/cg/orchestrator/pipeline.py`
  - deterministic evidence extraction。
  - evidence clusters。
  - claim gate `g(c)`。
  - hard-negative audit。
  - falsification plan。
  - report rendering。
- `scripts/render_fresh_pilot_artifacts.py`
  - fresh CUDA deterministic pilot artifact generator。
- `scripts/pilot_freeze_validator.py`
  - freeze validator。
  - mentor review packet generator。
  - 现在会显式提示 generic synthesis claims 需要导师 novelty/usefulness review。
- `scripts/idea_quality_evaluator.py`
  - 结构化 idea quality evaluator。
  - 2026-06-21 已加入 generic synthesis claim penalty。
- `scripts/ablation_quality_table.py`
  - artifact-masked structural ablation。
- `scripts/runtime_ablation_table.py`
  - runtime source-stage ablation。
  - 现在支持 `--existing-variant label=path`，可以重读已有 variant artifacts 而不重跑检索。
- `scripts/paper_experiment_packet.py`
  - 生成 paper experiment packet。
- `scripts/paper_manuscript_draft.py`
  - 从 experiment packet 生成 `docs/paper/aaai_draft.md`。

## 当前质量结论

### Fresh pilot

Fresh pilot artifact root：

`data/quality_audits/fresh_pilot_004_006_010_011_012_17a/`

覆盖 topic：

- 004 RAG with Knowledge Graphs
- 006 Mathematical Reasoning
- 010 Causal Reasoning with LLMs
- 011 Counterfactual Inference
- 012 Multi-hop Reasoning on Graphs

当前保守 evaluator 结果：

| Topic | Score | Report-ready | Falsification plans | Warning |
|---|---:|---:|---:|---|
| 004 RAG with Knowledge Graphs | 0.936 | 4 | 4 | generic synthesis needs advisor review |
| 006 Mathematical Reasoning | 0.883 | 3 | 3 | generic synthesis needs advisor review |
| 010 Causal Reasoning with LLMs | 0.919 | 1 | 1 | generic synthesis needs advisor review |
| 011 Counterfactual Inference | 0.936 | 4 | 4 | generic synthesis needs advisor review |
| 012 Multi-hop Reasoning on Graphs | 0.920 | 2 | 2 | generic synthesis needs advisor review |

解释：

- 5/5 artifact 仍然 freeze PASS。
- 14 条 report-ready claims。
- 14/14 report-ready claims 有 falsification plan。
- CUDA reranker provenance 存在。
- external LLM calls 为 false。
- 但是 5/5 topic 都被标记为 generic synthesis needs advisor review。

这个 warning 是有意加入的保守化判断。它说明当前系统在“审计性、可证伪性”上成立，但 report-ready claim 很多仍是证据轴综述，不应被包装成已验证的新 idea。

### Structural ablation

文件：

`data/quality_audits/fresh_pilot_004_006_010_011_012_17a/ablation_quality_table.md`

保守 evaluator 下的 mean delta：

- `minus_hard_negative`: -0.125
- `minus_claim_gate`: -0.115
- `minus_falsification`: -0.091
- `minus_source_roles`: -0.083
- `minus_experiment_framing`: -0.042

解释：

- hard-negative audit 和 claim gate 是当前最关键两个模块。
- source-role 和 falsification 仍有独立贡献。
- 这是 artifact-masked structural ablation，不是完整 runtime rerun。

### Runtime source-stage ablation

文件：

`data/quality_audits/runtime_ablation_004_006_010_011_012_17b/runtime_ablation_table.md`

结论：

- `no_reranker`：mean score delta -0.011，mean report-ready delta -0.400。
- `no_source_gate`：mean score delta -0.100，并触发 missing counterexample audit。
- `no_reranker_no_source_gate`：mean score delta -0.093，既有 source role 风险，也有 counterexample audit 风险。

解释：

- reranker 不一定每题大幅涨分，但会影响 report-ready synthesis 和 source purity。
- source gate 不只是过滤 source，还保留 rejected/boundary pool，使 hard-negative audit 能成立。

## 当前最大短板

当前短板已经从“检索会乱、claim 会乱写”转向更上游的问题：

> evidence source granularity 不够细。

现在主 embedding 是 paper-level `title + abstract`，不是 sentence/chunk-level。`focused_text` 只来自 PDF 前 5 页的前 8,000 字符。reranker 和 evidence extraction 会用 `focused_text`，但主召回不是全文/section/chunk 级。

直接后果：

- 能找到相关 paper，但不一定找到 paper 内最有价值的 idea evidence。
- method / experiment / limitation / future work 等 section 信号容易缺失。
- report-ready claim 容易变成“多篇论文共同围绕某分析轴”的 generic synthesis。

详细分析见：

`docs/embedding_granularity_mineru_plan.md`

## 暂停事项

当前不建议做：

- 不建议立刻跑 132-topic full batch。
- 不建议继续堆更多 if/else gate。
- 不建议把 `aaai_draft.md` 当投稿稿。
- 不建议让外部 LLM 直接读取本地论文内容，除非明确批准数据外发。

## 一周后建议恢复顺序

1. 先读本文件。
2. 读 `docs/advisor_idea_brief.md`。
3. 读 `docs/embedding_granularity_mineru_plan.md`。
4. 抽查 mentor packets：
   - `data/quality_audits/fresh_pilot_004_006_010_011_012_17a/004/mentor_review_packet.md`
   - `data/quality_audits/fresh_pilot_004_006_010_011_012_17a/010/mentor_review_packet.md`
   - `data/quality_audits/fresh_pilot_004_006_010_011_012_17a/011/mentor_review_packet.md`
5. 再决定下一步。

推荐下一步不是 132-topic，而是：

1. 选 004 RAG+KG 的 20-40 篇 accepted/rejected papers。
2. 用 MinerU 转 Markdown。
3. 做 section/chunk-level embedding。
4. 从 method / experiment / limitation / future_work 中提取 structured idea evidence。
5. 接回现有 `g(c)` / `h(c)`。
6. 对比 generic synthesis warning 是否减少、导师是否更容易判断 novelty/usefulness。

## 常用命令

后端测试：

```bash
backend/.venv/bin/python -m pytest backend
```

刷新 fresh pilot validator 和 mentor packets：

```bash
backend/.venv/bin/python scripts/pilot_freeze_validator.py data/quality_audits/fresh_pilot_004_006_010_011_012_17a
```

刷新 structural ablation：

```bash
backend/.venv/bin/python scripts/ablation_quality_table.py data/quality_audits/fresh_pilot_004_006_010_011_012_17a --output-dir data/quality_audits/fresh_pilot_004_006_010_011_012_17a
```

重读已有 runtime ablation variants：

```bash
backend/.venv/bin/python scripts/runtime_ablation_table.py \
  --baseline-root data/quality_audits/fresh_pilot_004_006_010_011_012_17a \
  --variants "" \
  --existing-variant no_reranker=data/quality_audits/runtime_ablation_004_006_010_011_012_17b/no_reranker \
  --existing-variant no_source_gate=data/quality_audits/runtime_ablation_004_006_010_011_012_17b_cuda_source_gate/no_source_gate \
  --existing-variant no_reranker_no_source_gate=data/quality_audits/runtime_ablation_004_006_010_011_012_17b/no_reranker_no_source_gate \
  --output-dir data/quality_audits/runtime_ablation_004_006_010_011_012_17b
```

刷新 paper experiment packet：

```bash
backend/.venv/bin/python scripts/paper_experiment_packet.py \
  --fresh-root data/quality_audits/fresh_pilot_004_006_010_011_012_17a \
  --structural-ablation-json data/quality_audits/fresh_pilot_004_006_010_011_012_17a/ablation_quality_table.json \
  --runtime-ablation-json data/quality_audits/runtime_ablation_004_006_010_011_012_17b/runtime_ablation_table.json \
  --output-dir data/quality_audits/paper_experiment_packet_17c
```

刷新 AAAI-style draft：

```bash
backend/.venv/bin/python scripts/paper_manuscript_draft.py \
  --packet-json data/quality_audits/paper_experiment_packet_17c/paper_experiment_packet.json \
  --output docs/paper/aaai_draft.md
```

安全扫描：

```bash
rg -n "<absolute-path-or-mirror-or-overclaim-patterns>" README.md backend/.env.example backend/cg backend/tests scripts docs
```

## 判断边界

可以说：

- ScholarInsight 生成的是 advisor-reviewable artifact。
- `g(c)` 和 `h(c)` 提供 auditability/falsifiability certificate。
- 当前 pilot 证明 pipeline 在 5 个 topic family 上能稳定产出可审计 artifact。

不要说：

- 系统已经能自动生成 AAAI-quality ideas。
- `h(c)` 证明 claim 为真。
- 当前 report-ready claim 已经证明 novelty/usefulness。
- 当前 5-topic pilot 足以证明 132-topic 泛化。

## 当前下一步决策点

一周后需要决定：

1. 是否先做 MinerU + chunk index pilot。
2. 是否允许用 GPT-5.5 / Gemini 做 blind strong-model novelty/usefulness review。
3. 是否让导师先人工看 004/010/011 三个 mentor packets。
4. 如果导师认可 artifact 形式，再决定是否跑 132-topic。
