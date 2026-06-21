# Embedding 粒度与 MinerU 改造判断

日期：2026-06-21

## 结论

当前 ScholarInsight 的主检索 embedding 不是句子级，也不是段落/chunk 级，而是**每篇论文一条向量**。

具体来源：

- `scripts/build_paper_index.py` 用 PyMuPDF 读取 PDF 前 5 页，保存：
  - `title`
  - `abstract`
  - `focused_text`：前 8,000 字符
  - `pdf_path`
- `scripts/build_embeddings.py` 对每篇 valid paper 构造：
  - `text = title + " " + abstract`
  - 模型：`BAAI/bge-large-en-v1.5`
  - 输出：`backend/data/embeddings.npy`
- 当前本地数据：
  - `paper_index.json` 条目数：54,430
  - valid papers：35,031
  - embeddings shape：`(35031, 1024)`

因此，第一阶段召回的语义粒度是**论文级 title+abstract embedding**。它适合粗召回 paper，但不适合定位论文内部的 method、experiment、limitation、future work、ablation、threats 等细粒度 idea evidence。

## 当前 pipeline 实际怎么用文本

当前系统不是完全只看 title+abstract：

1. Embedding 检索：
   - 用 query embedding 和每篇论文的 `title+abstract` embedding 做相似度。
2. Lexical score：
   - 使用 `title + abstract + focused_text[:3000]`。
3. Reranker：
   - 使用 `title + abstract + focused_text[:1800]`。
4. Evidence extraction：
   - SourceCandidate 的 `content` 是 `focused_text`。
   - deterministic evidence 从 `document.content` 切句并按 dimension keywords 排序。
   - LLM evidence extraction 最多看 `document.content[:8000]`。

这说明当前短板不是“完全没有全文”，而是：

- 主 embedding 没有 chunk-level recall。
- `focused_text` 只来自 PDF 前 5 页和前 8,000 字符。
- 很多论文的真正 idea 信号在 method / experiments / limitations / conclusion，可能不在 abstract 和前几页。
- 当前 claim 往往退化为 generic evidence-axis synthesis，例如“多篇论文共同围绕 modular system pipeline 组织证据”，说明 evidence 粒度不足以支撑更具体的新 idea。

## 是否应该用 MinerU 转 Markdown

我的判断：**应该做，但不要直接把 MinerU 输出丢给模型生成 idea。**

更稳的方案是把 MinerU 作为“文档结构化层”，然后做分层索引和 evidence extraction：

1. PDF -> MinerU Markdown
2. Markdown section parser
3. section/chunk-level embedding
4. section-role classifier
5. idea-evidence extractor
6. claim gate `g(c)` 和 falsification certificate `h(c)`

也就是说，MinerU 解决的是输入源头质量和结构粒度，不应该替代当前的 source-role gate、claim gate、hard-negative audit 和 falsification plan。

## 推荐的新索引粒度

保留当前 paper-level index，但新增 chunk-level index。

### Paper-level index

用途：

- 快速召回候选论文。
- source-role classifier。
- 去重、paper-level provenance。

字段：

- `paper_id`
- `title`
- `abstract`
- `pdf_path`
- `md_path`
- `venue/year`，如果能解析到
- `paper_embedding`

### Section/chunk-level index

用途：

- 精确召回 evidence。
- 找到 method、experiment、limitation、future work、negative result。
- 支撑更具体的 claim，而不是 generic synthesis。

推荐字段：

- `chunk_id`
- `paper_id`
- `section_path`
- `section_role`
- `chunk_text`
- `chunk_embedding`
- `token_count`
- `page_span`，如果 MinerU 能给
- `figure_or_table_refs`

推荐 section roles：

- `abstract`
- `introduction`
- `related_work`
- `method`
- `experiment`
- `ablation`
- `limitation`
- `discussion`
- `conclusion`
- `future_work`
- `appendix`

## 是否让模型先从 Markdown 提取 idea

可以，但应该分两层：

### 不建议

不要让模型直接读完整 Markdown 后输出“这个方向有什么 idea”。这会重新引入：

- hallucinated synthesis
- source drift
- single-paper overclaim
- 难以审计的长上下文压缩

### 建议

让模型只做受约束的 paper-local extraction：

对每篇论文或每个 high-value section，提取：

- `problem`
- `gap`
- `method_mechanism`
- `assumption`
- `benchmark`
- `failure_mode`
- `limitation`
- `future_work`
- `negative_result`
- `reusable_design_pattern`

每条都必须绑定：

- `paper_id`
- `chunk_id`
- 原文 quote
- section role
- confidence

然后再由 ScholarInsight 跨论文聚合这些 structured evidence，进入 `g(c)` / `h(c)`。

## 下一阶段最小可行实验

不建议一开始跑 35,031 篇全库 MinerU。先做 30-100 篇高价值 pilot。

推荐实验：

1. 选 004 RAG+KG 的 accepted core papers 和 rejected boundary papers。
2. 用 MinerU 转 Markdown。
3. 建 chunk index。
4. 比较当前 title+abstract paper embedding 与 chunk embedding 的差异：
   - 是否召回更多 method / limitation / ablation evidence。
   - generic synthesis claim 数量是否下降。
   - report-ready claim 是否更具体。
   - advisor 是否更容易判断 novelty/usefulness。
5. 如果 004 有改善，再扩展到 006、010、011、012。

## 对当前问题的直接回答

你怀疑“最开始源头不够”是合理的。当前 pipeline 的 source gate、reranker、claim gate、falsification 已经解决了很多审计问题，但 evidence 的源头仍偏粗：

- paper-level embedding 太粗。
- focused_text 只覆盖前几页。
- evidence extraction 多依赖 sentence keyword ranking。
- 这会把 claim 推向“多篇论文共同围绕某分析轴”的 generic synthesis，而不是具体 research idea。

所以下一步质量提升主线应从“继续加 gate”转向“提高 evidence source granularity”：

> MinerU Markdown + section/chunk embedding + paper-local idea-evidence extraction，再接回现有 `g(c)` / `h(c)` 审计框架。

这比直接跑 132-topic 更重要。
