# Paper Relationship Discovery

Given two papers, determine whether Paper A is an innovation successor of Paper B.

## Paper A: Candidate Successor

`{paper_a_json}`

## Paper B: Candidate Predecessor

`{paper_b_json}`

## Task

1. Decide whether Paper A builds on, improves, extends, reframes, or systematizes Paper B.
2. If yes, choose exactly one reasoning pattern from the CognoGraph schema.
3. Explain the bottleneck in Paper B and the mechanism introduced by Paper A.

If a relationship exists, return:

```json
{
  "has_edge": true,
  "reasoning_pattern": "pattern_id",
  "bottleneck": "specific technical bottleneck solved",
  "mechanism": "specific technical mechanism used",
  "evidence": "brief evidence from the paper information",
  "confidence": 0.85
}
```

If no relationship exists, return:

```json
{
  "has_edge": false,
  "reason": "brief reason"
}
```

Return valid JSON only.

## Batch Mode

If the input contains a JSON object with a `pairs` array, judge every pair in that array. Use each pair's `source_paper` as Paper A and `target_paper` as Paper B. Return a JSON list with exactly one result per input pair, in the same order.

Each batch result must preserve endpoint titles:

```json
{
  "source_title": "same source_title from the input pair",
  "target_title": "same target_title from the input pair",
  "has_edge": true,
  "reasoning_pattern": "pattern_id",
  "bottleneck": "specific technical bottleneck solved",
  "mechanism": "specific technical mechanism used",
  "evidence": "brief evidence from source_paper and target_paper",
  "confidence": 0.85
}
```

Do not judge from titles alone when `source_paper` and `target_paper` are available. Use title, abstract, methods, and references as evidence.
