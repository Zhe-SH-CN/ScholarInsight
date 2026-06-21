# ScholarInsight: Auditable and Falsifiable Literature-Grounded Research Ideation

Draft status: internal AAAI-style manuscript draft. Do not submit without citations, expert novelty/usefulness review, and final experimental decisions.

## One-Sentence Argument

In LLM-assisted literature ideation, ScholarInsight constrains early research directions through source-role retrieval, formal claim gating, hard-negative audit, and falsification-aware reporting, supported by a five-topic fresh pilot and ablations, with the boundary that the system makes claims inspectable rather than proving truth or novelty.

## Terminology Ledger

| Canonical term | First-use definition | Decision |
|---|---|---|
| ScholarInsight | ScholarInsight research-ideation pipeline | Use this name consistently. |
| report-ready claim | A claim that passes the report-body gate g(c) and is eligible for the main report body | Define once, then use consistently. |
| audit-only backlog | Evidence or observations preserved for review but excluded from the report body | Use when a claim is unsupported, sample-limited, or challenged. |
| source role | A topic-family label for how a paper functions in the target topic, such as core method, benchmark, adjacent source, or hard-negative boundary | Use instead of generic relevance when discussing retrieval quality. |
| g(c) | Formal report-body claim gate | Use for claim eligibility. |
| h(c) | Report certificate combining g(c), counterexample coverage, and falsification planning | Use for auditability and falsifiability, not for truth. |

## Abstract

LLM-assisted research ideation can produce fluent reports that appear useful while mixing in-scope papers, adjacent papers, and unsupported synthesis. This is risky when the output is used to allocate advisor time or experimental effort. We introduce ScholarInsight, a literature-grounded ideation pipeline that treats source relevance, claim generation, and falsifiability as linked audit problems. The pipeline builds a source-role-aware evidence pool, admits report-body claims only through a formal gate g(c), and issues a report certificate h(c) only when the claim also has counterexample coverage and a falsification plan. In a fresh local pilot over 5 AI/ML topic families (004 RAG with Knowledge Graphs, 006 Mathematical Reasoning, 010 Causal Reasoning with LLMs, 011 Counterfactual Inference, 012 Multi-hop Reasoning on Graphs), all 5 artifacts passed freeze validation, with idea-quality scores from 0.933 to 0.986, 14 report-ready claims, and 14 falsification plans. Structural ablation produced the largest mean drop when the claim gate was removed (-0.165), followed by hard-negative audit (-0.125) and falsification (-0.092). Runtime source-stage ablation showed that removing the reranker reduced report-ready synthesis in selected topics, while removing the source gate caused missing counterexample-audit flags. These results support ScholarInsight as an auditable and falsifiable ideation scaffold. They do not establish novelty, feasibility, or publishability.

## 1 Introduction

LLM systems are increasingly used to read papers, summarize fields, and propose research directions. The consequential use case is not merely text generation. It is an advisor-facing workflow in which a researcher decides which directions deserve further reading, implementation, or experiments. In this setting, a fluent but weakly grounded suggestion can waste substantial time because the failure is often hidden inside source selection and claim synthesis.

A recurring failure mode is source drift. For example, a researcher asking for a research direction on retrieval-augmented generation with knowledge graphs may receive a report that mixes core KG-RAG methods with knowledge-graph construction, KGQA, and graph-reasoning neighbors. These papers can be topically related, but they should not all support the same claim. A useful ideation system should expose which sources are core support, which sources are adjacent, and which rejected sources are useful as hard-negative boundaries.

Existing LLM and retrieval workflows do not fully solve this problem. Retrieval-oriented systems often optimize for topical match rather than source role. Claim generators often turn sample-limited or single-paper observations into broad synthesis. Report writers often omit the conditions under which a claim should be weakened or falsified. These omissions are especially harmful for early research reports because the reader needs to judge whether an idea is worth testing, not only whether the prose is plausible. [Citations needed: LLM-for-research systems, RAG survey, LLM-as-judge or verification work.]

Our goal is to make literature-grounded research ideation auditable and falsifiable. ScholarInsight requires every report-body research claim to carry an evidence certificate, source-role boundary, hard-negative audit, and falsification plan. The key distinction is between report-ready claims, which enter the main report body, and audit-only backlog observations, which remain inspectable but are not presented as conclusions.

This setting raises three technical challenges. First, source drift requires retrieval to distinguish core, adjacent, rejected, and hard-negative papers across heterogeneous topics. Second, claim overreach requires a gate that prevents broad conclusions from single-paper or mixed-role evidence. Third, falsifiability requires the report to specify what evidence would weaken or reject each research direction.

ScholarInsight addresses these challenges with a source-role-aware pipeline. It retrieves and reranks papers from a local index, classifies accepted and rejected sources by topic-family roles, builds evidence clusters, and admits claims into the report body only when g(c)=1. It then constructs h(c) by adding counterexample coverage and falsification planning. The resulting artifact is designed for human review. It is not a substitute for expert judgment about novelty, usefulness, or feasibility.

This draft makes four contributions. First, it formulates auditable and falsifiable literature-grounded ideation as a concrete problem setting. Second, it introduces a source-role-aware evidence pipeline with formal claim gate g(c) and report certificate h(c). Third, it reports a fresh five-topic pilot with freeze validation, structural ablation, and runtime source-stage ablation. Fourth, it defines advisor-facing artifacts that expose evidence certificates, rejected-source boundaries, and falsification plans.

## 2 Related Work

LLM-assisted research systems aim to help researchers summarize literature, generate hypotheses, or plan experiments. These systems make research workflows more interactive, but they can also obscure whether a generated direction is grounded in the right sources. ScholarInsight differs by treating the generated report as an auditable artifact. The central question is not whether the report is fluent, but whether each report-body claim exposes source roles, evidence support, and falsification conditions. [Citations needed.]

Retrieval-augmented generation connects generation to external documents, but standard retrieval scores do not by themselves decide how a source should function in a research argument. A paper can be close in embedding space while serving as an adjacent application, a benchmark, a rejected boundary, or a hard negative. ScholarInsight therefore keeps rejected sources and source-role labels as first-class audit data rather than discarding them after top-k selection. [Citations needed.]

Verification and LLM-as-judge work has studied factuality, consistency, and rubric-based assessment. ScholarInsight is complementary because it moves verification into the evidence-to-claim interface. The gate g(c) decides whether a claim can appear in the report body, while h(c) adds counterexample coverage and falsification planning. This makes the artifact inspectable even when the final novelty and usefulness judgment remains human.

## 3 Method

In literature-grounded research ideation, ScholarInsight exposes auditable report-body claims by combining source-role retrieval, formal evidence certification, hard-negative audit, and falsification-aware report rendering, with the boundary that these certificates support inspection rather than prove truth or novelty.

### Task Formulation and Notation

Input. A target research topic, optional seed papers, topic-family dimensions, a local paper index, and a local embedding/reranker stack.

Output. An advisor-facing report with report-ready claims, audit-only backlog observations, evidence certificates, rejected-source boundaries, and falsification plans.

Scope. The method is designed for early-stage AI/ML literature ideation. It constrains source relevance and claim auditing; it does not replace human novelty, feasibility, or usefulness review.

Notation. Let S be the retrieved source pool, A the accepted source set, and R the rejected or boundary source set. Each accepted source s in A has a source role r(s). A candidate claim c is supported by evidence items E_c and independent papers P_c. Role-specific paper support is counted as P_{c,r}={p in P_c: r(p)=r}.

### 3.1 Source-role retrieval and boundary source tracking

Embedding retrieval alone can rank topically similar but methodologically out-of-scope papers above core papers. ScholarInsight therefore treats retrieval as a source-boundary construction problem, not only as top-k recall.

The pipeline first retrieves candidates from the local paper index, reranks them with a cross-encoder reranker, and applies topic-family source-role classifiers. Accepted sources enter A with a relevance score and source role. Rejected sources remain in R with a rejection reason, so the system can later use them as boundary cases or hard negatives.

Ablation role. Runtime source-stage ablation showed that removing the reranker reduced report-ready counts for 004 and 006, while removing the source gate produced missing counterexample-audit flags in four of five fresh pilot topics.

### 3.2 Evidence clusters and formal claim gate g(c)

A fluent literature summary can overstate single-paper or mixed-role observations. The report body therefore needs a gate that separates synthesis claims from audit-only backlog observations.

ScholarInsight builds evidence clusters from accepted sources and admits a candidate claim into the report body only when g(c)=1. The gate requires verified status, non-high risk, synthesis wording, at least two evidence items, at least two independent papers, strong support, reportable source roles, cross-role balance when a cross-role claim is made, and no backlog or artifact wording.

Ablation role. Structural ablation over fresh artifacts showed the largest mean quality drop when the claim gate was removed, which supports treating g(c) as the central report-body inclusion criterion.

### 3.3 Hard-negative audit and report certificate h(c)

A research direction is weak if the report does not say how it could fail. Rejected and adjacent sources are therefore preserved as audit material rather than discarded.

For each report-ready claim, the system links boundary sources to hard-negative challenges, then creates a falsification plan with a falsification criterion, benchmark or task perturbation, expected failure mode, and negative-result logging schema. The report certificate is h(c)=g(c) AND counterexample_covered(c) AND falsification_plan_exists(c).

Ablation role. In the fresh five-topic pilot, falsification coverage matched the report-ready claim count for every topic. Structural ablation of falsification and hard-negative components produced consistent quality drops.

### 3.4 Report rendering and advisor-facing artifacts

Advisor-facing ideation artifacts must expose why a claim appears in the report and why other material was kept outside it.

The renderer separates report-ready findings from audit-only backlog, writes citations and evidence appendices, exports claims, evidence, rejected sources, counterexample audits, and falsification plans, and produces mentor review packets for human judgment.

Ablation role. The pilot freeze validator checks fresh provenance, CUDA reranker use, required report sections, report-ready claim counts, falsification coverage, and evaluator scores before a pilot is treated as frozen.

### 3.5 Formal Guarantee and Boundary

Proposition 1 (audit certificate). If h(c)=1 for a claim c, then the generated artifact exposes an auditable evidence certificate for c, a source-role boundary with hard-negative context, and a falsification plan for negative-result logging.

Proof sketch. By definition, h(c)=g(c) requires the evidence ids, independent-paper support, source-role constraints, and report-body eligibility recorded by g(c). The counterexample_covered(c) condition links c to boundary or hard-negative sources, and falsification_plan_exists(c) requires the failure criterion, perturbation, expected failure mode, and logging schema. The artifact renderer writes these fields to the report and exports.

Boundary. The proposition is not a truth, novelty, or publishability guarantee. It only states what the artifact makes inspectable. In particular, h(c) does not prove truth, novelty, or publishability.

Claim gate used in this draft: `g(c)=1 iff a claim is verified, low/non-high risk, comparative or cross-role, supported by at least two evidence items from at least two independent papers, has strong evidence support, uses only reportable source roles, satisfies cross-role balance when applicable, and is free of backlog/artifact wording.`

Report certificate used in this draft: `h(c)=g(c) AND counterexample_covered(c) AND falsification_plan_exists(c).`

## 4 Experiments

The pilot experiments tested whether ScholarInsight's quality controls produced fresh, auditable reports across multiple topic families and whether removing retrieval or audit components weakened report-ready evidence.

### 4.1 Setup

The pilot covered 004 RAG with Knowledge Graphs, 006 Mathematical Reasoning, 010 Causal Reasoning with LLMs, 011 Counterfactual Inference, 012 Multi-hop Reasoning on Graphs. All fresh pilots used the deterministic local pipeline, local paper artifacts, direct Hugging Face model loading, and CUDA bge-reranker provenance. No external LLM calls were used for the frozen pilot artifacts. The metrics were idea-quality score, report-ready claim count, falsification-plan coverage, validator pass/fail status, quality flags and evaluator flags, source-purity and counterexample-audit availability.

| Topic | Score | Accepted | Evidence | Claims | Report-ready | Falsification plans | Device |
|---|---:|---:|---:|---:|---:|---:|---|
| 004 RAG with Knowledge Graphs | 0.986 | 12 | 46 | 6 | 4 | 4 | cuda |
| 006 Mathematical Reasoning | 0.933 | 25 | 49 | 7 | 3 | 3 | cuda |
| 010 Causal Reasoning with LLMs | 0.969 | 9 | 29 | 4 | 1 | 1 | cuda |
| 011 Counterfactual Inference | 0.986 | 28 | 31 | 6 | 4 | 4 | cuda |
| 012 Multi-hop Reasoning on Graphs | 0.970 | 18 | 37 | 7 | 2 | 2 | cuda |

### 4.2 Fresh Pilot Freeze Validation

All 5 of 5 fresh pilot artifacts passed freeze validation. The score range was 0.933 to 0.986. The artifacts contained 14 report-ready claims and 14 falsification plans. This supports the claim that the current pipeline can produce advisor-reviewable artifacts on the tested topic families.

### 4.3 Structural Ablation

The structural ablation masks artifact fields from the same frozen reports. It is a proxy for module importance, not a full runtime rerun.

| Variant | Mean delta | Min delta | Max delta | Topics |
|---|---:|---:|---:|---:|
| minus_claim_gate | -0.165 | -0.186 | -0.132 | 5 |
| minus_experiment_framing | -0.041 | -0.042 | -0.041 | 5 |
| minus_falsification | -0.092 | -0.092 | -0.091 | 5 |
| minus_hard_negative | -0.125 | -0.125 | -0.125 | 5 |
| minus_source_roles | -0.083 | -0.084 | -0.083 | 5 |

The largest mean delta came from removing the claim gate. This is consistent with the design premise that report-body eligibility, not raw claim count, is the central quality control.

### 4.4 Runtime Source-Stage Ablation

The runtime ablation reran deterministic retrieval and source-stage variants without external LLM calls and without the 132-topic batch.

| Variant | Mean score delta | Mean report-ready delta | Flags |
|---|---:|---:|---|
| no_reranker | -0.011 | -0.400 | accepted_sources_include_non_reportable_roles |
| no_reranker_no_source_gate | -0.093 | -0.400 | accepted_sources_include_non_reportable_roles, missing_counterexample_audit |
| no_source_gate | -0.100 | 0 | missing_counterexample_audit |

Removing the reranker produced a mean score delta of -0.011 and a mean report-ready delta of -0.400. Removing the source gate produced a mean score delta of -0.100 and surfaced counterexample-audit failures in the tested artifacts. The 010 topic remained stable under these source-stage ablations, which suggests that source-stage stress tests should be complemented by human or strong-model novelty review.

### 4.5 Claim-Evidence Map

| Claim | Evidence | Status |
|---|---|---|
| Fresh artifacts are reproducible enough for advisor review. | pilot_freeze_validator pass status, provenance checks, mentor packets, and complete required report sections. | supported |
| g(c) and h(c) improve auditability and falsifiability. | structural ablation deltas for claim gate, hard-negative audit, falsification, and source roles. | supported as a structural proxy |
| The pipeline improves research novelty or usefulness. | human or blind strong-model review has not been run. | needs evidence |
| The system generalizes to all 132 topics. | only five frozen pilot topics have been validated. | needs evidence |

## 5 Discussion

The central advance is a shift from idea generation to auditable ideation. ScholarInsight does not claim that an LLM can prove a research direction is novel or publishable. Instead, it constrains which claims can enter a report and exposes the evidence, source boundaries, and falsification plans that a human reviewer needs to inspect.

The ablations suggest that several controls are doing real work. The claim gate produces the largest structural drop when removed, source roles help keep retrieval evidence aligned with the target topic, and the source gate preserves rejected material needed for hard-negative audit. These results support the method's internal logic, but they remain a pilot-scale evaluation.

The main alternative explanation is that the evaluator rewards artifacts that resemble the pipeline's own structure. This risk is real. The current evidence should therefore be interpreted as artifact auditability evidence, not as a final usefulness evaluation. A stronger paper needs blind expert or strong-model review of the generated research directions and, after advisor approval, broader topic coverage.

The current system also depends on local index quality, source-role classifiers, and reranker behavior. A topic with sparse or noisy literature may pass fewer claims through g(c), which is preferable to overclaiming but may reduce report richness. This boundary should be presented as a design choice, not as a failure to be hidden.

## 6 Conclusion

ScholarInsight provides a source-role-aware pipeline for auditable and falsifiable literature-grounded research ideation. In the current five-topic pilot, it produced frozen local artifacts with report-ready claims, evidence certificates, hard-negative context, and falsification plans. The formal gate g(c) and report certificate h(c) make the artifact inspectable, while preserving a clear boundary: the system does not prove novelty, feasibility, usefulness, or publication readiness. The next evidence needed for submission is expert or blind strong-model review of idea quality and a decision on whether to extend from the pilot to the paused 132-topic batch.

## Assumptions or Missing Inputs

- Final manuscript citations to prior AI-for-research and LLM-ideation systems are still needed.
- Human or blind strong-model review is still needed for novelty, usefulness, and feasibility.
- The public release plan for code, local index construction, and model weights should be stated before submission.
- Only five topic families are frozen; the 132-topic batch is intentionally paused.
- The evaluator is a structural quality proxy, not a substitute for expert review.
- The ablation package combines artifact-masked structural ablation with deterministic source-stage runtime ablation; it is not yet a full end-to-end randomized experiment.
- The current evidence does not establish novelty, feasibility, or publication quality.
- Replace citation placeholders with verified references before any submission.
- Decide whether the target format is AAAI full paper, workshop paper, or internal advisor packet before final rewriting.

## Why This Structure

- The draft follows a methods-paper chain: task, limitations, method, pilot validation, ablation, boundary.
- The introduction defines the new setting before explaining the system, because the setting is part of the contribution.
- The experiments report structural and runtime evidence separately so artifact-masked ablation is not overstated as a full rerun.
- The discussion explicitly separates auditability/falsifiability from novelty and usefulness.
