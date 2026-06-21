from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _fmt(value: float | int | str | None, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if value is None:
        return "n/a"
    return str(value)


def _variant(summary: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next((row for row in summary if row.get("variant") == name), {})


def _topic_name(row: dict[str, Any]) -> str:
    return f"{row.get('topic_id', '')} {row.get('topic', '')}".strip()


def _fresh_stats(packet: dict[str, Any]) -> dict[str, Any]:
    rows = packet.get("fresh_pilot") or []
    scores = [float(row.get("score") or 0) for row in rows]
    return {
        "topic_count": len(rows),
        "pass_count": sum(1 for row in rows if row.get("passed")),
        "score_min": min(scores) if scores else 0,
        "score_max": max(scores) if scores else 0,
        "report_ready_total": sum(int(row.get("report_ready_count") or 0) for row in rows),
        "falsification_total": sum(int(row.get("falsification_plan_count") or 0) for row in rows),
        "topics": ", ".join(_topic_name(row) for row in rows),
    }


def _fresh_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Topic | Score | Accepted | Evidence | Claims | Report-ready | Falsification plans | Device |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {_topic_name(row)} | {_fmt(row.get('score'))} | {row.get('accepted_count', 0)} "
            f"| {row.get('evidence_count', 0)} | {row.get('claim_count', 0)} "
            f"| {row.get('report_ready_count', 0)} | {row.get('falsification_plan_count', 0)} "
            f"| {row.get('reranker_device', '')} |"
        )
    return lines


def _delta_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Variant | Mean delta | Min delta | Max delta | Topics |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('variant', '')} | {_fmt(row.get('mean_delta'))} | {_fmt(row.get('min_delta'))} "
            f"| {_fmt(row.get('max_delta'))} | {row.get('topic_count', 0)} |"
        )
    return lines


def _runtime_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Variant | Mean score delta | Mean report-ready delta | Flags |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        flags = ", ".join(row.get("flags") or []) or "none"
        lines.append(
            f"| {row.get('variant', '')} | {_fmt(row.get('mean_score_delta'))} "
            f"| {_fmt(row.get('mean_report_ready_delta'))} | {flags} |"
        )
    return lines


def _claim_evidence_map(packet: dict[str, Any]) -> list[str]:
    items = packet.get("experiment_section_draft", {}).get("claim_evidence_map") or []
    lines = ["| Claim | Evidence | Status |", "|---|---|---|"]
    for item in items:
        lines.append(f"| {item.get('claim', '')} | {item.get('evidence', '')} | {item.get('status', '')} |")
    return lines


def render_markdown(packet: dict[str, Any]) -> str:
    fresh = _fresh_stats(packet)
    structural = packet.get("structural_ablation_summary") or []
    runtime = packet.get("runtime_ablation_summary") or []
    claim_gate = _variant(structural, "minus_claim_gate")
    hard_negative = _variant(structural, "minus_hard_negative")
    falsification = _variant(structural, "minus_falsification")
    no_reranker = _variant(runtime, "no_reranker")
    no_source_gate = _variant(runtime, "no_source_gate")
    method = packet["method_section_draft"]
    experiment = packet["experiment_section_draft"]
    gate = packet["formal_gate"]
    proposition = method["bounded_proposition"]

    lines: list[str] = [
        "# ScholarInsight: Auditable and Falsifiable Literature-Grounded Research Ideation",
        "",
        "Draft status: internal AAAI-style manuscript draft. Do not submit without citations, expert novelty/usefulness review, and final experimental decisions.",
        "",
        "## One-Sentence Argument",
        "",
        "In LLM-assisted literature ideation, ScholarInsight constrains early research directions through source-role retrieval, formal claim gating, hard-negative audit, and falsification-aware reporting, supported by a five-topic fresh pilot and ablations, with the boundary that the system makes claims inspectable rather than proving truth or novelty.",
        "",
        "## Terminology Ledger",
        "",
        "| Canonical term | First-use definition | Decision |",
        "|---|---|---|",
        "| ScholarInsight | ScholarInsight research-ideation pipeline | Use this name consistently. |",
        "| report-ready claim | A claim that passes the report-body gate g(c) and is eligible for the main report body | Define once, then use consistently. |",
        "| audit-only backlog | Evidence or observations preserved for review but excluded from the report body | Use when a claim is unsupported, sample-limited, or challenged. |",
        "| source role | A topic-family label for how a paper functions in the target topic, such as core method, benchmark, adjacent source, or hard-negative boundary | Use instead of generic relevance when discussing retrieval quality. |",
        "| g(c) | Formal report-body claim gate | Use for claim eligibility. |",
        "| h(c) | Report certificate combining g(c), counterexample coverage, and falsification planning | Use for auditability and falsifiability, not for truth. |",
        "",
        "## Abstract",
        "",
        f"LLM-assisted research ideation can produce fluent reports that appear useful while mixing in-scope papers, adjacent papers, and unsupported synthesis. This is risky when the output is used to allocate advisor time or experimental effort. We introduce ScholarInsight, a literature-grounded ideation pipeline that treats source relevance, claim generation, and falsifiability as linked audit problems. The pipeline builds a source-role-aware evidence pool, admits report-body claims only through a formal gate g(c), and issues a report certificate h(c) only when the claim also has counterexample coverage and a falsification plan. In a fresh local pilot over {fresh['topic_count']} AI/ML topic families ({fresh['topics']}), all {fresh['pass_count']} artifacts passed freeze validation, with idea-quality scores from {_fmt(fresh['score_min'])} to {_fmt(fresh['score_max'])}, {fresh['report_ready_total']} report-ready claims, and {fresh['falsification_total']} falsification plans. Structural ablation produced the largest mean drop when the claim gate was removed ({_fmt(claim_gate.get('mean_delta'))}), followed by hard-negative audit ({_fmt(hard_negative.get('mean_delta'))}) and falsification ({_fmt(falsification.get('mean_delta'))}). Runtime source-stage ablation showed that removing the reranker reduced report-ready synthesis in selected topics, while removing the source gate caused missing counterexample-audit flags. These results support ScholarInsight as an auditable and falsifiable ideation scaffold. They do not establish novelty, feasibility, or publishability.",
        "",
        "## 1 Introduction",
        "",
        "LLM systems are increasingly used to read papers, summarize fields, and propose research directions. The consequential use case is not merely text generation. It is an advisor-facing workflow in which a researcher decides which directions deserve further reading, implementation, or experiments. In this setting, a fluent but weakly grounded suggestion can waste substantial time because the failure is often hidden inside source selection and claim synthesis.",
        "",
        "A recurring failure mode is source drift. For example, a researcher asking for a research direction on retrieval-augmented generation with knowledge graphs may receive a report that mixes core KG-RAG methods with knowledge-graph construction, KGQA, and graph-reasoning neighbors. These papers can be topically related, but they should not all support the same claim. A useful ideation system should expose which sources are core support, which sources are adjacent, and which rejected sources are useful as hard-negative boundaries.",
        "",
        "Existing LLM and retrieval workflows do not fully solve this problem. Retrieval-oriented systems often optimize for topical match rather than source role. Claim generators often turn sample-limited or single-paper observations into broad synthesis. Report writers often omit the conditions under which a claim should be weakened or falsified. These omissions are especially harmful for early research reports because the reader needs to judge whether an idea is worth testing, not only whether the prose is plausible. [Citations needed: LLM-for-research systems, RAG survey, LLM-as-judge or verification work.]",
        "",
        f"Our goal is to make literature-grounded research ideation auditable and falsifiable. ScholarInsight requires every report-body research claim to carry an evidence certificate, source-role boundary, hard-negative audit, and falsification plan. The key distinction is between report-ready claims, which enter the main report body, and audit-only backlog observations, which remain inspectable but are not presented as conclusions.",
        "",
        "This setting raises three technical challenges. First, source drift requires retrieval to distinguish core, adjacent, rejected, and hard-negative papers across heterogeneous topics. Second, claim overreach requires a gate that prevents broad conclusions from single-paper or mixed-role evidence. Third, falsifiability requires the report to specify what evidence would weaken or reject each research direction.",
        "",
        "ScholarInsight addresses these challenges with a source-role-aware pipeline. It retrieves and reranks papers from a local index, classifies accepted and rejected sources by topic-family roles, builds evidence clusters, and admits claims into the report body only when g(c)=1. It then constructs h(c) by adding counterexample coverage and falsification planning. The resulting artifact is designed for human review. It is not a substitute for expert judgment about novelty, usefulness, or feasibility.",
        "",
        "This draft makes four contributions. First, it formulates auditable and falsifiable literature-grounded ideation as a concrete problem setting. Second, it introduces a source-role-aware evidence pipeline with formal claim gate g(c) and report certificate h(c). Third, it reports a fresh five-topic pilot with freeze validation, structural ablation, and runtime source-stage ablation. Fourth, it defines advisor-facing artifacts that expose evidence certificates, rejected-source boundaries, and falsification plans.",
        "",
        "## 2 Related Work",
        "",
        "LLM-assisted research systems aim to help researchers summarize literature, generate hypotheses, or plan experiments. These systems make research workflows more interactive, but they can also obscure whether a generated direction is grounded in the right sources. ScholarInsight differs by treating the generated report as an auditable artifact. The central question is not whether the report is fluent, but whether each report-body claim exposes source roles, evidence support, and falsification conditions. [Citations needed.]",
        "",
        "Retrieval-augmented generation connects generation to external documents, but standard retrieval scores do not by themselves decide how a source should function in a research argument. A paper can be close in embedding space while serving as an adjacent application, a benchmark, a rejected boundary, or a hard negative. ScholarInsight therefore keeps rejected sources and source-role labels as first-class audit data rather than discarding them after top-k selection. [Citations needed.]",
        "",
        "Verification and LLM-as-judge work has studied factuality, consistency, and rubric-based assessment. ScholarInsight is complementary because it moves verification into the evidence-to-claim interface. The gate g(c) decides whether a claim can appear in the report body, while h(c) adds counterexample coverage and falsification planning. This makes the artifact inspectable even when the final novelty and usefulness judgment remains human.",
        "",
        "## 3 Method",
        "",
        method["one_sentence_argument"],
        "",
        "### Task Formulation and Notation",
        "",
        f"Input. {method['task_formulation']['input']}",
        "",
        f"Output. {method['task_formulation']['output']}",
        "",
        f"Scope. {method['task_formulation']['scope']}",
        "",
        "Notation. Let S be the retrieved source pool, A the accepted source set, and R the rejected or boundary source set. Each accepted source s in A has a source role r(s). A candidate claim c is supported by evidence items E_c and independent papers P_c. Role-specific paper support is counted as P_{c,r}={p in P_c: r(p)=r}.",
        "",
    ]

    for subsection in method["subsections"]:
        lines.extend(
            [
                f"### {subsection['title']}",
                "",
                subsection["motivation"],
                "",
                subsection["mechanism"],
                "",
                f"Ablation role. {subsection['role_evidence']}",
                "",
            ]
        )

    lines.extend(
        [
            "### 3.5 Formal Guarantee and Boundary",
            "",
            f"{proposition['statement']}",
            "",
            f"Proof sketch. {proposition['proof_sketch']}",
            "",
            f"Boundary. {proposition['boundary']} In particular, h(c) does not prove truth, novelty, or publishability.",
            "",
            f"Claim gate used in this draft: `{gate['claim_gate']}`",
            "",
            f"Report certificate used in this draft: `{gate['report_certificate']}`",
            "",
            "## 4 Experiments",
            "",
            experiment["one_sentence_argument"],
            "",
            "### 4.1 Setup",
            "",
            f"The pilot covered {fresh['topics']}. {experiment['setup']['pipeline']} The metrics were {', '.join(experiment['setup']['metrics'])}.",
            "",
            *(_fresh_table(packet.get("fresh_pilot") or [])),
            "",
            "### 4.2 Fresh Pilot Freeze Validation",
            "",
            f"All {fresh['pass_count']} of {fresh['topic_count']} fresh pilot artifacts passed freeze validation. The score range was {_fmt(fresh['score_min'])} to {_fmt(fresh['score_max'])}. The artifacts contained {fresh['report_ready_total']} report-ready claims and {fresh['falsification_total']} falsification plans. This supports the claim that the current pipeline can produce advisor-reviewable artifacts on the tested topic families.",
            "",
            "### 4.3 Structural Ablation",
            "",
            "The structural ablation masks artifact fields from the same frozen reports. It is a proxy for module importance, not a full runtime rerun.",
            "",
            *(_delta_table(structural)),
            "",
            "The largest mean delta came from removing the claim gate. This is consistent with the design premise that report-body eligibility, not raw claim count, is the central quality control.",
            "",
            "### 4.4 Runtime Source-Stage Ablation",
            "",
            "The runtime ablation reran deterministic retrieval and source-stage variants without external LLM calls and without the 132-topic batch.",
            "",
            *(_runtime_table(runtime)),
            "",
            f"Removing the reranker produced a mean score delta of {_fmt(no_reranker.get('mean_score_delta'))} and a mean report-ready delta of {_fmt(no_reranker.get('mean_report_ready_delta'))}. Removing the source gate produced a mean score delta of {_fmt(no_source_gate.get('mean_score_delta'))} and surfaced counterexample-audit failures in the tested artifacts. The 010 topic remained stable under these source-stage ablations, which suggests that source-stage stress tests should be complemented by human or strong-model novelty review.",
            "",
            "### 4.5 Claim-Evidence Map",
            "",
            *(_claim_evidence_map(packet)),
            "",
            "## 5 Discussion",
            "",
            "The central advance is a shift from idea generation to auditable ideation. ScholarInsight does not claim that an LLM can prove a research direction is novel or publishable. Instead, it constrains which claims can enter a report and exposes the evidence, source boundaries, and falsification plans that a human reviewer needs to inspect.",
            "",
            "The ablations suggest that several controls are doing real work. The claim gate produces the largest structural drop when removed, source roles help keep retrieval evidence aligned with the target topic, and the source gate preserves rejected material needed for hard-negative audit. These results support the method's internal logic, but they remain a pilot-scale evaluation.",
            "",
            "The main alternative explanation is that the evaluator rewards artifacts that resemble the pipeline's own structure. This risk is real. The current evidence should therefore be interpreted as artifact auditability evidence, not as a final usefulness evaluation. A stronger paper needs blind expert or strong-model review of the generated research directions and, after advisor approval, broader topic coverage.",
            "",
            "The current system also depends on local index quality, source-role classifiers, and reranker behavior. A topic with sparse or noisy literature may pass fewer claims through g(c), which is preferable to overclaiming but may reduce report richness. This boundary should be presented as a design choice, not as a failure to be hidden.",
            "",
            "## 6 Conclusion",
            "",
            "ScholarInsight provides a source-role-aware pipeline for auditable and falsifiable literature-grounded research ideation. In the current five-topic pilot, it produced frozen local artifacts with report-ready claims, evidence certificates, hard-negative context, and falsification plans. The formal gate g(c) and report certificate h(c) make the artifact inspectable, while preserving a clear boundary: the system does not prove novelty, feasibility, usefulness, or publication readiness. The next evidence needed for submission is expert or blind strong-model review of idea quality and a decision on whether to extend from the pilot to the paused 132-topic batch.",
            "",
            "## Assumptions or Missing Inputs",
            "",
        ]
    )
    for item in method.get("assumptions_or_missing_inputs") or []:
        lines.append(f"- {item}")
    for item in experiment.get("limitations") or []:
        lines.append(f"- {item}")
    lines.extend(
        [
            "- Replace citation placeholders with verified references before any submission.",
            "- Decide whether the target format is AAAI full paper, workshop paper, or internal advisor packet before final rewriting.",
            "",
            "## Why This Structure",
            "",
            "- The draft follows a methods-paper chain: task, limitations, method, pilot validation, ablation, boundary.",
            "- The introduction defines the new setting before explaining the system, because the setting is part of the contribution.",
            "- The experiments report structural and runtime evidence separately so artifact-masked ablation is not overstated as a full rerun.",
            "- The discussion explicitly separates auditability/falsifiability from novelty and usefulness.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an AAAI-style manuscript draft from a ScholarInsight paper experiment packet.")
    parser.add_argument(
        "--packet-json",
        default="data/quality_audits/paper_experiment_packet_17c/paper_experiment_packet.json",
    )
    parser.add_argument("--output", default="docs/paper/aaai_draft.md")
    args = parser.parse_args()

    packet = read_json(Path(args.packet_json))
    output = Path(args.output)
    write_text(output, render_markdown(packet))
    print(json.dumps({"output": str(output), "fresh_topics": len(packet.get("fresh_pilot") or [])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
