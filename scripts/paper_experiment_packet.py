from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def fresh_pilot_rows(fresh_root: Path) -> list[dict[str, Any]]:
    validation = read_json(fresh_root / "pilot_freeze_validation.json")
    rows: list[dict[str, Any]] = []
    for item in validation.get("items") or []:
        summary = item.get("summary") or {}
        rows.append(
            {
                "topic_id": str(item.get("topic_id")),
                "topic": item.get("topic"),
                "passed": bool(item.get("passed")),
                "score": float(summary.get("idea_quality_score") or 0),
                "report_ready_count": int(summary.get("report_ready_count") or 0),
                "accepted_count": int(summary.get("accepted_count") or 0),
                "evidence_count": int(summary.get("evidence_count") or 0),
                "claim_count": int(summary.get("claim_count") or 0),
                "falsification_plan_count": int(summary.get("falsification_plan_count") or 0),
                "reranker_device": summary.get("reranker_device_loaded") or "",
                "errors": item.get("errors") or [],
                "warnings": item.get("warnings") or [],
            }
        )
    return rows


def structural_delta_summary(structural_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full_by_topic = {
        row["topic_id"]: float(row["overall_score"])
        for row in structural_rows
        if row.get("variant") == "full"
    }
    deltas: dict[str, list[float]] = defaultdict(list)
    for row in structural_rows:
        variant = row.get("variant")
        if variant == "full":
            continue
        topic_id = row.get("topic_id")
        if topic_id not in full_by_topic:
            continue
        deltas[str(variant)].append(float(row["overall_score"]) - full_by_topic[topic_id])
    return [
        {
            "variant": variant,
            "mean_delta": round(mean(values), 3),
            "min_delta": round(min(values), 3),
            "max_delta": round(max(values), 3),
            "topic_count": len(values),
        }
        for variant, values in sorted(deltas.items())
        if values
    ]


def runtime_delta_summary(runtime_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in runtime_rows:
        if row.get("variant") != "full":
            grouped[str(row.get("variant"))].append(row)
    summary = []
    for variant, rows in sorted(grouped.items()):
        deltas = [float(row.get("score_delta_vs_full") or 0) for row in rows]
        report_ready_deltas = []
        for row in rows:
            topic_id = row.get("topic_id")
            full = next(
                (
                    item
                    for item in runtime_rows
                    if item.get("topic_id") == topic_id and item.get("variant") == "full"
                ),
                None,
            )
            if full:
                report_ready_deltas.append(
                    int(row.get("report_ready_count") or 0)
                    - int(full.get("report_ready_count") or 0)
                )
        flags = sorted(
            {
                flag
                for row in rows
                for flag in (row.get("quality_flags") or []) + (row.get("evaluator_flags") or [])
            }
        )
        summary.append(
            {
                "variant": variant,
                "mean_score_delta": round(mean(deltas), 3),
                "min_score_delta": round(min(deltas), 3),
                "max_score_delta": round(max(deltas), 3),
                "mean_report_ready_delta": round(mean(report_ready_deltas), 3)
                if report_ready_deltas
                else 0,
                "flags": flags,
                "topic_count": len(rows),
            }
        )
    return summary


def formal_gate_note() -> dict[str, Any]:
    return {
        "claim_gate": "g(c)=1 iff a claim is verified, low/non-high risk, comparative or cross-role, supported by at least two evidence items from at least two independent papers, has strong evidence support, uses only reportable source roles, satisfies cross-role balance when applicable, and is free of backlog/artifact wording.",
        "report_certificate": "h(c)=g(c) AND counterexample_covered(c) AND falsification_plan_exists(c).",
        "guarantee": [
            "If h(c)=1, the report can expose the supporting evidence ids, independent paper count, source-role paper counts, boundary/hard-negative audit rows, and a falsification/negative-result logging plan.",
            "The guarantee is an auditability and falsifiability guarantee, not a guarantee that the claim is true, novel, or publishable.",
        ],
        "paper_safe_claim": "ScholarInsight constrains literature-grounded ideation by requiring report-body claims to carry an explicit evidence certificate and falsification interface.",
        "paper_unsafe_claim": "Do not claim automatic AAAI-level ideation or scientific novelty proof.",
    }


def paper_skeleton_note() -> dict[str, Any]:
    return {
        "paper_type": "New Problem/Setting paper with a technique contribution",
        "positioning_rationale": (
            "The paper should introduce auditable, falsifiable literature-grounded research ideation "
            "as the setting, then present ScholarInsight as the mechanism that makes the setting operational."
        ),
        "thinking_template": [
            {
                "stage": "Research background",
                "content": (
                    "LLM-assisted research ideation is increasingly used to scan literature, propose research questions, "
                    "and draft early-stage reports. The high-stakes use case is not answer generation, but deciding which "
                    "research directions deserve human attention, experiments, and advisor time."
                ),
            },
            {
                "stage": "Limitation 1",
                "content": (
                    "Generic LLM/RAG workflows can produce fluent research claims while drifting to adjacent papers; "
                    "they rarely expose why each source is relevant or why rejected sources should not support a conclusion."
                ),
            },
            {
                "stage": "Limitation 2",
                "content": (
                    "Existing ideation pipelines usually separate literature retrieval from claim validation, so challenged "
                    "or sample-limited observations can enter the report body as if they were verified synthesis."
                ),
            },
            {
                "stage": "Limitation 3",
                "content": (
                    "Early research reports often lack an explicit falsification interface: they suggest ideas without "
                    "hard-negative evidence, failure criteria, or negative-result logging plans."
                ),
            },
            {
                "stage": "Key Idea / Our Goal",
                "content": (
                    "ScholarInsight turns research ideation into an auditable evidence pipeline in which a report-body "
                    "claim must pass source-role grounding, formal evidence certification, hard-negative audit, and "
                    "falsification planning before it is presented as a research direction."
                ),
            },
            {
                "stage": "Challenge 1",
                "content": (
                    "Source drift: retrieval must distinguish core papers, adjacent papers, rejected papers, and "
                    "hard-negative boundary sources across heterogeneous topic families."
                ),
            },
            {
                "stage": "Challenge 2",
                "content": (
                    "Claim overreach: the system must prevent single-paper or sample-limited observations from becoming "
                    "general research claims while still preserving them as backlog evidence."
                ),
            },
            {
                "stage": "Challenge 3",
                "content": (
                    "Falsifiability: a generated research direction must carry conditions under which it should be "
                    "weakened, rejected, or moved back to evidence collection."
                ),
            },
            {
                "stage": "Methodology topic sentence",
                "content": (
                    "ScholarInsight is a source-role-aware ideation pipeline with formal claim gating, hard-negative "
                    "boundary audit, and falsification-aware report rendering."
                ),
            },
            {
                "stage": "Module A (addresses Challenge 1)",
                "content": (
                    "Source-role retrieval and reranking combine embedding search, cross-encoder reranking, "
                    "topic-family source classifiers, and rejected-source audit records."
                ),
            },
            {
                "stage": "Module B (addresses Challenge 2)",
                "content": (
                    "Evidence-to-claim gate g(c) requires verified synthesis, multi-evidence support, independent papers, "
                    "reportable source roles, and backlog separation before report-body inclusion."
                ),
            },
            {
                "stage": "Module C (addresses Challenge 3)",
                "content": (
                    "Hard-negative and falsification interface h(c) links report-ready claims to boundary sources, failure "
                    "criteria, benchmark perturbations, and negative-result logging schema."
                ),
            },
            {
                "stage": "Contribution 1",
                "content": (
                    "A problem framing and pipeline for auditable, falsifiable literature-grounded research ideation "
                    "(Sections 1-3)."
                ),
            },
            {
                "stage": "Contribution 2",
                "content": (
                    "A source-role-aware evidence and claim certification mechanism, including g(c) and h(c), that separates "
                    "report-ready claims from backlog observations (Section 3)."
                ),
            },
            {
                "stage": "Contribution 3",
                "content": (
                    "A pilot evaluation over five topic families with fresh CUDA reruns, freeze validation, structural "
                    "ablation, and runtime source-stage ablation (Section 4)."
                ),
            },
            {
                "stage": "Contribution 4",
                "content": (
                    "A report artifact design that exposes evidence certificates, rejected-source boundaries, and "
                    "falsification plans for advisor or human review (Sections 3-5)."
                ),
            },
        ],
        "methodology_outline": [
            {
                "section": "3.1 Source-role retrieval and boundary source tracking",
                "summary": "Describe embedding retrieval, reranker use, topic-family source roles, accepted/rejected sources, and why rejected sources remain useful for counterexample audit.",
            },
            {
                "section": "3.2 Evidence clusters and formal claim gate g(c)",
                "summary": "Define evidence clusters, reportable source roles, independent-paper support, and the exact conditions for g(c)=1.",
            },
            {
                "section": "3.3 Hard-negative audit and report certificate h(c)",
                "summary": "Define counterexample coverage, falsification plans, benchmark perturbations, and negative-result logging.",
            },
            {
                "section": "3.4 Report rendering and advisor-facing artifacts",
                "summary": "Explain report-ready findings, audit-only backlog, mentor packets, validation metadata, and what the system deliberately does not claim.",
            },
        ],
        "self_consistency_checks": {
            "limitations_to_key_idea": "pass",
            "key_idea_to_challenges": "pass",
            "challenges_to_methodology": "pass",
            "methodology_to_contributions": "pass",
        },
        "severity_summary": "0 CRITICAL, 0 MAJOR, 1 MINOR: novelty/usefulness still requires human or blind strong-model review.",
    }


def introduction_outline_note() -> dict[str, Any]:
    return {
        "type_positioning": {
            "type": "New Problem/Setting Paper",
            "rationale": (
                "The main contribution is the formulation of auditable and falsifiable research ideation, "
                "with ScholarInsight as the operational mechanism."
            ),
            "implication": (
                "Paragraph 3 should carry substantial weight: the goal and hard constraints are the paper's framing, "
                "while the pipeline explains why the framing is feasible."
            ),
        },
        "paragraphs": [
            {
                "id": "P1",
                "title": "Background and Motivation",
                "purpose": "Make research ideation a concrete, high-stakes task rather than generic LLM text generation.",
                "running_example": (
                    "A researcher asks for a RAG-with-Knowledge-Graphs research direction; a naive literature agent mixes "
                    "core KG-RAG papers with KG construction and KGQA neighbors, then writes a broad trend claim without "
                    "showing which sources are in-scope, out-of-scope, or useful as hard negatives."
                ),
                "writing_points": [
                    "Open with literature-grounded ideation as an advisor-facing workflow.",
                    "Use the RAG+KG example to show source drift and unsupported claim risk.",
                    "State that the desired output is not a final paper idea, but an auditable research direction for human review.",
                ],
                "gaps": ["MINOR: add 3-5 citations to recent LLM-for-research or AI-scientist systems before manuscript drafting."],
            },
            {
                "id": "P2",
                "title": "Limitations",
                "purpose": "Explain why existing LLM/RAG ideation workflows are insufficient.",
                "writing_points": [
                    "Limitation 1: retrieval-oriented agents do not reliably expose source-role relevance or rejected-source boundaries.",
                    "Limitation 2: claim generation and claim validation are often weakly coupled, so challenged observations can enter the report body.",
                    "Limitation 3: generated research directions rarely include falsification criteria, benchmark perturbations, or negative-result logging.",
                ],
                "gaps": ["MINOR: cite concrete prior systems in each limitation."],
            },
            {
                "id": "P3",
                "title": "Problem Essence and Our Goal",
                "purpose": "Define auditable, falsifiable research ideation as the paper's setting.",
                "hard_constraints": [
                    "The system must operate over heterogeneous topic families.",
                    "Report-body claims must be traceable to evidence and source roles.",
                    "The output must preserve rejected and hard-negative sources instead of hiding them.",
                    "The system must expose when a claim should be weakened or falsified.",
                ],
                "goal_sentence": (
                    "Our goal is to make literature-grounded research ideation auditable and falsifiable by requiring every "
                    "report-body research claim to carry an evidence certificate, source-role boundary, hard-negative audit, "
                    "and falsification plan."
                ),
                "writing_points": [
                    "Define report-body claim and audit-only backlog.",
                    "Introduce g(c) and h(c) at a high level, saving formal definitions for Method.",
                    "Emphasize that the goal is a constraint on ideation quality, not automatic novelty proof.",
                ],
                "gaps": [],
            },
            {
                "id": "P4",
                "title": "Key Challenges",
                "purpose": "Show why the setting is nontrivial.",
                "writing_points": [
                    "Challenge 1 source drift: naive retrieval treats adjacent papers as support because topical overlap is not the same as source role relevance.",
                    "Challenge 2 claim overreach: naive synthesis rewards broad conclusions even when evidence is single-paper, sample-limited, or mixed-role.",
                    "Challenge 3 falsifiability: naive reports present recommendations without specifying hard negatives or negative-result decisions.",
                ],
                "gaps": [],
            },
            {
                "id": "P5",
                "title": "Solution Overview",
                "purpose": "Map each challenge to one ScholarInsight module.",
                "challenge_to_module_mapping": [
                    "Challenge 1 -> source-role retrieval, reranking, and rejected-source boundary tracking.",
                    "Challenge 2 -> evidence clustering plus formal claim gate g(c).",
                    "Challenge 3 -> hard-negative audit plus report certificate h(c) and falsification-plan rendering.",
                ],
                "writing_points": [
                    "Return to the RAG+KG running example: core KG-RAG papers enter evidence extraction, adjacent KG construction/KGQA papers become rejected or hard-negative context.",
                    "Show that only claims passing g(c) enter report body.",
                    "Show that h(c) adds counterexample coverage and falsification plans before advisor review.",
                ],
                "gaps": [],
            },
            {
                "id": "P6",
                "title": "Contributions",
                "purpose": "State exactly what the paper delivers and where.",
                "contributions": [
                    "We formulate auditable, falsifiable literature-grounded research ideation as a problem setting and define the report-body/backlog distinction (Section 2).",
                    "We introduce ScholarInsight, a source-role-aware evidence pipeline with formal claim gate g(c), report certificate h(c), and falsification-aware report rendering (Section 3).",
                    "We evaluate the pipeline on five topic families with fresh CUDA reruns, freeze validation, structural ablation, and deterministic runtime source-stage ablation (Section 4).",
                    "We provide advisor-facing artifacts, including mentor review packets and paper experiment packets, that expose evidence certificates and remaining human-review gaps (Section 5).",
                ],
                "gaps": ["MINOR: final manuscript should add human or blind strong-model novelty/usefulness review before claiming advisor-facing usefulness."],
            },
        ],
        "flowchart_consistency": {
            "running_example_loop": "pass",
            "limitations_challenges_link": "pass",
            "goal_contribution1_link": "pass",
            "challenge_module_mapping": "pass",
            "contribution_section_mapping": "pass",
        },
        "integrity_gate": {
            "gate_1_running_example_reappears": "pass",
            "gate_2_limitations_at_most_three": "pass",
            "gate_3_challenges_at_most_three": "pass",
            "gate_4_mapping_one_to_one": "pass",
            "gate_5_contributions_three_or_four": "pass",
            "gate_6_no_vague_contributions": "pass",
            "gate_7_new_problem_weight_reflected": "pass",
        },
        "severity_summary": "0 CRITICAL, 0 MAJOR, 2 MINOR: add citations to prior ideation systems and complete human/strong-model usefulness review.",
    }


def method_section_draft_note() -> dict[str, Any]:
    return {
        "section": "Section 3 Method",
        "one_sentence_argument": (
            "In literature-grounded research ideation, ScholarInsight exposes auditable report-body claims by combining "
            "source-role retrieval, formal evidence certification, hard-negative audit, and falsification-aware report rendering, "
            "with the boundary that these certificates support inspection rather than prove truth or novelty."
        ),
        "task_formulation": {
            "input": (
                "A target research topic, optional seed papers, topic-family dimensions, a local paper index, and a local "
                "embedding/reranker stack."
            ),
            "output": (
                "An advisor-facing report with report-ready claims, audit-only backlog observations, evidence certificates, "
                "rejected-source boundaries, and falsification plans."
            ),
            "scope": (
                "The method is designed for early-stage AI/ML literature ideation. It constrains source relevance and claim "
                "auditing; it does not replace human novelty, feasibility, or usefulness review."
            ),
        },
        "notation": [
            "Let S be the retrieved source pool, A the accepted source set, and R the rejected or boundary source set.",
            "Each accepted source s in A has a source role r(s), such as core method, benchmark, application case, or adjacent/background.",
            "A candidate claim c is supported by evidence items E_c and independent papers P_c.",
            "Role-specific paper support is counted as P_{c,r}={p in P_c: r(p)=r}.",
        ],
        "subsections": [
            {
                "title": "3.1 Source-role retrieval and boundary source tracking",
                "motivation": (
                    "Embedding retrieval alone can rank topically similar but methodologically out-of-scope papers above "
                    "core papers. ScholarInsight therefore treats retrieval as a source-boundary construction problem, not "
                    "only as top-k recall."
                ),
                "mechanism": (
                    "The pipeline first retrieves candidates from the local paper index, reranks them with a cross-encoder "
                    "reranker, and applies topic-family source-role classifiers. Accepted sources enter A with a relevance "
                    "score and source role. Rejected sources remain in R with a rejection reason, so the system can later "
                    "use them as boundary cases or hard negatives."
                ),
                "role_evidence": (
                    "Runtime source-stage ablation showed that removing the reranker reduced report-ready counts for 004 "
                    "and 006, while removing the source gate produced missing counterexample-audit flags in four of five "
                    "fresh pilot topics."
                ),
            },
            {
                "title": "3.2 Evidence clusters and formal claim gate g(c)",
                "motivation": (
                    "A fluent literature summary can overstate single-paper or mixed-role observations. The report body "
                    "therefore needs a gate that separates synthesis claims from audit-only backlog observations."
                ),
                "mechanism": (
                    "ScholarInsight builds evidence clusters from accepted sources and admits a candidate claim into the "
                    "report body only when g(c)=1. The gate requires verified status, non-high risk, synthesis wording, at "
                    "least two evidence items, at least two independent papers, strong support, reportable source roles, "
                    "cross-role balance when a cross-role claim is made, and no backlog or artifact wording."
                ),
                "role_evidence": (
                    "Structural ablation over fresh artifacts showed the largest mean quality drop when the claim gate was "
                    "removed, which supports treating g(c) as the central report-body inclusion criterion."
                ),
            },
            {
                "title": "3.3 Hard-negative audit and report certificate h(c)",
                "motivation": (
                    "A research direction is weak if the report does not say how it could fail. Rejected and adjacent "
                    "sources are therefore preserved as audit material rather than discarded."
                ),
                "mechanism": (
                    "For each report-ready claim, the system links boundary sources to hard-negative challenges, then creates "
                    "a falsification plan with a falsification criterion, benchmark or task perturbation, expected failure "
                    "mode, and negative-result logging schema. The report certificate is h(c)=g(c) AND "
                    "counterexample_covered(c) AND falsification_plan_exists(c)."
                ),
                "role_evidence": (
                    "In the fresh five-topic pilot, falsification coverage matched the report-ready claim count for every "
                    "topic. Structural ablation of falsification and hard-negative components produced consistent quality "
                    "drops."
                ),
            },
            {
                "title": "3.4 Report rendering and advisor-facing artifacts",
                "motivation": (
                    "Advisor-facing ideation artifacts must expose why a claim appears in the report and why other material "
                    "was kept outside it."
                ),
                "mechanism": (
                    "The renderer separates report-ready findings from audit-only backlog, writes citations and evidence "
                    "appendices, exports claims, evidence, rejected sources, counterexample audits, and falsification plans, "
                    "and produces mentor review packets for human judgment."
                ),
                "role_evidence": (
                    "The pilot freeze validator checks fresh provenance, CUDA reranker use, required report sections, "
                    "report-ready claim counts, falsification coverage, and evaluator scores before a pilot is treated as "
                    "frozen."
                ),
            },
        ],
        "bounded_proposition": {
            "statement": (
                "Proposition 1 (audit certificate). If h(c)=1 for a claim c, then the generated artifact exposes an auditable "
                "evidence certificate for c, a source-role boundary with hard-negative context, and a falsification plan for "
                "negative-result logging."
            ),
            "proof_sketch": (
                "By definition, h(c)=g(c) requires the evidence ids, independent-paper support, source-role constraints, and "
                "report-body eligibility recorded by g(c). The counterexample_covered(c) condition links c to boundary or "
                "hard-negative sources, and falsification_plan_exists(c) requires the failure criterion, perturbation, "
                "expected failure mode, and logging schema. The artifact renderer writes these fields to the report and "
                "exports."
            ),
            "boundary": (
                "The proposition is not a truth, novelty, or publishability guarantee. It only states what the artifact makes "
                "inspectable."
            ),
        },
        "assumptions_or_missing_inputs": [
            "Final manuscript citations to prior AI-for-research and LLM-ideation systems are still needed.",
            "Human or blind strong-model review is still needed for novelty, usefulness, and feasibility.",
            "The public release plan for code, local index construction, and model weights should be stated before submission.",
        ],
    }


def experiment_section_draft_note() -> dict[str, Any]:
    return {
        "section": "Section 4 Experiments",
        "one_sentence_argument": (
            "The pilot experiments tested whether ScholarInsight's quality controls produced fresh, auditable reports across "
            "multiple topic families and whether removing retrieval or audit components weakened report-ready evidence."
        ),
        "setup": {
            "topics": [
                "004 RAG with Knowledge Graphs",
                "006 Mathematical Reasoning",
                "010 Causal Reasoning with LLMs",
                "011 Counterfactual Inference",
                "012 Multi-hop Reasoning on Graphs",
            ],
            "pipeline": (
                "All fresh pilots used the deterministic local pipeline, local paper artifacts, direct Hugging Face model "
                "loading, and CUDA bge-reranker provenance. No external LLM calls were used for the frozen pilot artifacts."
            ),
            "metrics": [
                "idea-quality score",
                "report-ready claim count",
                "falsification-plan coverage",
                "validator pass/fail status",
                "quality flags and evaluator flags",
                "source-purity and counterexample-audit availability",
            ],
            "artifact_boundary": (
                "Generated artifacts under data/quality_audits are local and ignored by git. The paper-facing code records "
                "how those artifacts were produced but does not push the local paper data."
            ),
        },
        "subsections": [
            {
                "title": "4.1 Experimental setup",
                "claim": (
                    "The evaluation used five topic families to test whether the source-role and claim-gating design "
                    "generalized beyond the RAG+KG debugging topic."
                ),
                "evidence": (
                    "The fresh pilot covered RAG+KG, mathematical reasoning, causal LLM reasoning, counterfactual inference, "
                    "and multi-hop graph reasoning. Each topic was validated for fresh provenance, non-replay status, and "
                    "CUDA reranker use."
                ),
            },
            {
                "title": "4.2 Fresh pilot freeze validation",
                "claim": (
                    "The current pipeline produced frozen local artifacts with nonzero report-ready claims and complete "
                    "falsification coverage for all five pilot topics."
                ),
                "evidence": (
                    "The fresh pilot table records per-topic evaluator scores, report-ready counts, and falsification "
                    "coverage. The conservative evaluator may flag generic synthesis claims for advisor novelty review "
                    "even when the freeze validator passes."
                ),
            },
            {
                "title": "4.3 Structural ablation",
                "claim": (
                    "Post-processing modules were not cosmetic: masking the claim gate, hard-negative audit, falsification, "
                    "or source-role fields reduced the structural quality proxy."
                ),
                "evidence": (
                    "The artifact-masked ablation table reports mean deltas for claim gate, hard-negative audit, "
                    "falsification, source-role, and experiment-framing variants. These deltas should be interpreted "
                    "as structural proxy evidence, not as a full runtime rerun."
                ),
            },
            {
                "title": "4.4 Runtime source-stage ablation",
                "claim": (
                    "Retrieval-time components affected both report-ready synthesis and the availability of counterexample "
                    "audit material."
                ),
                "evidence": (
                    "Removing the reranker reduced report-ready counts for 004 and 006 and introduced a non-reportable-role "
                    "flag in 011. Removing the source gate produced missing-counterexample-audit flags for 004, 006, 011, "
                    "and 012, while 010 remained stable under the tested source-stage ablations."
                ),
            },
            {
                "title": "4.5 Failure modes and remaining human review",
                "claim": (
                    "The current evidence supports an auditability claim, not a claim that the generated directions are novel, "
                    "useful, or ready for submission."
                ),
                "evidence": (
                    "The evaluation is a five-topic pilot with a structural evaluator. It has not yet run the 132-topic batch, "
                    "human novelty review, or blind strong-model usefulness review."
                ),
            },
        ],
        "claim_evidence_map": [
            {
                "claim": "Fresh artifacts are reproducible enough for advisor review.",
                "evidence": "pilot_freeze_validator pass status, provenance checks, mentor packets, and complete required report sections.",
                "status": "supported",
            },
            {
                "claim": "g(c) and h(c) improve auditability and falsifiability.",
                "evidence": "structural ablation deltas for claim gate, hard-negative audit, falsification, and source roles.",
                "status": "supported as a structural proxy",
            },
            {
                "claim": "The pipeline improves research novelty or usefulness.",
                "evidence": "human or blind strong-model review has not been run.",
                "status": "needs evidence",
            },
            {
                "claim": "The system generalizes to all 132 topics.",
                "evidence": "only five frozen pilot topics have been validated.",
                "status": "needs evidence",
            },
        ],
        "limitations": [
            "Only five topic families are frozen; the 132-topic batch is intentionally paused.",
            "The evaluator is a structural quality proxy, not a substitute for expert review.",
            "The ablation package combines artifact-masked structural ablation with deterministic source-stage runtime ablation; it is not yet a full end-to-end randomized experiment.",
            "The current evidence does not establish novelty, feasibility, or publication quality.",
        ],
    }


def render_markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# ScholarInsight Paper Experiment Packet",
        "",
        "This packet consolidates the current non-human evidence for the ScholarInsight quality loop. It should be treated as paper experiment material, not as a substitute for advisor/human novelty review.",
        "",
        "## Fresh CUDA Pilot",
        "",
        "| Topic | Pass | Score | Accepted | Evidence | Claims | Report-ready | Falsification plans | Device |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in packet["fresh_pilot"]:
        lines.append(
            f"| {row['topic_id']} {row['topic']} | {'yes' if row['passed'] else 'no'} "
            f"| {row['score']:.3f} | {row['accepted_count']} | {row['evidence_count']} "
            f"| {row['claim_count']} | {row['report_ready_count']} "
            f"| {row['falsification_plan_count']} | {row['reranker_device']} |"
        )
    lines.extend(
        [
            "",
            "## Structural Ablation Summary",
            "",
            "Artifact-masked ablation over the same fresh reports. This isolates post-processing and reporting modules but is not a runtime rerun.",
            "",
            "| Variant | Mean delta | Min delta | Max delta | Topics |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in packet["structural_ablation_summary"]:
        lines.append(
            f"| {row['variant']} | {row['mean_delta']:.3f} | {row['min_delta']:.3f} "
            f"| {row['max_delta']:.3f} | {row['topic_count']} |"
        )
    lines.extend(
        [
            "",
            "## Runtime Ablation Summary",
            "",
            "Deterministic reruns of retrieval/source-stage settings. No external LLM calls and no 132-topic batch.",
            "",
            "| Variant | Mean score delta | Min | Max | Mean report-ready delta | Flags |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in packet["runtime_ablation_summary"]:
        lines.append(
            f"| {row['variant']} | {row['mean_score_delta']:.3f} | {row['min_score_delta']:.3f} "
            f"| {row['max_score_delta']:.3f} | {row['mean_report_ready_delta']:.3f} "
            f"| {', '.join(row['flags']) or 'none'} |"
        )
    gate = packet["formal_gate"]
    lines.extend(
        [
            "",
            "## Formal Gate Note",
            "",
            f"- Claim gate: `{gate['claim_gate']}`",
            f"- Report certificate: `{gate['report_certificate']}`",
            "",
            "Guarantee:",
        ]
    )
    for item in gate["guarantee"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            f"Paper-safe claim: {gate['paper_safe_claim']}",
            "",
            f"Unsafe claim to avoid: {gate['paper_unsafe_claim']}",
            "",
            "## Paper Logic Skeleton",
            "",
            f"- Type: {packet['paper_skeleton']['paper_type']}",
            f"- Rationale: {packet['paper_skeleton']['positioning_rationale']}",
            "",
            "| Stage | Content |",
            "|---|---|",
        ]
    )
    for row in packet["paper_skeleton"]["thinking_template"]:
        lines.append(f"| {row['stage']} | {row['content']} |")
    lines.extend(
        [
            "",
            "### Methodology Outline",
            "",
        ]
    )
    for item in packet["paper_skeleton"]["methodology_outline"]:
        lines.append(f"- **{item['section']}**: {item['summary']}")
    lines.extend(
        [
            "",
            "### Self-consistency Checks",
            "",
        ]
    )
    for name, status in packet["paper_skeleton"]["self_consistency_checks"].items():
        lines.append(f"- {name}: {status}")
    lines.extend(
        [
            f"- Severity summary: {packet['paper_skeleton']['severity_summary']}",
            "",
            "## Introduction Outline",
            "",
            f"- Type: {packet['introduction_outline']['type_positioning']['type']}",
            f"- Rationale: {packet['introduction_outline']['type_positioning']['rationale']}",
            f"- Implication: {packet['introduction_outline']['type_positioning']['implication']}",
            "",
        ]
    )
    for paragraph in packet["introduction_outline"]["paragraphs"]:
        lines.extend(
            [
                f"### {paragraph['id']}: {paragraph['title']}",
                "",
                f"- Purpose: {paragraph['purpose']}",
            ]
        )
        if paragraph.get("running_example"):
            lines.append(f"- Running example: {paragraph['running_example']}")
        if paragraph.get("hard_constraints"):
            lines.append("- Hard constraints:")
            for item in paragraph["hard_constraints"]:
                lines.append(f"  - {item}")
        if paragraph.get("goal_sentence"):
            lines.append(f"- Goal sentence candidate: \"{paragraph['goal_sentence']}\"")
        if paragraph.get("challenge_to_module_mapping"):
            lines.append("- Challenge to module mapping:")
            for item in paragraph["challenge_to_module_mapping"]:
                lines.append(f"  - {item}")
        if paragraph.get("contributions"):
            lines.append("- Contributions:")
            for idx, item in enumerate(paragraph["contributions"], 1):
                lines.append(f"  {idx}. {item}")
        if paragraph.get("writing_points"):
            lines.append("- Writing points:")
            for item in paragraph["writing_points"]:
                lines.append(f"  - {item}")
        lines.append(f"- Gaps: {', '.join(paragraph.get('gaps') or ['none'])}")
        lines.append("")
    lines.extend(
        [
            "### Introduction Consistency",
            "",
        ]
    )
    for name, status in packet["introduction_outline"]["flowchart_consistency"].items():
        lines.append(f"- {name}: {status}")
    for name, status in packet["introduction_outline"]["integrity_gate"].items():
        lines.append(f"- {name}: {status}")
    lines.extend(
        [
            f"- Severity summary: {packet['introduction_outline']['severity_summary']}",
            "",
            "## Method Section Draft",
            "",
            f"- Section: {packet['method_section_draft']['section']}",
            f"- One-sentence argument: {packet['method_section_draft']['one_sentence_argument']}",
            "",
            "### Task Formulation",
            "",
        ]
    )
    for name, value in packet["method_section_draft"]["task_formulation"].items():
        lines.append(f"- {name}: {value}")
    lines.extend(
        [
            "",
            "### Notation",
            "",
        ]
    )
    for item in packet["method_section_draft"]["notation"]:
        lines.append(f"- {item}")
    lines.append("")
    for subsection in packet["method_section_draft"]["subsections"]:
        lines.extend(
            [
                f"### {subsection['title']}",
                "",
                f"- Motivation: {subsection['motivation']}",
                f"- Mechanism: {subsection['mechanism']}",
                f"- Evidence/role: {subsection['role_evidence']}",
                "",
            ]
        )
    proposition = packet["method_section_draft"]["bounded_proposition"]
    lines.extend(
        [
            "### Bounded Proposition",
            "",
            f"- Statement: {proposition['statement']}",
            f"- Proof sketch: {proposition['proof_sketch']}",
            f"- Boundary: {proposition['boundary']}",
            "",
            "### Method Assumptions or Missing Inputs",
            "",
        ]
    )
    for item in packet["method_section_draft"]["assumptions_or_missing_inputs"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Experiment Section Draft",
            "",
            f"- Section: {packet['experiment_section_draft']['section']}",
            f"- One-sentence argument: {packet['experiment_section_draft']['one_sentence_argument']}",
            "",
            "### Setup",
            "",
            f"- Topics: {', '.join(packet['experiment_section_draft']['setup']['topics'])}",
            f"- Pipeline: {packet['experiment_section_draft']['setup']['pipeline']}",
            f"- Metrics: {', '.join(packet['experiment_section_draft']['setup']['metrics'])}",
            f"- Artifact boundary: {packet['experiment_section_draft']['setup']['artifact_boundary']}",
            "",
        ]
    )
    for subsection in packet["experiment_section_draft"]["subsections"]:
        lines.extend(
            [
                f"### {subsection['title']}",
                "",
                f"- Claim: {subsection['claim']}",
                f"- Evidence: {subsection['evidence']}",
                "",
            ]
        )
    lines.extend(
        [
            "### Claim-Evidence Map",
            "",
        ]
    )
    for item in packet["experiment_section_draft"]["claim_evidence_map"]:
        lines.append(f"- Claim: {item['claim']} | Evidence: {item['evidence']} | Status: {item['status']}")
    lines.extend(
        [
            "",
            "### Experiment Limitations",
            "",
        ]
    )
    for item in packet["experiment_section_draft"]["limitations"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Remaining Non-mechanical Gap",
            "",
            "Human or strong-model blind review is still required for novelty, usefulness, feasibility, and writing quality. The current evaluator measures structural evidence quality only.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a paper-ready experiment packet from ScholarInsight pilot artifacts.")
    parser.add_argument("--fresh-root", required=True)
    parser.add_argument("--structural-ablation-json", required=True)
    parser.add_argument("--runtime-ablation-json", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    fresh_root = Path(args.fresh_root)
    structural = read_json(Path(args.structural_ablation_json))
    runtime = read_json(Path(args.runtime_ablation_json))
    packet = {
        "fresh_root": str(fresh_root),
        "structural_ablation_json": args.structural_ablation_json,
        "runtime_ablation_json": args.runtime_ablation_json,
        "fresh_pilot": fresh_pilot_rows(fresh_root),
        "structural_ablation_summary": structural_delta_summary(structural.get("items") or []),
        "runtime_ablation_summary": runtime_delta_summary(runtime.get("items") or []),
        "formal_gate": formal_gate_note(),
        "paper_skeleton": paper_skeleton_note(),
        "introduction_outline": introduction_outline_note(),
        "method_section_draft": method_section_draft_note(),
        "experiment_section_draft": experiment_section_draft_note(),
    }
    output_dir = Path(args.output_dir) if args.output_dir else fresh_root / "paper_experiment_packet"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "paper_experiment_packet.json", packet)
    write_text(output_dir / "paper_experiment_packet.md", render_markdown(packet))
    print(json.dumps({"output_dir": str(output_dir), "fresh_topics": len(packet["fresh_pilot"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
