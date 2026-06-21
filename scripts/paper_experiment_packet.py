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
        "paper_unsafe_claim": "Do not claim that ScholarInsight automatically generates AAAI-quality ideas or proves scientific novelty.",
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
    }
    output_dir = Path(args.output_dir) if args.output_dir else fresh_root / "paper_experiment_packet"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "paper_experiment_packet.json", packet)
    write_text(output_dir / "paper_experiment_packet.md", render_markdown(packet))
    print(json.dumps({"output_dir": str(output_dir), "fresh_topics": len(packet["fresh_pilot"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
