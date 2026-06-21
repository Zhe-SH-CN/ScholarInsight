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
    }
    output_dir = Path(args.output_dir) if args.output_dir else fresh_root / "paper_experiment_packet"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "paper_experiment_packet.json", packet)
    write_text(output_dir / "paper_experiment_packet.md", render_markdown(packet))
    print(json.dumps({"output_dir": str(output_dir), "fresh_topics": len(packet["fresh_pilot"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
