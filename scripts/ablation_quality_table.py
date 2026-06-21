from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.idea_quality_evaluator import (  # noqa: E402
    QUALITY_DIMENSIONS,
    evaluate_loaded,
    load_artifact_dir,
    read_json,
    read_text,
    split_evidence_appendix,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def topic_dirs(root: Path) -> list[Path]:
    if (root / "exports" / "claims.json").exists():
        return [root]
    return sorted(
        item
        for item in root.iterdir()
        if item.is_dir() and (item / "exports" / "claims.json").exists()
    )


def raw_artifact(path: Path, label: str) -> dict[str, Any]:
    exports = path / "exports"
    reports = path / "reports"
    report_md = read_text(reports / "report.md")
    report_body_text, evidence_appendix_text = split_evidence_appendix(report_md)
    report_body_text = "\n\n".join(
        text
        for text in [
            report_body_text,
            read_text(reports / "executive_summary.md"),
            read_text(reports / "methodology.md"),
        ]
        if text
    )
    return {
        "label": label,
        "path": str(path),
        "kind": "artifact_ablation_topic",
        "report_text": "\n\n".join(text for text in [report_body_text, evidence_appendix_text] if text),
        "report_body_text": report_body_text,
        "evidence_appendix_text": evidence_appendix_text,
        "claims": read_json(exports / "claims.json") if (exports / "claims.json").exists() else [],
        "counterexample_audit": read_json(exports / "counterexample_audit.json")
        if (exports / "counterexample_audit.json").exists()
        else [],
        "falsification_plan": read_json(exports / "falsification_plan.json")
        if (exports / "falsification_plan.json").exists()
        else [],
        "observability": read_json(exports / "observability.json") if (exports / "observability.json").exists() else {},
        "matrix": read_json(exports / "matrix.json") if (exports / "matrix.json").exists() else {},
        "recommendations": read_json(exports / "recommendations.json")
        if (exports / "recommendations.json").exists()
        else [],
        "summary": read_json(path / "summary.json") if (path / "summary.json").exists() else {},
        "evidence_file_count": 1 if (path / "evidence" / "evidence.json").exists() else 0,
        "source_file_count": 1 if (path / "sources" / "selected_sources.json").exists() else 0,
    }


def remove_sections(text: str, section_markers: list[str]) -> str:
    lines = text.splitlines()
    output: list[str] = []
    skip = False
    for line in lines:
        if line.startswith("## "):
            skip = any(line.strip() == marker for marker in section_markers)
            if skip:
                continue
        if not skip:
            output.append(line)
    return "\n".join(output)


def apply_variant(base: dict[str, Any], variant: str) -> dict[str, Any]:
    data = copy.deepcopy(base)
    data["label"] = f"{base['label']}::{variant}"
    if variant == "full":
        return data
    if variant == "minus_source_roles":
        data["summary"]["accepted_subtypes"] = {}
        for claim in data["claims"]:
            claim["supporting_source_subtypes"] = []
            claim["supporting_source_subtype_counts"] = {}
            claim["supporting_source_subtype_paper_counts"] = {}
    elif variant == "minus_claim_gate":
        data["summary"]["report_ready_count"] = 0
        for claim in data["claims"]:
            claim["evidence_support_level"] = "weak"
            claim["source_paper_count"] = min(int(claim.get("source_paper_count") or 0), 1)
            claim["claim_type"] = "single_paper_observation"
            claim["final_wording"] = f"作为单论文观察，{claim.get('final_wording') or claim.get('claim') or ''}"
        data["report_body_text"] = remove_sections(
            data["report_body_text"],
            [
                "## 形式化证据门控与数学背书",
                "## 可投稿研究命题与实验化路径",
            ],
        )
    elif variant == "minus_hard_negative":
        data["counterexample_audit"] = []
        data["report_body_text"] = remove_sections(
            data["report_body_text"],
            ["## 反例与负结果审计"],
        )
    elif variant == "minus_falsification":
        data["falsification_plan"] = []
        data["report_body_text"] = remove_sections(
            data["report_body_text"],
            ["## 可证伪实验设计与负结果记录"],
        )
    elif variant == "minus_experiment_framing":
        data["recommendations"] = []
        for row in data["falsification_plan"]:
            row["benchmark_or_task_perturbation"] = ""
        for row in data["recommendations"]:
            row["next_steps"] = []
        data["report_body_text"] = remove_sections(
            data["report_body_text"],
            [
                "## 可投稿研究命题与实验化路径",
                "## 可检验研究假设与学术背书",
                "## 研究机会与下一步",
            ],
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
    data["report_text"] = "\n\n".join(
        text for text in [data.get("report_body_text", ""), data.get("evidence_appendix_text", "")] if text
    )
    return data


def evaluate_topic(path: Path) -> list[dict[str, Any]]:
    label = path.name
    base = raw_artifact(path, label)
    rows = []
    for variant in [
        "full",
        "minus_source_roles",
        "minus_claim_gate",
        "minus_hard_negative",
        "minus_falsification",
        "minus_experiment_framing",
    ]:
        item = evaluate_loaded(apply_variant(base, variant))
        rows.append(
            {
                "topic_id": label,
                "topic": base["summary"].get("topic") or label,
                "variant": variant,
                "overall_score": item["overall_score"],
                "verdict": item["verdict"],
                "scores": item["scores"],
                "flags": item["flags"],
            }
        )
    return rows


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# ScholarInsight Structural Ablation Table",
        "",
        "This table is an artifact-masked ablation: each variant starts from the same fresh artifact and removes one quality component before re-running the structural evaluator. It is not a runtime rerun ablation.",
        "",
        "| Topic | Variant | Overall | Source | Evidence | Claims | Falsify | Experiment | Novelty proxy | Flags |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        scores = row["scores"]
        lines.append(
            f"| {row['topic_id']} {row['topic']} | {row['variant']} | {row['overall_score']:.3f} "
            f"| {scores['source_relevance']:.3f} | {scores['evidence_grounding']:.3f} "
            f"| {scores['claim_validity']:.3f} | {scores['falsifiability']:.3f} "
            f"| {scores['experimentability']:.3f} | {scores['novelty_proxy']:.3f} "
            f"| {', '.join(row['flags']) or 'none'} |"
        )
    lines.append("")
    lines.extend(["## Deltas vs Full", "", "| Topic | Variant | Delta |", "|---|---|---:|"])
    full_by_topic = {
        row["topic_id"]: row["overall_score"]
        for row in rows
        if row["variant"] == "full"
    }
    for row in rows:
        if row["variant"] == "full":
            continue
        delta = row["overall_score"] - full_by_topic[row["topic_id"]]
        lines.append(f"| {row['topic_id']} | {row['variant']} | {delta:.3f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build structural ablation table for fresh ScholarInsight artifacts.")
    parser.add_argument("artifact_root")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    root = Path(args.artifact_root)
    output_dir = Path(args.output_dir) if args.output_dir else root
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for path in topic_dirs(root):
        rows.extend(evaluate_topic(path))
    result = {
        "artifact_root": str(root),
        "ablation_type": "artifact_masked_structural",
        "items": rows,
    }
    write_json(output_dir / "ablation_quality_table.json", result)
    write_text(output_dir / "ablation_quality_table.md", render_markdown(rows))
    print(json.dumps({"items": len(rows), "output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
