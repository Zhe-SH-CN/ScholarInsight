from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.idea_quality_evaluator import load_artifact_dir  # noqa: E402


REQUIRED_FILES = [
    "summary.json",
    "provenance.json",
    "exports/claims.json",
    "exports/counterexample_audit.json",
    "exports/falsification_plan.json",
    "exports/matrix.json",
    "exports/observability.json",
    "reports/report.md",
    "reports/executive_summary.md",
    "reports/methodology.md",
]

REQUIRED_REPORT_SECTIONS = [
    "## 形式化证据门控与数学背书",
    "## 可投稿研究命题与实验化路径",
    "## 反例与负结果审计",
    "## 可证伪实验设计与负结果记录",
    "## Evidence 附录",
]

MECHANICAL_PATTERNS = [
    r"相关论文文本中出现了可核验表述",
    r"FROM ABSTRACT TO CONTEXTUAL",
    r"Published as a conference paper",
    r"Findings of the Association",
    r"Proceedings of the",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


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


def report_ready_claims(claims: list[dict[str, Any]], summary: dict[str, Any]) -> list[dict[str, Any]]:
    ready_ids = {row.get("claim_id") for row in summary.get("report_ready_claims") or []}
    if ready_ids:
        return [claim for claim in claims if claim.get("claim_id") in ready_ids]
    return [
        claim
        for claim in claims
        if claim.get("is_report_ready")
        or (
            claim.get("verification_status") == "verified"
            and claim.get("risk_level") != "high"
            and claim.get("claim_type") in {"comparative", "cross_role_contrast"}
            and int(claim.get("source_paper_count") or 0) >= 2
            and claim.get("evidence_support_level") == "strong"
        )
    ]


def validate_artifact(path: Path, require_cuda: bool) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    missing = [rel for rel in REQUIRED_FILES if not (path / rel).exists()]
    if missing:
        errors.extend([f"missing_file:{rel}" for rel in missing])

    summary = read_json(path / "summary.json") if (path / "summary.json").exists() else {}
    provenance = read_json(path / "provenance.json") if (path / "provenance.json").exists() else {}
    claims = read_json(path / "exports" / "claims.json") if (path / "exports" / "claims.json").exists() else []
    counterexamples = (
        read_json(path / "exports" / "counterexample_audit.json")
        if (path / "exports" / "counterexample_audit.json").exists()
        else []
    )
    falsification = (
        read_json(path / "exports" / "falsification_plan.json")
        if (path / "exports" / "falsification_plan.json").exists()
        else []
    )
    report = read_text(path / "reports" / "report.md")
    for section in REQUIRED_REPORT_SECTIONS:
        if section not in report:
            errors.append(f"missing_report_section:{section}")
    for pattern in MECHANICAL_PATTERNS:
        if re.search(pattern, report, flags=re.IGNORECASE):
            errors.append(f"mechanical_text_pattern:{pattern}")

    artifact_kind = summary.get("artifact_kind") or provenance.get("artifact_kind")
    if artifact_kind != "fresh_deterministic_local":
        errors.append(f"not_fresh_deterministic_local:{artifact_kind or 'missing'}")
    if summary.get("replay_derived_from") or summary.get("replay_derivation"):
        errors.append("replay_marker_present")
    if provenance.get("external_llm_calls"):
        errors.append("external_llm_calls_present")
    if provenance.get("reranker_error"):
        errors.append(f"reranker_error:{provenance.get('reranker_error')}")
    loaded_device = str(provenance.get("reranker_device_loaded") or "")
    requested_device = str(provenance.get("reranker_device_requested") or "")
    if require_cuda:
        if requested_device.lower() != "cuda":
            errors.append(f"reranker_device_not_requested_cuda:{requested_device or 'missing'}")
        if not loaded_device.startswith("cuda"):
            errors.append(f"reranker_device_not_loaded_cuda:{loaded_device or 'missing'}")

    ready = report_ready_claims(claims, summary)
    ready_ids = {claim.get("claim_id") for claim in ready if claim.get("claim_id")}
    if not ready:
        errors.append("no_report_ready_claims")
    falsification_ids = {row.get("target_claim_id") for row in falsification if row.get("target_claim_id")}
    missing_falsification = sorted(ready_ids - falsification_ids)
    if missing_falsification:
        errors.append(f"falsification_missing:{','.join(missing_falsification)}")
    visible_counterexample_ids = {
        row.get("target_claim_id")
        for row in counterexamples
        if row.get("target_claim_id") and row.get("report_visible", True)
    }
    missing_counterexample = sorted(ready_ids - visible_counterexample_ids)
    if missing_counterexample:
        warnings.append(f"visible_counterexample_missing:{','.join(missing_counterexample)}")

    evaluation = load_artifact_dir(path, path.name)
    if evaluation["flags"]:
        warnings.extend([f"evaluator_flag:{flag}" for flag in evaluation["flags"]])
    if evaluation["overall_score"] < 0.75:
        errors.append(f"low_idea_quality_score:{evaluation['overall_score']:.3f}")
    return {
        "topic_dir": str(path),
        "topic_id": summary.get("topic_id") or path.name,
        "topic": summary.get("topic") or provenance.get("topic") or path.name,
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "artifact_kind": artifact_kind,
            "git_commit": provenance.get("git_commit"),
            "reranker_model": provenance.get("reranker_model"),
            "reranker_device_requested": requested_device,
            "reranker_device_loaded": loaded_device,
            "accepted_count": summary.get("accepted_count"),
            "rejected_count": summary.get("rejected_count"),
            "evidence_count": summary.get("evidence_count"),
            "claim_count": summary.get("claim_count"),
            "verified_claim_count": summary.get("verified_claim_count"),
            "report_ready_count": len(ready),
            "counterexample_report_visible_count": summary.get("counterexample_report_visible_count"),
            "falsification_plan_count": len(falsification),
            "idea_quality_score": evaluation["overall_score"],
            "idea_quality_verdict": evaluation["verdict"],
        },
        "evaluation": evaluation,
    }


def compact_claim_text(claim: dict[str, Any], limit: int = 360) -> str:
    text = str(claim.get("final_wording") or claim.get("claim") or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def mentor_packet(path: Path, validation: dict[str, Any]) -> str:
    summary = read_json(path / "summary.json")
    claims = read_json(path / "exports" / "claims.json")
    ready = report_ready_claims(claims, summary)
    counterexamples = read_json(path / "exports" / "counterexample_audit.json")
    falsification = read_json(path / "exports" / "falsification_plan.json")
    accepted = read_json(path / "sources" / "accepted_sources.json") if (path / "sources" / "accepted_sources.json").exists() else []
    rejected = read_json(path / "sources" / "rejected_sources.json") if (path / "sources" / "rejected_sources.json").exists() else []
    falsification_by_claim = {row.get("target_claim_id"): row for row in falsification}
    cex_by_claim: dict[str, list[dict[str, Any]]] = {}
    for row in counterexamples:
        if row.get("report_visible", True):
            cex_by_claim.setdefault(str(row.get("target_claim_id")), []).append(row)

    lines = [
        f"# Mentor Review Packet: {summary.get('topic', path.name)}",
        "",
        "## Freeze Validation",
        "",
        f"- Status: {'PASS' if validation['passed'] else 'FAIL'}",
        f"- Idea quality: {validation['summary']['idea_quality_score']:.3f} ({validation['summary']['idea_quality_verdict']})",
        f"- Git commit: `{validation['summary'].get('git_commit')}`",
        f"- Reranker: `{validation['summary'].get('reranker_model')}` on `{validation['summary'].get('reranker_device_loaded')}`",
        f"- Errors: {', '.join(validation['errors']) or 'none'}",
        f"- Warnings: {', '.join(validation['warnings']) or 'none'}",
        "",
        "## Pipeline Counts",
        "",
        f"- Accepted/rejected sources: {summary.get('accepted_count')} / {summary.get('rejected_count')}",
        f"- Evidence / claims / report-ready claims: {summary.get('evidence_count')} / {summary.get('claim_count')} / {validation['summary']['report_ready_count']}",
        f"- Falsification coverage: {summary.get('falsification_covered_report_ready_claim_count')}/{summary.get('report_ready_count')}",
        "",
        "## Advisor Review Readiness",
        "",
        "- Freeze PASS means the artifact is auditable and falsification-aware; it is not a novelty, usefulness, or publishability guarantee.",
        f"- Generic synthesis claims needing advisor novelty review: {validation['evaluation']['metrics'].get('generic_report_ready_like_claim_count', 0)}",
        "- Required human decision: whether each report-ready claim is a useful research direction, a generic survey observation, off-topic, or not novel enough.",
        "",
        "## Report-ready Claims",
        "",
    ]
    if not ready:
        lines.append("No report-ready claims.")
    for idx, claim in enumerate(ready, 1):
        claim_id = str(claim.get("claim_id"))
        plan = falsification_by_claim.get(claim_id, {})
        hard_negatives = cex_by_claim.get(claim_id, [])
        lines.extend(
            [
                f"### Claim {idx}: {claim.get('dimension_label') or claim.get('dimension')}",
                "",
                compact_claim_text(claim),
                "",
                f"- Type: `{claim.get('claim_type')}`",
                f"- Independent papers: {claim.get('source_paper_count')}",
                f"- Source-role paper counts: `{claim.get('supporting_source_subtype_paper_counts') or {}}`",
                f"- Falsification criterion: {plan.get('falsification_criterion') or 'missing'}",
                f"- Benchmark/task perturbation: {plan.get('benchmark_or_task_perturbation') or 'missing'}",
                f"- Hard-negative sources: {len(hard_negatives)}",
                "",
            ]
        )
        for row in hard_negatives[:2]:
            lines.append(f"  - {row.get('source_title')}: {row.get('boundary_challenge')}")
        lines.append("")

    lines.extend(["## Top Accepted Sources", ""])
    for row in accepted[:8]:
        lines.append(
            f"- `{row.get('source_subtype')}` {row.get('title')} "
            f"(rel={float(row.get('relevance_score') or 0):.3f}, rerank={row.get('reranker_score')})"
        )
    lines.extend(["", "## Top Rejected / Boundary Sources", ""])
    for row in rejected[:8]:
        lines.append(
            f"- `{row.get('source_subtype')}` {row.get('title')} "
            f"(reason={row.get('rejection_reason') or 'not selected'})"
        )
    lines.extend(
        [
            "",
            "## Human Review Form",
            "",
            "| Criterion | Score 1-5 | Notes |",
            "|---|---:|---|",
            "| Novelty |  |  |",
            "| Usefulness |  |  |",
            "| Feasibility |  |  |",
            "| Evidence alignment |  |  |",
            "| Writing clarity |  |  |",
            "",
        ]
    )
    return "\n".join(lines)


def render_validation_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Pilot Freeze Validation",
        "",
        f"- Overall pass: {'PASS' if result['passed'] else 'FAIL'}",
        f"- Artifact root: `{result['artifact_root']}`",
        "",
        "| Topic | Pass | Score | Report-ready | CUDA | Errors | Warnings |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for item in result["items"]:
        summary = item["summary"]
        lines.append(
            f"| {item['topic_id']} {item['topic']} | {'PASS' if item['passed'] else 'FAIL'} "
            f"| {summary['idea_quality_score']:.3f} | {summary['report_ready_count']} "
            f"| {summary.get('reranker_device_loaded') or 'missing'} "
            f"| {', '.join(item['errors']) or 'none'} | {', '.join(item['warnings']) or 'none'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate fresh ScholarInsight pilot artifacts and build mentor packets.")
    parser.add_argument("artifact_root")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--no-require-cuda", action="store_true")
    args = parser.parse_args()

    root = Path(args.artifact_root)
    output_dir = Path(args.output_dir) if args.output_dir else root
    output_dir.mkdir(parents=True, exist_ok=True)
    items = [validate_artifact(path, require_cuda=not args.no_require_cuda) for path in topic_dirs(root)]
    result = {
        "artifact_root": str(root),
        "passed": all(item["passed"] for item in items),
        "items": items,
    }
    write_json(output_dir / "pilot_freeze_validation.json", result)
    write_text(output_dir / "pilot_freeze_validation.md", render_validation_markdown(result))
    for item in items:
        path = Path(item["topic_dir"])
        write_text(path / "mentor_review_packet.md", mentor_packet(path, item))
    print(json.dumps({"passed": result["passed"], "items": len(items), "output_dir": str(output_dir)}, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
