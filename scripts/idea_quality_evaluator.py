from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


QUALITY_DIMENSIONS = [
    "source_relevance",
    "evidence_grounding",
    "claim_validity",
    "falsifiability",
    "experimentability",
    "novelty_proxy",
]


PROTOCOL = {
    "source_relevance": [
        "typed source roles or accepted-source subtype summary",
        "rejected/counterexample audit evidence for boundary control",
        "low metadata-noise/reference-title artifact rate",
        "absence of source-drift quality flags",
    ],
    "evidence_grounding": [
        "claims bind concrete supporting evidence ids",
        "strong multi-paper/report-ready support is present",
        "evidence appendix or structured evidence artifacts are available",
        "report uses explicit citations rather than unsupported prose",
    ],
    "claim_validity": [
        "verified, low-risk claims dominate the usable claim set",
        "report-ready claims are separated from audit-only/backlog claims",
        "challenged/high-risk claims are not treated as body conclusions",
        "claim pass rate is not inflated by sample-limited observations",
    ],
    "falsifiability": [
        "report-ready claims have falsification criteria",
        "hard-negative/counterexample rows challenge claim boundaries",
        "report text exposes falsification or scope-narrowing conditions",
        "negative-result logging schema is machine-readable",
    ],
    "experimentability": [
        "claims are converted into hypothesis/evaluation-plan language",
        "benchmark/task perturbations are specified",
        "recommendations include concrete next steps",
        "formal evidence gate or certificate constrains what enters experiments",
    ],
    "novelty_proxy": [
        "comparative or cross-role report-ready claims are present",
        "multiple source roles/dimensions support the synthesis",
        "hard negatives define a nontrivial boundary, not just positive support",
        "problem/method/boundary framing is explicit; literature novelty remains human-audited",
    ],
}


BAD_REFERENCE_PATTERNS = [
    re.compile(r"\b000\b", re.IGNORECASE),
    re.compile(r"proceedings of the", re.IGNORECASE),
    re.compile(r"findings of the association", re.IGNORECASE),
    re.compile(r"published as a conference paper", re.IGNORECASE),
    re.compile(r"^abstract$", re.IGNORECASE),
    re.compile(r"^references$", re.IGNORECASE),
]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if math.isnan(value):
        return low
    return round(max(low, min(high, value)), 3)


def ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def load_artifact_dir(path: Path, label: str) -> dict[str, Any]:
    exports = path / "exports"
    reports = path / "reports"
    report_text = "\n\n".join(
        text
        for text in [
            read_text(reports / "report.md"),
            read_text(reports / "executive_summary.md"),
            read_text(reports / "methodology.md"),
        ]
        if text
    )
    if (exports / "claims.json").exists():
        claims = read_json(exports / "claims.json")
    else:
        claims = read_jsonl(path / "claims" / "_index.jsonl")
    data = {
        "label": label,
        "path": str(path),
        "kind": "artifact_dir",
        "report_text": report_text,
        "claims": claims,
        "counterexample_audit": read_json(exports / "counterexample_audit.json")
        if (exports / "counterexample_audit.json").exists()
        else [],
        "falsification_plan": read_json(exports / "falsification_plan.json")
        if (exports / "falsification_plan.json").exists()
        else [],
        "observability": read_json(exports / "observability.json")
        if (exports / "observability.json").exists()
        else {},
        "matrix": read_json(exports / "matrix.json")
        if (exports / "matrix.json").exists()
        else {},
        "recommendations": read_json(exports / "recommendations.json")
        if (exports / "recommendations.json").exists()
        else [],
        "summary": read_json(path / "summary.json") if (path / "summary.json").exists() else {},
        "evidence_file_count": count_files(path / "evidence", "*.json"),
        "source_file_count": count_files(path / "sources", "*.json"),
    }
    return evaluate_loaded(data)


def load_audit_json(path: Path, label_prefix: str) -> list[dict[str, Any]]:
    payload = read_json(path)
    items: list[dict[str, Any]] = []
    for topic_id, topic in sorted((payload.get("topics") or {}).items()):
        label = f"{label_prefix}_{topic_id}"
        claims = topic.get("claims") or []
        data = {
            "label": label,
            "path": f"{path}#{topic_id}",
            "kind": "local_quality_audit_topic",
            "report_text": "",
            "claims": claims,
            "counterexample_audit": topic.get("counterexample_audit") or [],
            "falsification_plan": topic.get("falsification_plan") or [],
            "observability": {},
            "matrix": {},
            "recommendations": [],
            "summary": {
                **(topic.get("source_summary") or {}),
                **(topic.get("deterministic_summary") or {}),
                **(topic.get("quality_diagnostics") or {}),
                "counterexample_audit_summary": topic.get("counterexample_audit_summary") or {},
                "falsification_plan_summary": topic.get("falsification_plan_summary") or {},
                "topic": topic.get("topic", topic_id),
            },
            "evidence_file_count": int((topic.get("deterministic_summary") or {}).get("evidence_count") or 0),
            "source_file_count": int((topic.get("source_summary") or {}).get("selected_count") or 0),
        }
        items.append(evaluate_loaded(data))
    return items


def count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.glob(pattern) if item.is_file())


def evaluate_loaded(data: dict[str, Any]) -> dict[str, Any]:
    claims = normalize_list(data.get("claims"))
    counterexamples = normalize_list(data.get("counterexample_audit"))
    falsification = normalize_list(data.get("falsification_plan"))
    observability = data.get("observability") or {}
    matrix = data.get("matrix") or {}
    summary = data.get("summary") or {}
    report_text = data.get("report_text") or ""

    metrics = collect_metrics(
        claims=claims,
        counterexamples=counterexamples,
        falsification=falsification,
        observability=observability,
        matrix=matrix,
        summary=summary,
        report_text=report_text,
        evidence_file_count=int(data.get("evidence_file_count") or 0),
        source_file_count=int(data.get("source_file_count") or 0),
        recommendations=normalize_list(data.get("recommendations")),
    )
    scores = score_metrics(metrics, report_text)
    overall = clamp(sum(scores.values()) / len(QUALITY_DIMENSIONS))
    flags = flags_for(metrics, scores)
    return {
        "label": data["label"],
        "path": data["path"],
        "kind": data["kind"],
        "overall_score": overall,
        "verdict": verdict(overall, flags),
        "scores": scores,
        "metrics": metrics,
        "flags": flags,
        "novelty_note": "Novelty proxy is structural only; human literature review is still required.",
    }


def normalize_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def collect_metrics(
    *,
    claims: list[dict[str, Any]],
    counterexamples: list[dict[str, Any]],
    falsification: list[dict[str, Any]],
    observability: dict[str, Any],
    matrix: dict[str, Any],
    summary: dict[str, Any],
    report_text: str,
    evidence_file_count: int,
    source_file_count: int,
    recommendations: list[dict[str, Any]],
) -> dict[str, Any]:
    claim_count = len(claims)
    verified = [claim for claim in claims if claim.get("verification_status") == "verified"]
    low_risk_verified = [
        claim for claim in verified if claim.get("risk_level", "medium") != "high"
    ]
    high_risk = [claim for claim in claims if claim.get("risk_level") == "high"]
    supported = [claim for claim in claims if claim.get("supporting_evidence_ids")]
    strong_support = [
        claim for claim in claims
        if claim.get("evidence_support_level") == "strong"
        or int(claim.get("source_paper_count") or 0) >= 2
    ]
    report_ready_count = int(
        summary.get("report_ready_count")
        or summary.get("report_ready_verified_count")
        or (summary.get("falsification_plan_summary") or {}).get("report_ready_claim_count")
        or 0
    )
    if report_ready_count == 0:
        report_ready_count = len(
            [
                claim for claim in claims
                if claim.get("verification_status") == "verified"
                and claim.get("risk_level", "medium") != "high"
                and claim.get("evidence_support_level") == "strong"
                and int(claim.get("source_paper_count") or 0) >= 2
                and "不单独构成领域趋势" not in (claim.get("claim") or "")
            ]
        )

    source_subtypes = Counter()
    for claim in claims:
        for subtype, count in (claim.get("supporting_source_subtype_paper_counts") or {}).items():
            source_subtypes[subtype] += int(count or 0)
        for subtype in claim.get("supporting_source_subtypes") or []:
            source_subtypes[subtype] += 1
    accepted_subtypes = summary.get("accepted_subtypes") or {}
    source_subtypes.update({k: int(v or 0) for k, v in accepted_subtypes.items()})

    citations = re.findall(r"\[[0-9]+\]", report_text)
    words = re.findall(r"\S+", report_text)
    bad_reference_count = count_bad_references(report_text, matrix)
    reference_count = max(1, len(re.findall(r"^\[[0-9]+\]", report_text, flags=re.MULTILINE)))
    counterexample_visible = len([row for row in counterexamples if row.get("report_visible", True)])
    hard_negative_count = len(
        [
            row for row in counterexamples
            if row.get("counterexample_type") == "hard_negative_boundary"
        ]
    )
    falsification_claim_ids = {row.get("target_claim_id") for row in falsification if row.get("target_claim_id")}
    counterexample_claim_ids = {row.get("target_claim_id") for row in counterexamples if row.get("target_claim_id")}
    claim_ids = {claim.get("claim_id") for claim in claims if claim.get("claim_id")}
    target_claim_denominator = max(1, report_ready_count or len(low_risk_verified) or len(claim_ids))
    quality_flags = set(summary.get("quality_flags") or [])
    quality_flags.update((summary.get("quality_diagnostics") or {}).get("quality_flags") or [])
    matrix_dimensions = matrix.get("dimensions") or []
    matrix_cells = matrix.get("cells") or []
    strong_or_partial_cells = [
        cell for cell in matrix_cells if cell.get("status") in {"strong", "partial"}
    ]
    next_steps = sum(len(item.get("next_steps") or []) for item in recommendations)
    structured_benchmark = any(row.get("benchmark_or_task_perturbation") for row in falsification)
    structured_formal_gate = any(row.get("evidence_certificate") for row in falsification)
    structured_negative_logging = any(row.get("negative_result_logging_schema") for row in falsification)
    return {
        "claim_count": claim_count,
        "verified_claim_count": len(verified),
        "low_risk_verified_claim_count": len(low_risk_verified),
        "high_risk_claim_count": len(high_risk),
        "supported_claim_count": len(supported),
        "strong_support_claim_count": len(strong_support),
        "report_ready_claim_count": report_ready_count,
        "counterexample_count": len(counterexamples),
        "visible_counterexample_count": counterexample_visible,
        "hard_negative_count": hard_negative_count,
        "falsification_plan_count": len(falsification),
        "falsification_claim_coverage": ratio(len(falsification_claim_ids), target_claim_denominator),
        "counterexample_claim_coverage": ratio(len(counterexample_claim_ids), target_claim_denominator),
        "source_role_count": len([role for role in source_subtypes if role and role != "unclassified"]),
        "unclassified_source_role_count": int(source_subtypes.get("unclassified", 0)),
        "source_file_count": source_file_count,
        "evidence_file_count": evidence_file_count,
        "observability_claim_pass_rate": float(observability.get("claim_pass_rate") or 0),
        "observability_report_confidence": float(observability.get("report_confidence") or 0),
        "observability_evidence_coverage": float(observability.get("evidence_coverage_score") or 0),
        "citation_count": len(citations),
        "word_count": len(words),
        "citation_density_per_1000_words": round(1000 * len(citations) / max(1, len(words)), 3),
        "bad_reference_count": bad_reference_count,
        "bad_reference_ratio": round(bad_reference_count / reference_count, 3),
        "quality_flags": sorted(quality_flags),
        "dimension_count": len(matrix_dimensions),
        "strong_or_partial_matrix_cell_count": len(strong_or_partial_cells),
        "comparative_claim_count": len(
            [claim for claim in claims if claim.get("claim_type") in {"comparative", "cross_role_contrast"}]
        ),
        "cross_role_claim_count": len([claim for claim in claims if claim.get("claim_type") == "cross_role_contrast"]),
        "recommendation_next_step_count": next_steps,
        "has_formal_gate_text": structured_formal_gate or contains_any(report_text, ["g(c)", "evidence_certificate", "形式化证据门控"]),
        "has_falsification_text": bool(falsification) or contains_any(report_text, ["falsification", "可证伪", "推翻条件", "scope_narrowing"]),
        "has_negative_logging_text": structured_negative_logging or contains_any(report_text, ["failure_observation", "negative-result", "负结果记录"]),
        "has_hypothesis_text": contains_any(report_text, ["研究假设", "hypothesis", "实验化验证"]),
        "has_benchmark_text": structured_benchmark or contains_any(report_text, ["benchmark", "任务切片", "task perturbation", "评测协议"]),
        "has_backlog_separation_text": contains_any(report_text, ["待验证观察", "backlog", "audit-only", "暂不进入主结论"]),
        "has_problem_boundary_text": contains_any(report_text, ["问题定义", "边界条件", "scope", "贡献边界"]),
    }


def contains_any(text: str, needles: list[str]) -> bool:
    lower = text.lower()
    return any(needle.lower() in lower for needle in needles)


def count_bad_references(report_text: str, matrix: dict[str, Any]) -> int:
    candidates: list[str] = []
    in_references = False
    for line in report_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## ") and "参考" in stripped:
            in_references = True
            continue
        if in_references and stripped.startswith("["):
            candidates.append(stripped)
    candidates.extend(str(item) for item in matrix.get("papers") or [])
    return sum(1 for item in candidates if is_bad_reference(item))


def is_bad_reference(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return any(pattern.search(stripped) for pattern in BAD_REFERENCE_PATTERNS)


def score_metrics(metrics: dict[str, Any], report_text: str) -> dict[str, float]:
    source_relevance = 0.0
    source_relevance += 0.25 if metrics["source_role_count"] > 0 else 0.0
    source_relevance += 0.25 if metrics["counterexample_count"] > 0 else 0.0
    source_relevance += 0.25 * (1 - min(1.0, metrics["bad_reference_ratio"]))
    source_relevance += 0.25 if not source_drift_flags(metrics) else 0.05

    evidence_grounding = 0.0
    evidence_grounding += 0.30 * ratio(metrics["supported_claim_count"], metrics["claim_count"])
    evidence_grounding += 0.30 * ratio(metrics["strong_support_claim_count"], max(1, metrics["claim_count"]))
    evidence_grounding += 0.20 if metrics["evidence_file_count"] > 0 or "Evidence 附录" in report_text else 0.0
    evidence_grounding += 0.20 * min(1.0, metrics["citation_density_per_1000_words"] / 12)

    claim_validity = 0.0
    claim_validity += 0.40 * ratio(metrics["low_risk_verified_claim_count"], metrics["claim_count"])
    claim_validity += 0.25 * ratio(metrics["report_ready_claim_count"], max(1, metrics["claim_count"]))
    claim_validity += 0.20 if metrics["has_backlog_separation_text"] else 0.0
    claim_validity += 0.15 * (1 - ratio(metrics["high_risk_claim_count"], metrics["claim_count"]))

    falsifiability = 0.0
    falsifiability += 0.35 * min(1.0, metrics["falsification_claim_coverage"])
    falsifiability += 0.25 * min(1.0, metrics["counterexample_claim_coverage"])
    falsifiability += 0.20 if metrics["has_falsification_text"] else 0.0
    falsifiability += 0.20 if metrics["has_negative_logging_text"] else 0.0

    experimentability = 0.0
    experimentability += 0.25 if metrics["has_hypothesis_text"] else 0.0
    experimentability += 0.25 if metrics["has_benchmark_text"] else 0.0
    experimentability += 0.25 * min(1.0, metrics["recommendation_next_step_count"] / 6)
    experimentability += 0.25 if metrics["has_formal_gate_text"] else 0.0

    novelty_proxy = 0.0
    novelty_proxy += 0.30 * ratio(metrics["comparative_claim_count"], metrics["claim_count"])
    novelty_proxy += 0.25 * min(1.0, metrics["source_role_count"] / 2)
    novelty_proxy += 0.25 if metrics["hard_negative_count"] > 0 else 0.0
    novelty_proxy += 0.20 if metrics["has_problem_boundary_text"] else 0.0

    return {
        "source_relevance": clamp(source_relevance),
        "evidence_grounding": clamp(evidence_grounding),
        "claim_validity": clamp(claim_validity),
        "falsifiability": clamp(falsifiability),
        "experimentability": clamp(experimentability),
        "novelty_proxy": clamp(novelty_proxy),
    }


def source_drift_flags(metrics: dict[str, Any]) -> bool:
    flags = set(metrics.get("quality_flags") or [])
    return bool(
        flags
        & {
            "accepted_sources_include_unclassified_roles",
            "accepted_sources_include_non_reportable_roles",
            "evidence_dimensions_collapse_to_other",
        }
    )


def flags_for(metrics: dict[str, Any], scores: dict[str, float]) -> list[str]:
    flags: list[str] = []
    for name, score in scores.items():
        if score < 0.45:
            flags.append(f"low_{name}")
    if metrics["bad_reference_ratio"] > 0.1:
        flags.append("metadata_reference_noise")
    if metrics["falsification_plan_count"] == 0:
        flags.append("missing_falsification_plan")
    if metrics["counterexample_count"] == 0:
        flags.append("missing_counterexample_audit")
    if metrics["report_ready_claim_count"] == 0:
        flags.append("no_report_ready_claims")
    if metrics["high_risk_claim_count"] > metrics["low_risk_verified_claim_count"]:
        flags.append("high_risk_claims_dominate")
    return sorted(set(flags))


def verdict(score: float, flags: list[str] | None = None) -> str:
    flags = flags or []
    hard_revision_flags = {
        "missing_falsification_plan",
        "no_report_ready_claims",
        "high_risk_claims_dominate",
    }
    has_hard_revision_flag = bool(hard_revision_flags & set(flags))
    if score >= 0.75:
        if has_hard_revision_flag:
            return "promising but needs revision"
        return "AAAI-draft candidate"
    if score >= 0.55:
        return "promising but needs revision"
    return "baseline/weak for AAAI without revision"


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    label, raw_path = value.split("=", 1)
    return label.strip(), Path(raw_path.strip())


def build_pairs(items: list[dict[str, Any]], pair_specs: list[str]) -> list[dict[str, Any]]:
    by_label = {item["label"]: item for item in items}
    pairs: list[dict[str, Any]] = []
    for spec in pair_specs:
        if ":" not in spec:
            continue
        left, right = [part.strip() for part in spec.split(":", 1)]
        if left not in by_label or right not in by_label:
            continue
        baseline = by_label[left]
        candidate = by_label[right]
        delta = {
            name: round(candidate["scores"][name] - baseline["scores"][name], 3)
            for name in QUALITY_DIMENSIONS
        }
        pairs.append(
            {
                "baseline": left,
                "candidate": right,
                "overall_delta": round(candidate["overall_score"] - baseline["overall_score"], 3),
                "score_delta": delta,
                "winner": right if candidate["overall_score"] > baseline["overall_score"] else left,
            }
        )
    return pairs


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Idea Quality Evaluation",
        "",
        "## Protocol",
        "",
    ]
    for dimension, bullets in PROTOCOL.items():
        lines.append(f"### {dimension}")
        for bullet in bullets:
            lines.append(f"- {bullet}")
        lines.append("")

    lines.extend(["## Items", ""])
    for item in result["items"]:
        lines.extend(
            [
                f"### {item['label']}",
                "",
                f"- Path: `{item['path']}`",
                f"- Overall: {item['overall_score']:.3f} ({item['verdict']})",
                f"- Flags: {', '.join(item['flags']) or 'none'}",
                "",
                "| Dimension | Score |",
                "|---|---:|",
            ]
        )
        for dimension in QUALITY_DIMENSIONS:
            lines.append(f"| {dimension} | {item['scores'][dimension]:.3f} |")
        lines.append("")

    if result["pairs"]:
        lines.extend(["## Pairwise Deltas", "", "| Baseline | Candidate | Overall delta | Winner |", "|---|---|---:|---|"])
        for pair in result["pairs"]:
            lines.append(
                f"| {pair['baseline']} | {pair['candidate']} | {pair['overall_delta']:.3f} | {pair['winner']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ScholarInsight idea quality artifacts.")
    parser.add_argument("--input", action="append", default=[], help="label=path to run/artifact directory")
    parser.add_argument("--audit-json", action="append", default=[], help="label=local_quality_audit.json")
    parser.add_argument("--pair", action="append", default=[], help="baseline_label:candidate_label")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    items: list[dict[str, Any]] = []
    for item in args.input:
        label, path = parse_labeled_path(item)
        items.append(load_artifact_dir(path, label))
    for item in args.audit_json:
        label, path = parse_labeled_path(item)
        items.extend(load_audit_json(path, label))

    pairs = build_pairs(items, args.pair)
    result = {
        "protocol": PROTOCOL,
        "items": items,
        "pairs": pairs,
    }
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "idea_quality_eval.json").write_text(output, encoding="utf-8")
        (output_dir / "idea_quality_eval.md").write_text(render_markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
