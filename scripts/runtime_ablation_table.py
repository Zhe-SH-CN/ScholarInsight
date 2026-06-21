from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _repo_root()
BACKEND_ROOT = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts.idea_quality_evaluator import load_artifact_dir  # noqa: E402
from scripts.local_quality_audit import TOPIC_QUERIES, jsonable  # noqa: E402
from scripts.render_fresh_pilot_artifacts import git_commit, render_topic_artifact  # noqa: E402

from cg.settings import Settings  # noqa: E402
from cg.tools.local_paper_search import LocalPaperSearchTool  # noqa: E402


VARIANT_SETTINGS: dict[str, dict[str, Any]] = {
    "no_reranker": {
        "scholar_enable_reranker": False,
        "scholar_reranker_device": "auto",
    },
    "no_source_gate": {
        "scholar_source_gate_enabled": False,
        "scholar_reranker_device": "cuda",
    },
    "no_reranker_no_source_gate": {
        "scholar_enable_reranker": False,
        "scholar_source_gate_enabled": False,
        "scholar_reranker_device": "auto",
    },
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def topic_dirs(root: Path) -> list[Path]:
    return sorted(
        item
        for item in root.iterdir()
        if item.is_dir() and (item / "exports" / "claims.json").exists()
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    label, raw_path = value.split("=", 1)
    return label.strip(), Path(raw_path.strip())


def summarize_artifact(path: Path, variant: str, full_scores: dict[str, float]) -> dict[str, Any]:
    summary = read_json(path / "summary.json")
    evaluation = load_artifact_dir(path, f"{path.name}::{variant}")
    topic_id = summary.get("topic_id") or path.name
    full_score = full_scores.get(topic_id)
    return {
        "topic_id": topic_id,
        "topic": summary.get("topic") or path.name,
        "variant": variant,
        "artifact_dir": str(path),
        "accepted_count": summary.get("accepted_count"),
        "rejected_count": summary.get("rejected_count"),
        "evidence_count": summary.get("evidence_count"),
        "claim_count": summary.get("claim_count"),
        "verified_claim_count": summary.get("verified_claim_count"),
        "report_ready_count": summary.get("report_ready_count"),
        "falsification_coverage": (
            f"{summary.get('falsification_covered_report_ready_claim_count')}/"
            f"{summary.get('report_ready_count')}"
        ),
        "quality_flags": summary.get("quality_flags") or [],
        "overall_score": evaluation["overall_score"],
        "score_delta_vs_full": round(evaluation["overall_score"] - full_score, 3)
        if full_score is not None
        else None,
        "verdict": evaluation["verdict"],
        "evaluator_flags": evaluation["flags"],
        "scores": evaluation["scores"],
    }


def baseline_rows(baseline_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in topic_dirs(baseline_root):
        evaluation = load_artifact_dir(path, f"{path.name}::full")
        summary = read_json(path / "summary.json")
        rows.append(
            {
                "topic_id": summary.get("topic_id") or path.name,
                "topic": summary.get("topic") or path.name,
                "variant": "full",
                "artifact_dir": str(path),
                "accepted_count": summary.get("accepted_count"),
                "rejected_count": summary.get("rejected_count"),
                "evidence_count": summary.get("evidence_count"),
                "claim_count": summary.get("claim_count"),
                "verified_claim_count": summary.get("verified_claim_count"),
                "report_ready_count": summary.get("report_ready_count"),
                "falsification_coverage": (
                    f"{summary.get('falsification_covered_report_ready_claim_count')}/"
                    f"{summary.get('report_ready_count')}"
                ),
                "quality_flags": summary.get("quality_flags") or [],
                "overall_score": evaluation["overall_score"],
                "score_delta_vs_full": 0.0,
                "verdict": evaluation["verdict"],
                "evaluator_flags": evaluation["flags"],
                "scores": evaluation["scores"],
            }
        )
    return rows


async def run_variant(
    *,
    variant: str,
    topic_ids: list[str],
    output_root: Path,
    max_results_per_query: int,
    max_sources: int,
    claim_limit: int,
    allow_hf_network: bool,
) -> Path:
    if variant not in VARIANT_SETTINGS:
        raise SystemExit(f"Unknown runtime ablation variant: {variant}")
    settings = Settings(**VARIANT_SETTINGS[variant])
    variant_root = output_root / variant
    variant_root.mkdir(parents=True, exist_ok=True)
    tool = LocalPaperSearchTool(settings)
    provenance_base = {
        "artifact_kind": "fresh_deterministic_local",
        "ablation_type": "runtime_settings",
        "ablation_variant": variant,
        "ablation_settings": VARIANT_SETTINGS[variant],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "reranker_model": settings.scholar_reranker_model,
        "reranker_device_requested": settings.scholar_reranker_device,
        "source_gate_enabled": settings.scholar_source_gate_enabled,
        "min_source_relevance": settings.scholar_min_source_relevance,
        "hf_offline": not allow_hf_network,
        "max_results_per_query": max_results_per_query,
        "max_sources": max_sources,
        "claim_limit": claim_limit,
        "external_llm_calls": False,
    }
    summaries = []
    for topic_id in topic_ids:
        summaries.append(
            await render_topic_artifact(
                tool=tool,
                settings=settings,
                topic_id=topic_id,
                output_root=variant_root,
                max_results_per_query=max_results_per_query,
                max_sources=max_sources,
                claim_limit=claim_limit,
                provenance_base=provenance_base,
            )
        )
    write_json(
        variant_root / "summary.json",
        {
            **provenance_base,
            "artifact_dir": str(variant_root),
            "topics": summaries,
        },
    )
    return variant_root


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# ScholarInsight Runtime Ablation Table",
        "",
        "This table reruns the deterministic local pipeline under retrieval/source-stage settings. It does not call external LLMs and does not run the 132-topic batch.",
        "",
        "| Topic | Variant | Score | Delta | Accepted | Evidence | Claims | Report-ready | Falsification | Flags |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        delta = row["score_delta_vs_full"]
        delta_text = "" if delta is None else f"{delta:.3f}"
        lines.append(
            f"| {row['topic_id']} {row['topic']} | {row['variant']} | {row['overall_score']:.3f} "
            f"| {delta_text} | {row['accepted_count']} | {row['evidence_count']} | {row['claim_count']} "
            f"| {row['report_ready_count']} | {row['falsification_coverage']} "
            f"| {', '.join(row['quality_flags'] + row['evaluator_flags']) or 'none'} |"
        )
    lines.append("")
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run ScholarInsight deterministic runtime ablations.")
    parser.add_argument("--baseline-root", required=True, help="Fresh full artifact root to compare against.")
    parser.add_argument("--variants", default="no_reranker,no_reranker_no_source_gate")
    parser.add_argument("--topics", default="004,006,010,011,012")
    parser.add_argument("--max-results-per-query", type=int, default=8)
    parser.add_argument("--max-sources", type=int, default=12)
    parser.add_argument("--claim-limit", type=int, default=24)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--allow-hf-network", action="store_true")
    parser.add_argument(
        "--existing-variant",
        action="append",
        default=[],
        help="label=path to an existing variant root; summarize without rerunning retrieval.",
    )
    args = parser.parse_args()

    if not args.allow_hf_network:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    baseline_root = Path(args.baseline_root)
    topic_ids = [item.strip().zfill(3) for item in args.topics.split(",") if item.strip()]
    unknown = [item for item in topic_ids if item not in TOPIC_QUERIES]
    if unknown:
        raise SystemExit(f"Unknown topic id(s): {', '.join(unknown)}")
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    output_root = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT
        / "data"
        / "quality_audits"
        / f"runtime_ablation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{'_'.join(topic_ids)}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    rows = baseline_rows(baseline_root)
    full_scores = {row["topic_id"]: row["overall_score"] for row in rows if row["variant"] == "full"}
    for item in args.existing_variant:
        variant, variant_root = parse_labeled_path(item)
        for path in topic_dirs(variant_root):
            rows.append(summarize_artifact(path, variant, full_scores))
    for variant in variants:
        if any(variant == parse_labeled_path(item)[0] for item in args.existing_variant):
            continue
        variant_root = await run_variant(
            variant=variant,
            topic_ids=topic_ids,
            output_root=output_root,
            max_results_per_query=args.max_results_per_query,
            max_sources=args.max_sources,
            claim_limit=args.claim_limit,
            allow_hf_network=args.allow_hf_network,
        )
        for path in topic_dirs(variant_root):
            rows.append(summarize_artifact(path, variant, full_scores))
    result = {
        "baseline_root": str(baseline_root),
        "output_root": str(output_root),
        "ablation_type": "runtime_settings",
        "variants": variants,
        "items": rows,
    }
    write_json(output_root / "runtime_ablation_table.json", result)
    write_text(output_root / "runtime_ablation_table.md", render_markdown(rows))
    print(json.dumps({"items": len(rows), "output_dir": str(output_root)}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
