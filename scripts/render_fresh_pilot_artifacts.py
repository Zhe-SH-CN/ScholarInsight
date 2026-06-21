from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from collections import Counter
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

from scripts.local_quality_audit import TOPIC_QUERIES, audit_topic, jsonable, topic_slug  # noqa: E402

from cg.orchestrator.pipeline import (  # noqa: E402
    average,
    build_analysis_report,
    build_evidence_csv,
    build_evidence_graph,
    build_executive_summary,
    build_methodology,
    build_observability,
    build_paper_pattern_matrix,
    build_recommendations,
    build_recommendations_markdown,
    build_matrix_csv,
    claim_report_ready_reason,
    paper_key,
)
from cg.schemas.research import (  # noqa: E402
    Claim,
    CounterexampleAuditRow,
    Evidence,
    EvidenceCluster,
    FalsificationPlanRow,
    ResearchPlan,
    ResearchRequest,
    RunMetrics,
    SourceDocument,
    dimensions_for_topic,
)
from cg.settings import Settings  # noqa: E402
from cg.tools.local_paper_search import LocalPaperSearchTool  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count / total, 3)


def top_papers(evidence: list[Evidence], limit: int) -> list[str]:
    counts = Counter(paper_key(ev) for ev in evidence)
    return [paper for paper, _count in counts.most_common(limit)]


def make_request(topic_id: str, topic: str, max_results_per_query: int, max_sources: int) -> ResearchRequest:
    return ResearchRequest(
        project_name=f"ScholarInsight Fresh Pilot {topic_id}",
        target_topic=topic,
        topic_description="Fresh deterministic local pilot for pipeline freeze validation; no external LLM calls.",
        analysis_dimensions=dimensions_for_topic(topic),
        max_sources=max_sources,
        max_sources_per_query=max_results_per_query,
        max_search_rounds=1,
        max_research_loops=1,
    )


def make_metrics(result: dict[str, Any], evidence: list[Evidence], claims: list[Claim], matrix_cell_count: int) -> RunMetrics:
    source_summary = result["source_summary"]
    claim_statuses = Counter(claim.verification_status for claim in claims)
    return RunMetrics(
        source_candidates=source_summary["candidate_count"],
        sources_rejected=source_summary["rejected_count"],
        sources_fetched=result["deterministic_summary"]["document_count"],
        sources_failed=max(0, source_summary["selected_count"] - result["deterministic_summary"]["document_count"]),
        evidence_count=len(evidence),
        claim_count=len(claims),
        verified_claim_count=claim_statuses.get("verified", 0),
        challenged_claim_count=len(claims) - claim_statuses.get("verified", 0),
        matrix_cell_count=matrix_cell_count,
        recommendation_count=0,
        average_evidence_confidence=round(average([ev.confidence for ev in evidence]), 3),
        coverage_score=0,
    )


def report_ready_claims(claims: list[Claim]) -> list[Claim]:
    return [claim for claim in claims if not claim_report_ready_reason(claim)]


def summary_for_topic(
    *,
    topic: str,
    topic_id: str,
    result: dict[str, Any],
    evidence: list[Evidence],
    claims: list[Claim],
    counterexamples: list[CounterexampleAuditRow],
    falsification: list[FalsificationPlanRow],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    ready = report_ready_claims(claims)
    ready_ids = {claim.claim_id for claim in ready}
    falsification_ids = {row.target_claim_id for row in falsification}
    audit_only_reasons = Counter(
        reason for claim in claims if (reason := claim_report_ready_reason(claim))
    )
    return {
        "artifact_kind": "fresh_deterministic_local",
        "topic_id": topic_id,
        "topic": topic,
        "candidate_count": result["source_summary"]["candidate_count"],
        "accepted_count": result["source_summary"]["accepted_count"],
        "rejected_count": result["source_summary"]["rejected_count"],
        "selected_count": result["source_summary"]["selected_count"],
        "accepted_subtypes": result["source_summary"]["accepted_subtypes"],
        "evidence_count": len(evidence),
        "claim_count": len(claims),
        "verified_claim_count": sum(1 for claim in claims if claim.verification_status == "verified"),
        "report_ready_count": len(ready),
        "audit_only_count": len(claims) - len(ready),
        "counterexample_audit_count": len(counterexamples),
        "counterexample_report_visible_count": sum(1 for row in counterexamples if row.report_visible),
        "counterexample_types": Counter(row.counterexample_type for row in counterexamples),
        "counterexample_metadata_noise_count": sum(
            1 for row in counterexamples if row.counterexample_type == "metadata_noise"
        ),
        "report_ready_claim_types": Counter(claim.claim_type for claim in ready),
        "audit_only_reasons": audit_only_reasons,
        "quality_flags": result["quality_diagnostics"]["quality_flags"],
        "report_ready_claims": [
            {
                "claim_id": claim.claim_id,
                "claim_type": claim.claim_type,
                "dimension": claim.dimension,
                "source_paper_count": claim.source_paper_count,
                "supporting_source_subtype_paper_counts": claim.supporting_source_subtype_paper_counts,
                "claim": claim.final_wording or claim.claim,
            }
            for claim in ready
        ],
        "falsification_plan_count": len(falsification),
        "falsification_covered_report_ready_claim_count": len(ready_ids & falsification_ids),
        "falsification_missing_report_ready_claim_ids": sorted(ready_ids - falsification_ids),
        "provenance": provenance,
    }


async def render_topic_artifact(
    *,
    tool: LocalPaperSearchTool,
    settings: Settings,
    topic_id: str,
    output_root: Path,
    max_results_per_query: int,
    max_sources: int,
    claim_limit: int,
    provenance_base: dict[str, Any],
) -> dict[str, Any]:
    topic, queries = TOPIC_QUERIES[topic_id]
    result = await audit_topic(
        tool,
        settings,
        topic_id,
        max_results_per_query,
        max_sources,
        claim_limit,
        source_only=False,
    )
    run_id = f"fresh_pilot_{topic_id}_{topic_slug(topic)}"
    request = make_request(topic_id, topic, max_results_per_query, max_sources)
    evidence = [Evidence(**row) for row in result["evidence"]]
    claims = [Claim(**row) for row in result["claims"]]
    clusters = [EvidenceCluster(**row) for row in result["clusters"]]
    documents = [SourceDocument(**row) for row in result["documents"]]
    counterexamples = [CounterexampleAuditRow(**row) for row in result["counterexample_audit"]]
    falsification = [FalsificationPlanRow(**row) for row in result["falsification_plan"]]
    papers = top_papers(evidence, limit=max_sources) or [request.target_topic]
    matrix = build_paper_pattern_matrix(evidence, papers, request.analysis_dimensions)
    metrics = make_metrics(result, evidence, claims, len(matrix.cells))
    metrics.coverage_score = round(
        average(list(matrix.coverage_by_paper.values()) + list(matrix.coverage_by_dimension.values())),
        3,
    )
    recommendations = build_recommendations(run_id, request, claims, evidence, matrix)
    metrics.recommendation_count = len(recommendations)
    plan = ResearchPlan(
        research_goal=request.research_goal,
        papers=papers,
        dimensions=request.analysis_dimensions,
        queries=queries,
        required_agents=[
            "ChiefResearchPlanner",
            "SearchAndSourceAgent",
            "EvidenceStructuringAgent",
            "AnalysisAndReviewAgent",
            "ReportComposerAgent",
        ],
        quality_rules=[
            "fresh deterministic local artifact",
            "CUDA reranker must be explicitly loaded when SCHOLAR_RERANKER_DEVICE=cuda",
            "report-ready claims require formal evidence gate g(c)",
        ],
    )
    observability = build_observability(
        datetime.now(timezone.utc),
        plan,
        metrics,
        evidence,
        claims,
        matrix,
        recommendations,
    )
    evidence_graph = build_evidence_graph(evidence, claims)
    report = build_analysis_report(
        request,
        evidence,
        claims,
        metrics,
        matrix,
        recommendations,
        observability,
        counterexamples,
        falsification,
    )
    executive_summary = build_executive_summary(request, metrics, observability, recommendations, claims)
    methodology = build_methodology(request, observability)

    reranker_device_loaded = getattr(tool.index, "_reranker", None)[2] if getattr(tool.index, "_reranker", None) else None
    provenance = {
        **provenance_base,
        "run_id": run_id,
        "topic_id": topic_id,
        "topic": topic,
        "queries": queries,
        "dimensions": request.analysis_dimensions,
        "reranker_error": getattr(tool.index, "_reranker_error", None),
        "reranker_device_loaded": reranker_device_loaded,
    }
    topic_dir = output_root / topic_id
    write_json(topic_dir / "manifest.json", request)
    write_json(topic_dir / "provenance.json", provenance)
    write_json(topic_dir / "sources" / "selected_sources.json", result["selected_sources"])
    write_json(topic_dir / "sources" / "accepted_sources.json", result["accepted_sources"])
    write_json(topic_dir / "sources" / "rejected_sources.json", result["rejected_sources"])
    write_json(topic_dir / "documents" / "documents.json", documents)
    write_json(topic_dir / "evidence" / "evidence.json", evidence)
    write_json(topic_dir / "exports" / "claims.json", claims)
    write_json(topic_dir / "exports" / "claim_backlog.json", [
        {
            "claim_id": claim.claim_id,
            "reason": claim_report_ready_reason(claim),
            "claim": claim.final_wording or claim.claim,
        }
        for claim in claims
        if claim_report_ready_reason(claim)
    ])
    write_json(topic_dir / "exports" / "counterexample_audit.json", counterexamples)
    write_json(topic_dir / "exports" / "falsification_plan.json", falsification)
    write_json(topic_dir / "exports" / "matrix.json", matrix)
    write_text(topic_dir / "exports" / "matrix.csv", build_matrix_csv(matrix))
    write_json(topic_dir / "exports" / "recommendations.json", recommendations)
    write_text(topic_dir / "exports" / "recommendations.md", build_recommendations_markdown(recommendations))
    write_json(topic_dir / "exports" / "observability.json", observability)
    write_json(topic_dir / "exports" / "evidence_clusters.json", clusters)
    write_json(topic_dir / "exports" / "evidence_graph.json", evidence_graph)
    write_text(topic_dir / "exports" / "evidence_matrix.csv", build_evidence_csv(evidence, {ev.evidence_id: ev for ev in evidence}))
    write_text(topic_dir / "reports" / "report.md", report)
    write_text(topic_dir / "reports" / "executive_summary.md", executive_summary)
    write_text(topic_dir / "reports" / "methodology.md", methodology)
    write_json(
        topic_dir / "reports" / "report.json",
        {
            "run_id": run_id,
            "request": request,
            "metrics": metrics,
            "claims": claims,
            "matrix": matrix,
            "recommendations": recommendations,
            "observability": observability,
            "counterexample_audit": counterexamples,
            "falsification_plan": falsification,
            "evidence_ids": [ev.evidence_id for ev in evidence],
        },
    )
    summary = summary_for_topic(
        topic=topic,
        topic_id=topic_id,
        result=result,
        evidence=evidence,
        claims=claims,
        counterexamples=counterexamples,
        falsification=falsification,
        provenance=provenance,
    )
    write_json(topic_dir / "summary.json", summary)
    print(
        topic_id,
        topic,
        "accepted",
        summary["accepted_count"],
        "report_ready",
        summary["report_ready_count"],
        "falsification",
        f"{summary['falsification_covered_report_ready_claim_count']}/{summary['report_ready_count']}",
        "flags",
        summary["quality_flags"],
        flush=True,
    )
    return summary


async def main() -> None:
    parser = argparse.ArgumentParser(description="Render fresh deterministic local ScholarInsight pilot artifacts.")
    parser.add_argument("--topics", default="004,006,010,011,012")
    parser.add_argument("--max-results-per-query", type=int, default=8)
    parser.add_argument("--max-sources", type=int, default=12)
    parser.add_argument("--claim-limit", type=int, default=24)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--allow-hf-network", action="store_true")
    args = parser.parse_args()

    if not args.allow_hf_network:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    topic_ids = [item.strip().zfill(3) for item in args.topics.split(",") if item.strip()]
    unknown = [item for item in topic_ids if item not in TOPIC_QUERIES]
    if unknown:
        raise SystemExit(f"Unknown topic id(s): {', '.join(unknown)}")

    settings = Settings()
    output_root = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT
        / "data"
        / "quality_audits"
        / f"fresh_pilot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{'_'.join(topic_ids)}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    tool = LocalPaperSearchTool(settings)
    provenance_base = {
        "artifact_kind": "fresh_deterministic_local",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "reranker_model": settings.scholar_reranker_model,
        "reranker_device_requested": settings.scholar_reranker_device,
        "source_gate_enabled": settings.scholar_source_gate_enabled,
        "min_source_relevance": settings.scholar_min_source_relevance,
        "hf_offline": not args.allow_hf_network,
        "max_results_per_query": args.max_results_per_query,
        "max_sources": args.max_sources,
        "claim_limit": args.claim_limit,
        "external_llm_calls": False,
    }

    summaries = []
    for topic_id in topic_ids:
        summaries.append(
            await render_topic_artifact(
                tool=tool,
                settings=settings,
                topic_id=topic_id,
                output_root=output_root,
                max_results_per_query=args.max_results_per_query,
                max_sources=args.max_sources,
                claim_limit=args.claim_limit,
                provenance_base=provenance_base,
            )
        )
    root_summary = {
        **provenance_base,
        "artifact_dir": str(output_root),
        "topics": summaries,
    }
    write_json(output_root / "summary.json", root_summary)
    print(f"WROTE {output_root}")


if __name__ == "__main__":
    asyncio.run(main())
