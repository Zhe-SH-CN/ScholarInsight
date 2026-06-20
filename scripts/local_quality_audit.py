from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _repo_root()
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from cg.agents.research_agents import deterministic_red_team  # noqa: E402
from cg.orchestrator.pipeline import (  # noqa: E402
    SYNTHESIS_SOURCE_ROLES,
    build_counterexample_audit_rows,
    build_claims_from_clusters,
    build_evidence_clusters,
    build_falsification_plan_rows,
    claim_report_ready_reason,
    extract_evidence_from_document,
    prepare_claims_for_review,
    stable_id,
)
from cg.schemas.research import ResearchRequest, SourceDocument, dimensions_for_topic  # noqa: E402
from cg.settings import Settings  # noqa: E402
from cg.tools.local_paper_search import LocalPaperSearchTool  # noqa: E402


TOPIC_QUERIES: dict[str, tuple[str, list[str]]] = {
    "004": (
        "RAG with Knowledge Graphs",
        [
            "retrieval augmented generation knowledge graphs graph retrieval benchmark",
            "knowledge graph enhanced RAG question answering large language models",
            "graph RAG methods knowledge graph reasoning LLM retrieval",
            "KG-RAG evaluation hallucination factuality evidence",
        ],
    ),
    "005": (
        "Scientific Reasoning with LLMs",
        [
            "scientific reasoning with large language models benchmark hypothesis generation",
            "LLM scientific reasoning benchmark experiment planning discovery",
            "scientific discovery large language models reasoning evaluation",
            "AI scientist LLM scientific reasoning literature hypothesis",
        ],
    ),
    "006": (
        "Mathematical Reasoning",
        [
            "mathematical reasoning large language models benchmark proof theorem solving",
            "LLM mathematical reasoning chain of thought formal proof evaluation",
            "math word problem reasoning large language models GSM8K MATH benchmark",
            "theorem proving language models mathematical proof generation",
        ],
    ),
    "010": (
        "Causal Reasoning with LLMs",
        [
            "large language models causal reasoning benchmark causal discovery intervention counterfactual evaluation",
            "CausalBench causal question answering benchmark large language models causal reasoning",
            "large language models do-calculus intervention reasoning causal inference Pearl causal graphs",
            "causal reasoning evaluation large language models not fairness bias benchmark causal inference tasks",
        ],
    ),
    "011": (
        "Counterfactual Inference",
        [
            "counterfactual inference treatment effect estimation causal machine learning benchmark",
            "counterfactual explanations causal inference fairness recourse methods",
            "potential outcomes counterfactual inference heterogeneous treatment effects",
            "structural causal models counterfactual inference identifiability assumptions",
        ],
    ),
    "012": (
        "Multi-hop Reasoning on Graphs",
        [
            "multi hop reasoning on knowledge graphs benchmark path reasoning",
            "graph multi-hop question answering knowledge graph reasoning language models",
            "multi-hop graph reasoning retrieval path-based reasoning benchmark",
            "compositional reasoning on graphs knowledge graph neural symbolic",
        ],
    ),
}


def topic_slug(topic: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", topic).strip("_")


def jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


def source_row(candidate: Any) -> dict[str, Any]:
    return {
        "title": candidate.title,
        "url": candidate.url,
        "query": candidate.query,
        "relevance_score": candidate.relevance_score,
        "relevance_label": candidate.relevance_label,
        "rejection_reason": candidate.rejection_reason,
        "source_subtype": candidate.source_subtype,
        "source_subtype_reason": candidate.source_subtype_reason,
        "embedding_score": candidate.embedding_score,
        "lexical_score": candidate.lexical_score,
        "reranker_score": candidate.reranker_score,
    }


def claim_row(claim: Any) -> dict[str, Any]:
    row = claim.model_dump(mode="json")
    row["report_ready_rejection_reason"] = claim_report_ready_reason(claim)
    row["is_report_ready"] = not row["report_ready_rejection_reason"]
    return row


def ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count / total, 3)


def quality_diagnostics(accepted: list[Any], evidence: list[Any], claims: list[Any]) -> dict[str, Any]:
    accepted_subtypes = Counter(item.source_subtype or "unclassified" for item in accepted)
    evidence_dimensions = Counter(ev.dimension or "other" for ev in evidence)
    claim_statuses = Counter(claim.verification_status for claim in claims)
    claim_types = Counter(claim.claim_type for claim in claims)
    accepted_unclassified_count = accepted_subtypes.get("unclassified", 0)
    accepted_non_reportable_count = sum(
        count for subtype, count in accepted_subtypes.items() if subtype not in SYNTHESIS_SOURCE_ROLES
    )
    other_dimension_count = evidence_dimensions.get("other", 0)
    audit_verified_synthesis_count = sum(
        1
        for claim in claims
        if claim.verification_status == "verified"
        and claim.claim_type in {"comparative", "cross_role_contrast"}
        and claim.risk_level != "high"
        and not claim.backlog_reason
    )
    report_ready_rejection_reasons = Counter(
        reason for claim in claims if (reason := claim_report_ready_reason(claim))
    )
    report_ready_verified_count = len(claims) - sum(report_ready_rejection_reasons.values())
    verified_single_paper_count = sum(
        1
        for claim in claims
        if claim.verification_status == "verified"
        and claim.claim_type == "single_paper_observation"
        and claim.risk_level != "high"
        and not claim.backlog_reason
    )
    flags: list[str] = []
    if accepted_unclassified_count:
        flags.append("accepted_sources_include_unclassified_roles")
    if accepted and accepted_non_reportable_count == len(accepted):
        flags.append("no_accepted_sources_have_reportable_roles")
    elif accepted_non_reportable_count:
        flags.append("accepted_sources_include_non_reportable_roles")
    if evidence and ratio(other_dimension_count, len(evidence)) >= 0.5:
        flags.append("evidence_dimensions_collapse_to_other")
    if claims and audit_verified_synthesis_count == 0:
        flags.append("no_audit_verified_synthesis")
    if claims and report_ready_verified_count == 0:
        flags.append("no_report_ready_verified_synthesis")
    if audit_verified_synthesis_count and report_ready_verified_count == 0:
        flags.append("audit_verified_claims_not_report_ready")
    if claims and claim_statuses.get("verified", 0) == 0:
        flags.append("no_verified_claims")

    return {
        "accepted_unclassified_count": accepted_unclassified_count,
        "accepted_non_reportable_count": accepted_non_reportable_count,
        "accepted_reportable_count": len(accepted) - accepted_non_reportable_count,
        "evidence_dimensions": evidence_dimensions,
        "other_dimension_count": other_dimension_count,
        "other_dimension_ratio": ratio(other_dimension_count, len(evidence)),
        "claim_statuses": claim_statuses,
        "claim_types": claim_types,
        "audit_verified_synthesis_count": audit_verified_synthesis_count,
        "report_ready_verified_count": report_ready_verified_count,
        "report_worthy_verified_count": report_ready_verified_count,
        "report_ready_rejection_reasons": report_ready_rejection_reasons,
        "verified_single_paper_observation_count": verified_single_paper_count,
        "quality_flags": flags,
    }


def candidate_to_document(run_id: str, candidate: Any) -> SourceDocument | None:
    content = (candidate.content or candidate.snippet or "").strip()
    if len(content) < 80:
        return None
    source_id = stable_id("src", f"{candidate.url}:{candidate.source_provider}:{candidate.query}")
    content_hash = hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()
    return SourceDocument(
        source_id=source_id,
        run_id=run_id,
        url=candidate.url,
        title=candidate.title,
        content=content,
        excerpt=content[:500],
        source_type=candidate.source_type or "academic_paper",
        http_status=200,
        content_hash=content_hash,
        fetched_at=datetime.now(timezone.utc),
        parser=candidate.content_source or "provider_content",
        provider=candidate.source_provider or "local_papers",
        query=candidate.query,
        content_source=candidate.content_source or "provider_content",
        source_score=candidate.score,
        relevance_score=candidate.relevance_score,
        relevance_label=candidate.relevance_label,
        rejection_reason=candidate.rejection_reason,
        source_subtype=candidate.source_subtype,
        source_subtype_reason=candidate.source_subtype_reason,
        embedding_score=candidate.embedding_score,
        lexical_score=candidate.lexical_score,
        reranker_score=candidate.reranker_score,
        ok=True,
        error=None,
        published_at=candidate.published_at,
        date_source=candidate.date_source,
    )


async def audit_topic(
    tool: LocalPaperSearchTool,
    settings: Settings,
    key: str,
    max_results_per_query: int,
    max_sources: int,
    claim_limit: int,
    source_only: bool,
) -> dict[str, Any]:
    topic, queries = TOPIC_QUERIES[key]
    run_id = f"local_quality_{key}_{topic_slug(topic)}"
    by_url: dict[str, Any] = {}

    for query in queries:
        for candidate in await tool.search_for_topic(query, topic, max_results=max_results_per_query):
            current = by_url.get(candidate.url)
            if current is None or candidate.relevance_score > current.relevance_score:
                by_url[candidate.url] = candidate

    candidates = sorted(by_url.values(), key=lambda item: item.relevance_score, reverse=True)
    accepted = [
        item
        for item in candidates
        if item.relevance_label != "reject"
        and item.relevance_score >= settings.scholar_min_source_relevance
    ]
    accepted_urls = {item.url for item in accepted}
    rejected = [item for item in candidates if item.url not in accepted_urls]
    selected = accepted[:max_sources]
    dimensions = dimensions_for_topic(topic)
    request = ResearchRequest(
        project_name=f"Local Quality Audit {key}",
        target_topic=topic,
        topic_description="Local deterministic quality audit; no external LLM calls.",
        analysis_dimensions=dimensions,
        max_sources=max_sources,
        max_sources_per_query=max_results_per_query,
        max_search_rounds=1,
        max_research_loops=1,
    )

    documents: list[SourceDocument] = []
    evidence = []
    clusters = []
    claims = []
    if not source_only:
        for candidate in selected:
            document = candidate_to_document(run_id, candidate)
            if document is None:
                continue
            documents.append(document)
            evidence.extend(extract_evidence_from_document(run_id, document, dimensions, []))
        clusters = build_evidence_clusters(evidence)
        claims = build_claims_from_clusters(run_id, clusters, evidence, limit=claim_limit)
        claims = prepare_claims_for_review(claims, evidence)
        claims = deterministic_red_team(claims, evidence)
    counterexample_audit = build_counterexample_audit_rows(
        request,
        claims,
        candidates,
        selected_source_urls=[item.url for item in selected],
    )
    falsification_plan = build_falsification_plan_rows(
        request,
        claims,
        counterexample_audit,
    )
    counterexample_types = Counter(row.counterexample_type for row in counterexample_audit)
    counterexample_visible_count = sum(1 for row in counterexample_audit if row.report_visible)
    report_ready_claim_ids = {
        claim.claim_id
        for claim in claims
        if not claim_report_ready_reason(claim)
    }
    falsification_claim_ids = {row.target_claim_id for row in falsification_plan}

    claim_statuses = Counter(claim.verification_status for claim in claims)
    claim_types = Counter(claim.claim_type for claim in claims)
    mixed_role_verified = sum(
        1
        for claim in claims
        if claim.verification_status == "verified"
        and len(claim.supporting_source_subtype_counts or {}) > 1
    )

    return {
        "topic": topic,
        "queries": queries,
        "dimensions": dimensions,
        "source_summary": {
            "candidate_count": len(candidates),
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "selected_count": len(selected),
            "accepted_subtypes": Counter(item.source_subtype for item in accepted),
            "rejected_subtypes": Counter(item.source_subtype for item in rejected),
            "top_accepted": [source_row(item) for item in accepted[:12]],
            "top_rejected": [source_row(item) for item in rejected[:12]],
        },
        "deterministic_summary": {
            "document_count": len(documents),
            "evidence_count": len(evidence),
            "cluster_count": len(clusters),
            "claim_count": len(claims),
            "claim_statuses": claim_statuses,
            "claim_types": claim_types,
            "mixed_role_verified": mixed_role_verified,
        },
        "quality_diagnostics": quality_diagnostics(accepted, evidence, claims),
        "counterexample_audit_summary": {
            "row_count": len(counterexample_audit),
            "report_visible_count": counterexample_visible_count,
            "counterexample_types": counterexample_types,
            "metadata_noise_count": counterexample_types.get("metadata_noise", 0),
        },
        "falsification_plan_summary": {
            "row_count": len(falsification_plan),
            "report_ready_claim_count": len(report_ready_claim_ids),
            "covered_report_ready_claim_count": len(report_ready_claim_ids & falsification_claim_ids),
            "missing_report_ready_claim_ids": sorted(report_ready_claim_ids - falsification_claim_ids),
        },
        "counterexample_audit": [row.model_dump(mode="json") for row in counterexample_audit],
        "falsification_plan": [row.model_dump(mode="json") for row in falsification_plan],
        "claims": [claim_row(claim) for claim in claims],
        "clusters": [cluster.model_dump(mode="json") for cluster in clusters],
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local-only ScholarInsight source/evidence/claim quality audit."
    )
    parser.add_argument(
        "--topics",
        default="005,006",
        help="Comma-separated topic ids from: " + ",".join(sorted(TOPIC_QUERIES)),
    )
    parser.add_argument("--max-results-per-query", type=int, default=8)
    parser.add_argument("--max-sources", type=int, default=12)
    parser.add_argument("--claim-limit", type=int, default=24)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--output", default="")
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
    tool = LocalPaperSearchTool(settings)
    audit = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "reranker_model": settings.scholar_reranker_model,
            "reranker_device": settings.scholar_reranker_device,
            "source_gate_enabled": settings.scholar_source_gate_enabled,
            "hf_offline": not args.allow_hf_network,
            "max_results_per_query": args.max_results_per_query,
            "max_sources": args.max_sources,
            "claim_limit": args.claim_limit,
            "source_only": args.source_only,
        },
        "topics": {},
    }

    for topic_id in topic_ids:
        result = await audit_topic(
            tool,
            settings,
            topic_id,
            args.max_results_per_query,
            args.max_sources,
            args.claim_limit,
            args.source_only,
        )
        audit["topics"][topic_id] = result
        source_summary = result["source_summary"]
        deterministic_summary = result["deterministic_summary"]
        print(
            topic_id,
            result["topic"],
            "accepted",
            source_summary["accepted_count"],
            dict(source_summary["accepted_subtypes"]),
            "claims",
            deterministic_summary["claim_count"],
            dict(deterministic_summary["claim_statuses"]),
            "flags",
            result["quality_diagnostics"]["quality_flags"],
            flush=True,
        )

    audit["reranker_error"] = getattr(tool.index, "_reranker_error", None)
    if getattr(tool.index, "_reranker", None):
        audit["reranker_device_loaded"] = tool.index._reranker[2]

    output = Path(args.output) if args.output else (
        REPO_ROOT
        / "data"
        / "quality_audits"
        / f"local_quality_audit_{'_'.join(topic_ids)}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(jsonable(audit), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {output} reranker_error={audit['reranker_error']} device={audit.get('reranker_device_loaded')}")


if __name__ == "__main__":
    asyncio.run(main())
