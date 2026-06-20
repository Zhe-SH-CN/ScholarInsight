"""Evidence cluster construction and cluster-backed claims."""

from __future__ import annotations

from datetime import datetime, timezone

from cg.orchestrator.pipeline import build_claims_from_clusters, build_evidence_clusters
from cg.schemas.research import Claim, Evidence


def test_evidence_clusters_group_related_evidence_across_papers() -> None:
    evidence = [
        _evidence("ev_a1", "Paper A", "data_evaluation_engineering", "The benchmark evaluates contamination."),
        _evidence("ev_a2", "Paper A", "data_evaluation_engineering", "The dataset adds evaluation metrics."),
        _evidence("ev_b1", "Paper B", "data_evaluation_engineering", "A benchmark tests reasoning robustness."),
    ]

    clusters = build_evidence_clusters(evidence)

    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.label == "evaluation and benchmark design"
    assert cluster.evidence_count == 3
    assert cluster.independent_paper_count == 2
    assert cluster.status == "candidate"


def test_cluster_claims_start_with_atomic_observations() -> None:
    evidence = [
        _evidence("ev_a1", "Paper A", "inference_time_control", "Self-consistency improves search."),
        _evidence("ev_a2", "Paper A", "inference_time_control", "Tree search controls test-time compute."),
        _evidence("ev_b1", "Paper B", "inference_time_control", "Best-of-N decoding uses verifier guided search."),
    ]
    clusters = build_evidence_clusters(evidence)

    claims = build_claims_from_clusters("run_test", clusters, evidence)

    assert len(claims) == 1
    claim = claims[0]
    assert claim.claim_type == "single_paper_observation"
    assert claim.evidence_support_level == "single_paper"
    assert claim.source_paper_count == 1
    assert claim.evidence_cluster_id == clusters[0].cluster_id
    assert claim.evidence_cluster_label == "search and inference-time control"


def test_cluster_claims_synthesize_only_same_source_role() -> None:
    evidence = [
        _evidence(
            "ev_a1",
            "Paper A",
            "inference_time_control",
            "Graph QA uses path search.",
            source_subtype="kgqa_or_graph_reasoning",
        ),
        _evidence(
            "ev_b1",
            "Paper B",
            "inference_time_control",
            "KG reasoning uses graph path search planning.",
            source_subtype="kgqa_or_graph_reasoning",
        ),
        _evidence(
            "ev_c1",
            "Paper C",
            "inference_time_control",
            "A generic GNN propagates messages.",
            source_subtype="graph_ml_adjacent",
        ),
    ]
    clusters = build_evidence_clusters(evidence)

    claims = build_claims_from_clusters("run_test", clusters, evidence)

    synthesis_claims = [claim for claim in claims if claim.claim_type == "comparative"]
    assert len(synthesis_claims) == 1
    claim = synthesis_claims[0]
    assert claim.source_paper_count == 2
    assert "KGQA/graph reasoning" in claim.claim
    assert set(claim.supporting_evidence_ids) == {"ev_a1", "ev_b1"}


def test_evidence_clusters_track_verified_claims() -> None:
    evidence = [
        _evidence("ev_a1", "Paper A", "formal_experimental_tightening", "Ablation validates the method."),
        _evidence("ev_b1", "Paper B", "formal_experimental_tightening", "Controlled experiment checks robustness."),
    ]
    clusters = build_evidence_clusters(evidence)
    claim = Claim(
        claim_id="claim_a",
        run_id="run_test",
        dimension="formal_experimental_tightening",
        dimension_label="formal_experimental_tightening",
        claim="Experiments validate the methods.",
        supporting_evidence_ids=["ev_a1", "ev_b1"],
        confidence=0.8,
        risk_level="low",
        reasoning_summary="Supported by two papers.",
        verification_status="verified",
        evidence_cluster_id=clusters[0].cluster_id,
        evidence_cluster_label=clusters[0].label,
    )

    updated = build_evidence_clusters(evidence, [claim])

    assert updated[0].status == "verified"
    assert updated[0].verified_claim_ids == ["claim_a"]


def _evidence(
    evidence_id: str,
    paper: str,
    dimension: str,
    fact: str,
    source_subtype: str = "unclassified",
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        run_id="run_test",
        url=f"/papers/{evidence_id}.pdf",
        title=paper,
        content=fact,
        fetched_at=datetime.now(timezone.utc),
        source_type="academic_paper",
        dimension=dimension,
        dimension_label=dimension,
        paper=paper,
        fact=fact,
        quote=fact,
        source_title=paper,
        source_url=f"/papers/{evidence_id}.pdf",
        source_subtype=source_subtype,
        confidence=0.82,
    )
