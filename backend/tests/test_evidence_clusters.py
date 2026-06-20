"""Evidence cluster construction and cluster-backed claims."""

from __future__ import annotations

from datetime import datetime, timezone

from cg.orchestrator.pipeline import apply_claim_discipline, build_claims_from_clusters, build_evidence_clusters
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
    assert "跨论文对比性观察" in claim.claim
    assert "共同模式" not in claim.claim
    assert "提供了相近证据" not in claim.claim
    assert set(claim.supporting_evidence_ids) == {"ev_a1", "ev_b1"}
    assert claim.supporting_source_subtypes == ["kgqa_or_graph_reasoning"]
    assert claim.supporting_source_subtype_counts == {"kgqa_or_graph_reasoning": 2}


def test_claim_discipline_backlogs_unsupported_source_roles() -> None:
    evidence = [
        _evidence(
            "ev_a1",
            "Paper A",
            "probabilistic_modeling",
            "Causal discovery estimates graph structure.",
            source_subtype="causal_inference_adjacent",
        ),
        _evidence(
            "ev_b1",
            "Paper B",
            "probabilistic_modeling",
            "Interventional distributions are estimated.",
            source_subtype="causal_inference_adjacent",
        ),
    ]
    claim = Claim(
        claim_id="claim_a",
        run_id="run_test",
        dimension="probabilistic_modeling",
        dimension_label="probabilistic_modeling",
        claim="Causal-adjacent papers show a shared probabilistic pattern.",
        supporting_evidence_ids=["ev_a1", "ev_b1"],
        confidence=0.8,
        risk_level="low",
        reasoning_summary="Supported by two papers.",
        verification_status="verified",
    )

    reviewed = apply_claim_discipline([claim], evidence)

    assert reviewed[0].verification_status == "needs_evidence"
    assert reviewed[0].backlog_reason == "unsupported_source_role"
    assert reviewed[0].risk_level == "medium"
    assert reviewed[0].supporting_source_subtypes == ["causal_inference_adjacent"]
    assert reviewed[0].supporting_source_subtype_counts == {"causal_inference_adjacent": 2}


def test_graph_rag_adjacent_sources_do_not_create_core_synthesis() -> None:
    evidence = [
        _evidence(
            "ev_a1",
            "Graph RAG Paper A",
            "inference_time_control",
            "Graph-RAG search traversal retrieves paths for LLM answers.",
            source_subtype="graph_retrieval_rag_adjacent",
        ),
        _evidence(
            "ev_a2",
            "Graph RAG Paper A",
            "inference_time_control",
            "The system uses search and retrieval over graph context.",
            source_subtype="graph_retrieval_rag_adjacent",
        ),
        _evidence(
            "ev_b1",
            "Graph RAG Paper B",
            "inference_time_control",
            "RAG over knowledge graphs uses search to improve answer grounding.",
            source_subtype="graph_retrieval_rag_adjacent",
        ),
        _evidence(
            "ev_b2",
            "Graph RAG Paper B",
            "inference_time_control",
            "The method is evaluated as retrieval-augmented generation with graph search.",
            source_subtype="graph_retrieval_rag_adjacent",
        ),
    ]
    clusters = build_evidence_clusters(evidence)

    claims = build_claims_from_clusters("run_test", clusters, evidence)

    assert [claim.claim_type for claim in claims] == [
        "single_paper_observation",
        "single_paper_observation",
    ]
    assert not [claim for claim in claims if claim.claim_type == "comparative"]


def test_claim_discipline_backlogs_graph_rag_adjacent_verified_claims() -> None:
    evidence = [
        _evidence(
            "ev_a1",
            "Graph RAG Paper A",
            "inference_time_control",
            "Graph-RAG traversal retrieves paths for LLM answers.",
            source_subtype="graph_retrieval_rag_adjacent",
        ),
        _evidence(
            "ev_b1",
            "Graph RAG Paper B",
            "inference_time_control",
            "RAG over knowledge graphs improves answer grounding.",
            source_subtype="graph_retrieval_rag_adjacent",
        ),
    ]
    claim = Claim(
        claim_id="claim_graph_rag",
        run_id="run_test",
        dimension="inference_time_control",
        dimension_label="inference_time_control",
        claim="Graph-RAG papers support a multi-hop graph reasoning pattern.",
        supporting_evidence_ids=["ev_a1", "ev_b1"],
        confidence=0.8,
        risk_level="low",
        reasoning_summary="Supported by two papers.",
        verification_status="verified",
    )

    reviewed = apply_claim_discipline([claim], evidence)

    assert reviewed[0].verification_status == "needs_evidence"
    assert reviewed[0].backlog_reason == "unsupported_source_role"
    assert reviewed[0].risk_level == "medium"


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
