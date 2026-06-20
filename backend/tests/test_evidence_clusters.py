"""Evidence cluster construction and cluster-backed claims."""

from __future__ import annotations

from datetime import datetime, timezone

from cg.agents.research_agents import deterministic_red_team
from cg.orchestrator.pipeline import (
    apply_claim_discipline,
    build_claims_from_clusters,
    build_evidence_clusters,
    claim_report_ready_reason,
    is_report_ready_claim,
    prepare_claims_for_review,
)
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
    assert "不单独构成领域趋势" in claim.claim
    assert "共同模式" not in claim.claim
    assert "提供了相近证据" not in claim.claim
    assert "相关论文文本中出现了可核验表述" not in claim.claim
    assert set(claim.supporting_evidence_ids) == {"ev_a1", "ev_b1"}
    assert claim.supporting_source_subtypes == ["kgqa_or_graph_reasoning"]
    assert claim.supporting_source_subtype_counts == {"kgqa_or_graph_reasoning": 2}
    assert claim.supporting_source_subtype_paper_counts == {"kgqa_or_graph_reasoning": 2}


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
    assert reviewed[0].supporting_source_subtype_paper_counts == {"causal_inference_adjacent": 2}


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


def test_prepare_claims_rewrites_mixed_roles_as_cross_role_contrast() -> None:
    evidence = [
        _evidence(
            "ev_a1",
            "Benchmark Paper",
            "llm_causal_benchmarking",
            "The benchmark isolates causal reasoning failures.",
            source_subtype="causal_reasoning_benchmark",
        ),
        _evidence(
            "ev_b1",
            "Core Method Paper",
            "llm_causal_benchmarking",
            "The method introduces causal graph prompting.",
            source_subtype="core_llm_causal_reasoning",
        ),
    ]
    claim = Claim(
        claim_id="claim_mixed",
        run_id="run_test",
        dimension="llm_causal_benchmarking",
        dimension_label="LLM causal benchmarking",
        claim="Causal reasoning papers show a shared trend across benchmarks and methods.",
        supporting_evidence_ids=["ev_a1", "ev_b1"],
        confidence=0.86,
        risk_level="low",
        reasoning_summary="Supported by benchmark and method papers.",
        verification_status="draft",
    )

    prepared = prepare_claims_for_review([claim], evidence)
    reviewed = apply_claim_discipline(prepared, evidence)

    assert reviewed[0].claim_type == "cross_role_contrast"
    assert reviewed[0].backlog_reason == ""
    assert reviewed[0].confidence == 0.72
    assert reviewed[0].risk_level == "medium"
    assert reviewed[0].supporting_source_subtypes == [
        "causal_reasoning_benchmark",
        "core_llm_causal_reasoning",
    ]
    assert reviewed[0].supporting_source_subtype_paper_counts == {
        "causal_reasoning_benchmark": 1,
        "core_llm_causal_reasoning": 1,
    }
    assert "跨 source-role 对照" in reviewed[0].claim
    assert "LLM causal reasoning benchmark" in reviewed[0].claim
    assert "core LLM causal reasoning" in reviewed[0].claim
    assert "不作为统一领域趋势" in reviewed[0].claim


def test_deterministic_red_team_preserves_cross_role_contrast_type() -> None:
    evidence = [
        _evidence(
            "ev_a1",
            "Benchmark Paper",
            "llm_causal_benchmarking",
            "The benchmark isolates causal reasoning failures.",
            source_subtype="causal_reasoning_benchmark",
        ),
        _evidence(
            "ev_b1",
            "Core Method Paper",
            "llm_causal_benchmarking",
            "The method introduces causal graph prompting.",
            source_subtype="core_llm_causal_reasoning",
        ),
    ]
    claim = Claim(
        claim_id="claim_cross_role",
        run_id="run_test",
        dimension="llm_causal_benchmarking",
        dimension_label="LLM causal benchmarking",
        claim="作为跨 source-role 对照，benchmark 论文侧重评测协议，core method 论文侧重机制设计。",
        supporting_evidence_ids=["ev_a1", "ev_b1"],
        confidence=0.72,
        risk_level="medium",
        reasoning_summary="Supported by benchmark and method papers.",
        claim_type="cross_role_contrast",
    )

    reviewed = deterministic_red_team([claim], evidence)
    disciplined = apply_claim_discipline(reviewed, evidence)

    assert disciplined[0].claim_type == "cross_role_contrast"
    assert disciplined[0].verification_status == "verified"
    assert disciplined[0].backlog_reason == ""
    assert claim_report_ready_reason(disciplined[0]) == "insufficient_cross_role_total_papers"


def test_cross_role_contrast_report_ready_requires_role_level_paper_support() -> None:
    valid = Claim(
        claim_id="claim_cross_role_ready",
        run_id="run_test",
        dimension="llm_causal_benchmarking",
        dimension_label="LLM causal benchmarking",
        claim=(
            "Benchmark sources emphasize evaluation protocols, while core method sources emphasize "
            "causal graph prompting mechanisms across independent papers."
        ),
        supporting_evidence_ids=["ev_a1", "ev_a2", "ev_b1", "ev_b2"],
        confidence=0.82,
        risk_level="low",
        reasoning_summary="Supported by two benchmark papers and two core method papers.",
        verification_status="verified",
        claim_type="cross_role_contrast",
        source_paper_count=4,
        evidence_support_level="strong",
        supporting_source_subtypes=[
            "causal_reasoning_benchmark",
            "core_llm_causal_reasoning",
        ],
        supporting_source_subtype_counts={
            "causal_reasoning_benchmark": 2,
            "core_llm_causal_reasoning": 2,
        },
        supporting_source_subtype_paper_counts={
            "causal_reasoning_benchmark": 2,
            "core_llm_causal_reasoning": 2,
        },
    )
    thin = valid.model_copy(
        update={
            "claim_id": "claim_cross_role_thin",
            "source_paper_count": 3,
            "supporting_source_subtype_paper_counts": {
                "causal_reasoning_benchmark": 2,
                "core_llm_causal_reasoning": 1,
            },
        }
    )
    imbalanced = valid.model_copy(
        update={
            "claim_id": "claim_cross_role_imbalanced",
            "source_paper_count": 4,
            "supporting_source_subtypes": [
                "causal_reasoning_benchmark",
                "core_llm_causal_reasoning",
                "llm_counterfactual_reasoning",
            ],
            "supporting_source_subtype_counts": {
                "causal_reasoning_benchmark": 2,
                "core_llm_causal_reasoning": 1,
                "llm_counterfactual_reasoning": 1,
            },
            "supporting_source_subtype_paper_counts": {
                "causal_reasoning_benchmark": 2,
                "core_llm_causal_reasoning": 1,
                "llm_counterfactual_reasoning": 1,
            },
        }
    )

    assert claim_report_ready_reason(valid) == ""
    assert is_report_ready_claim(valid)
    assert claim_report_ready_reason(thin) == "insufficient_cross_role_total_papers"
    assert claim_report_ready_reason(imbalanced) == "insufficient_cross_role_role_papers"


def test_report_ready_claim_requires_more_than_audit_verified() -> None:
    ready = Claim(
        claim_id="claim_ready",
        run_id="run_test",
        dimension="llm_causal_benchmarking",
        dimension_label="LLM causal benchmarking",
        claim="Two benchmark papers independently show that causal reasoning evaluations isolate intervention and counterfactual failure modes.",
        supporting_evidence_ids=["ev_a1", "ev_b1"],
        confidence=0.82,
        risk_level="low",
        reasoning_summary="Supported by two benchmark papers.",
        verification_status="verified",
        claim_type="comparative",
        source_paper_count=2,
        evidence_support_level="strong",
        supporting_source_subtypes=["causal_reasoning_benchmark"],
        supporting_source_subtype_counts={"causal_reasoning_benchmark": 2},
    )
    audit_only = ready.model_copy(
        update={
            "claim_id": "claim_audit_only",
            "claim": "作为跨论文对比性观察，benchmark 来源呈现互补切入点。这只说明当前样本内的机制差异，不单独构成领域趋势。",
        }
    )

    assert is_report_ready_claim(ready)
    assert claim_report_ready_reason(ready) == ""
    assert not is_report_ready_claim(audit_only)
    assert claim_report_ready_reason(audit_only) == "sample_limited_observation"


def test_high_support_same_role_cluster_can_create_report_ready_synthesis() -> None:
    evidence = [
        _evidence(
            "ev_a1",
            "Benchmark Paper A",
            "data_evaluation_engineering",
            "The benchmark defines task datasets and evaluation metrics.",
            source_subtype="scientific_reasoning_benchmark",
        ),
        _evidence(
            "ev_b1",
            "Benchmark Paper B",
            "data_evaluation_engineering",
            "The dataset introduces evaluation tasks and benchmark metrics.",
            source_subtype="scientific_reasoning_benchmark",
        ),
        _evidence(
            "ev_c1",
            "Benchmark Paper C",
            "data_evaluation_engineering",
            "The evaluation protocol compares benchmark task performance.",
            source_subtype="scientific_reasoning_benchmark",
        ),
        _evidence(
            "ev_d1",
            "Benchmark Paper D",
            "data_evaluation_engineering",
            "The task benchmark reports dataset-level evaluation metrics.",
            source_subtype="scientific_reasoning_benchmark",
        ),
    ]
    clusters = build_evidence_clusters(evidence)

    claims = deterministic_red_team(
        build_claims_from_clusters("run_test", clusters, evidence, limit=8),
        evidence,
    )
    report_ready = [claim for claim in claims if is_report_ready_claim(claim)]

    assert len(report_ready) == 1
    claim = report_ready[0]
    assert claim.claim_type == "comparative"
    assert claim.source_paper_count == 4
    assert "范围限定综合结论" in claim.claim
    assert "不单独构成领域趋势" not in claim.claim
    assert claim_report_ready_reason(claim) == ""


def test_counterfactual_treatment_effect_evidence_uses_topic_specific_cluster_key() -> None:
    evidence = [
        _evidence(
            "ev_a1",
            "Treatment Paper A",
            "treatment_effect_estimation",
            "The paper estimates heterogeneous treatment effects under potential outcomes.",
            source_subtype="treatment_effect_estimation",
        ),
        _evidence(
            "ev_b1",
            "Treatment Paper B",
            "treatment_effect_estimation",
            "The method predicts counterfactual outcomes for individual treatment effect estimation.",
            source_subtype="treatment_effect_estimation",
        ),
        _evidence(
            "ev_c1",
            "Treatment Paper C",
            "treatment_effect_estimation",
            "The model studies causal effect estimation with counterfactual regression.",
            source_subtype="treatment_effect_estimation",
        ),
        _evidence(
            "ev_d1",
            "Treatment Paper D",
            "treatment_effect_estimation",
            "The evaluation measures treatment effect estimation across potential outcome settings.",
            source_subtype="treatment_effect_estimation",
        ),
    ]
    clusters = build_evidence_clusters(evidence)

    claims = deterministic_red_team(
        build_claims_from_clusters("run_test", clusters, evidence, limit=8),
        evidence,
    )
    report_ready = [claim for claim in claims if is_report_ready_claim(claim)]

    assert clusters[0].label == "treatment-effect estimation protocol"
    assert clusters[0].independent_paper_count == 4
    assert len(report_ready) == 1
    assert report_ready[0].supporting_source_subtype_paper_counts == {
        "treatment_effect_estimation": 4
    }


def test_multi_hop_kgqa_evidence_uses_topic_specific_cluster_key() -> None:
    evidence = [
        _evidence(
            "ev_a1",
            "KGQA Paper A",
            "graph_reasoning_benchmark",
            "The system solves knowledge graph question answering through semantic parsing.",
            source_subtype="kgqa_or_graph_reasoning",
        ),
        _evidence(
            "ev_b1",
            "KGQA Paper B",
            "graph_reasoning_benchmark",
            "The method generates logical form queries for graph question answering.",
            source_subtype="kgqa_or_graph_reasoning",
        ),
        _evidence(
            "ev_c1",
            "KGQA Paper C",
            "graph_reasoning_benchmark",
            "The approach handles multi-hop question answering over knowledge graphs.",
            source_subtype="kgqa_or_graph_reasoning",
        ),
        _evidence(
            "ev_d1",
            "KGQA Paper D",
            "graph_reasoning_benchmark",
            "The pipeline constructs executable queries for KGQA reasoning.",
            source_subtype="kgqa_or_graph_reasoning",
        ),
    ]
    clusters = build_evidence_clusters(evidence)

    claims = deterministic_red_team(
        build_claims_from_clusters("run_test", clusters, evidence, limit=8),
        evidence,
    )
    report_ready = [claim for claim in claims if is_report_ready_claim(claim)]

    assert clusters[0].label == "KGQA and semantic parsing pipeline"
    assert clusters[0].independent_paper_count == 4
    assert len(report_ready) == 1
    assert report_ready[0].supporting_source_subtype_paper_counts == {
        "kgqa_or_graph_reasoning": 4
    }


def test_math_proof_verification_evidence_uses_topic_specific_cluster_key() -> None:
    evidence = [
        _evidence(
            "ev_a1",
            "Proof Paper A",
            "math_benchmark_evaluation",
            "The benchmark evaluates natural language math proofs with proof verification.",
            source_subtype="formal_math_proving",
        ),
        _evidence(
            "ev_b1",
            "Proof Paper B",
            "formal_proof_symbolic_reasoning",
            "The method uses Lean proof assistants for formal proof checking.",
            source_subtype="formal_math_proving",
        ),
        _evidence(
            "ev_c1",
            "Proof Paper C",
            "self_consistency_search_verification",
            "The verifier checks theorem proving steps against ground-truth proof structure.",
            source_subtype="formal_math_proving",
        ),
        _evidence(
            "ev_d1",
            "Proof Paper D",
            "natural_language_to_formal_math",
            "The system translates natural language math proofs into formal verification targets.",
            source_subtype="formal_math_proving",
        ),
    ]
    clusters = build_evidence_clusters(evidence)

    claims = deterministic_red_team(
        build_claims_from_clusters("run_test", clusters, evidence, limit=8),
        evidence,
    )
    report_ready = [claim for claim in claims if is_report_ready_claim(claim)]

    assert len(clusters) == 1
    assert clusters[0].dimension == "formal_proof_symbolic_reasoning"
    assert clusters[0].label == "mathematical proof verification protocol"
    assert clusters[0].independent_paper_count == 4
    assert len(report_ready) == 1
    assert report_ready[0].supporting_source_subtype_paper_counts == {
        "formal_math_proving": 4
    }


def test_math_cluster_keys_do_not_override_non_math_dimensions() -> None:
    evidence = [
        _evidence(
            "ev_a1",
            "Causal Paper A",
            "llm_causal_benchmarking",
            "The paper mentions Lean proof checking only as an unrelated example.",
            source_subtype="causal_reasoning_benchmark",
        ),
        _evidence(
            "ev_b1",
            "Causal Paper B",
            "llm_causal_benchmarking",
            "The evaluation discusses proof verification as a contrastive task.",
            source_subtype="causal_reasoning_benchmark",
        ),
    ]

    clusters = build_evidence_clusters(evidence)

    assert {cluster.dimension for cluster in clusters} == {"llm_causal_benchmarking"}
    assert "mathematical proof verification protocol" not in {
        cluster.label for cluster in clusters
    }


def test_scientific_workflow_evidence_uses_topic_specific_cluster_key() -> None:
    evidence = [
        _evidence(
            "ev_a1",
            "Discovery Agent A",
            "lab_workflow_reasoning",
            "The agent orchestrates a complete research workflow from hypothesis generation to experiment planning.",
            source_subtype="scientific_discovery_agent",
        ),
        _evidence(
            "ev_b1",
            "Discovery Agent B",
            "tool_augmented_scientific_reasoning",
            "The multi-agent architecture coordinates data-driven discovery and scientific workflow actions.",
            source_subtype="scientific_discovery_agent",
        ),
        _evidence(
            "ev_c1",
            "Discovery Agent C",
            "domain_grounding_verification",
            "The system acts as a scientific partner for biological discovery pipeline design.",
            source_subtype="scientific_discovery_agent",
        ),
        _evidence(
            "ev_d1",
            "Discovery Agent D",
            "scientific_problem_benchmarking",
            "The discovery process evaluates hypothesis generation and experimental design tasks.",
            source_subtype="scientific_discovery_agent",
        ),
    ]
    clusters = build_evidence_clusters(evidence)

    claims = deterministic_red_team(
        build_claims_from_clusters("run_test", clusters, evidence, limit=8),
        evidence,
    )
    report_ready = [claim for claim in claims if is_report_ready_claim(claim)]

    assert len(clusters) == 1
    assert clusters[0].dimension == "lab_workflow_reasoning"
    assert clusters[0].label == "scientific discovery workflow protocol"
    assert clusters[0].independent_paper_count == 4
    assert len(report_ready) == 1
    assert report_ready[0].supporting_source_subtype_paper_counts == {
        "scientific_discovery_agent": 4
    }


def test_scientific_cluster_keys_do_not_override_non_scientific_dimensions() -> None:
    evidence = [
        _evidence(
            "ev_a1",
            "Causal Paper A",
            "llm_causal_benchmarking",
            "The paper mentions hypothesis generation only as a contrastive example.",
            source_subtype="causal_reasoning_benchmark",
        ),
        _evidence(
            "ev_b1",
            "Causal Paper B",
            "llm_causal_benchmarking",
            "The system discusses agentic workflow as unrelated background.",
            source_subtype="causal_reasoning_benchmark",
        ),
    ]

    clusters = build_evidence_clusters(evidence)

    assert {cluster.dimension for cluster in clusters} == {"llm_causal_benchmarking"}
    assert "scientific discovery workflow protocol" not in {
        cluster.label for cluster in clusters
    }


def test_scientific_workflow_cluster_can_create_cross_role_report_ready_contrast() -> None:
    evidence = [
        _evidence(
            "ev_a1",
            "Benchmark Paper A",
            "lab_workflow_reasoning",
            "The benchmark defines hypothesis generation and experimental design tasks.",
            source_subtype="scientific_reasoning_benchmark",
        ),
        _evidence(
            "ev_b1",
            "Benchmark Paper B",
            "lab_workflow_reasoning",
            "The benchmark organizes data-driven discovery workflow stages from hypothesis generation to experiment design.",
            source_subtype="scientific_reasoning_benchmark",
        ),
        _evidence(
            "ev_c1",
            "Agent Paper C",
            "tool_augmented_scientific_reasoning",
            "The multi-agent architecture coordinates scientific discovery workflow actions.",
            source_subtype="scientific_discovery_agent",
        ),
        _evidence(
            "ev_d1",
            "Agent Paper D",
            "domain_grounding_verification",
            "The agent orchestrates hypothesis generation and experiment planning.",
            source_subtype="scientific_discovery_agent",
        ),
    ]
    clusters = build_evidence_clusters(evidence)

    claims = deterministic_red_team(
        build_claims_from_clusters("run_test", clusters, evidence, limit=8),
        evidence,
    )
    report_ready = [claim for claim in claims if is_report_ready_claim(claim)]
    cross_role = [claim for claim in report_ready if claim.claim_type == "cross_role_contrast"]

    assert len(cross_role) == 1
    claim = cross_role[0]
    assert claim_report_ready_reason(claim) == ""
    assert claim.source_paper_count == 4
    assert claim.supporting_source_subtype_paper_counts == {
        "scientific_discovery_agent": 2,
        "scientific_reasoning_benchmark": 2,
    }
    assert "来源角色分工对照" in claim.claim
    assert "scientific discovery agent/workflow" in claim.claim
    assert "scientific reasoning benchmark" in claim.claim


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
