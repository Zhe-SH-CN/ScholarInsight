"""Topic-family dimension templates for quality-oriented pilots."""

from __future__ import annotations

from datetime import datetime, timezone

from cg.agents.research_agents import deterministic_plan, normalize_claim_dimension, normalize_dimension_key
from cg.orchestrator.pipeline import extract_evidence_from_document
from cg.schemas.research import ResearchRequest, SourceDocument, TOPIC_FAMILY_DIMENSIONS, dimensions_for_topic


def test_research_request_uses_counterfactual_family_dimensions_by_default() -> None:
    request = ResearchRequest(target_topic="Counterfactual Inference")

    assert request.analysis_dimensions == TOPIC_FAMILY_DIMENSIONS["counterfactual_inference"]
    assert "gap_driven_reframing" not in request.analysis_dimensions
    assert "treatment_effect_estimation" in request.analysis_dimensions


def test_explicit_dimensions_are_not_overridden() -> None:
    request = ResearchRequest(
        target_topic="Counterfactual Inference",
        analysis_dimensions=["gap_driven_reframing", "data_evaluation_engineering"],
    )

    assert request.analysis_dimensions == ["gap_driven_reframing", "data_evaluation_engineering"]


def test_rag_with_kg_defaults_do_not_invite_application_cases() -> None:
    request = ResearchRequest(target_topic="RAG with Knowledge Graphs")

    assert request.analysis_dimensions == TOPIC_FAMILY_DIMENSIONS["rag_with_knowledge_graphs"]
    assert "rag_kg_boundary_analysis" in request.analysis_dimensions
    assert "application_boundary_cases" not in request.analysis_dimensions


def test_deterministic_plan_uses_topic_specific_queries() -> None:
    request = ResearchRequest(target_topic="Causal Reasoning with LLMs")

    plan = deterministic_plan(request)

    assert plan.dimensions == TOPIC_FAMILY_DIMENSIONS["causal_reasoning_llms"]
    assert all(task.dimension in plan.dimensions for task in plan.source_tasks)
    assert any("causal reasoning" in task.query.lower() for task in plan.source_tasks)
    assert not any("limitations challenges bottlenecks survey" in task.query.lower() for task in plan.source_tasks)


def test_dimensions_for_multi_hop_graphs_and_label_normalization() -> None:
    dimensions = dimensions_for_topic("Multi-hop Reasoning on Graphs")

    assert dimensions == TOPIC_FAMILY_DIMENSIONS["multi_hop_graph_reasoning"]
    assert normalize_dimension_key("KGQA/图推理") == "kgqa_or_graph_reasoning"
    assert (
        normalize_dimension_key("N/A", "multi-hop graph reasoning semantic parsing over knowledge graphs")
        == "semantic_parsing_grounding"
    )


def test_normalize_counterfactual_dimension_aliases() -> None:
    assert (
        normalize_dimension_key(
            "counterfactual_benchmark_evaluation",
            "benchmark metrics datasets evaluation protocol for counterfactual prediction",
        )
        == "counterfactual_benchmarking"
    )
    assert (
        normalize_dimension_key(
            "counterfactual_explanation_and_fairness",
            "algorithmic recourse and counterfactual fairness",
        )
        == "counterfactual_explanation_fairness"
    )
    assert (
        normalize_dimension_key(
            "identifiability_and_assumption_sensitivity",
            "identifiability assumptions sensitivity analysis",
        )
        == "identifiability_assumption_sensitivity"
    )


def test_normalize_claim_dimension_rejects_meta_claims_across_dimensions() -> None:
    valid = {
        "treatment_effect_estimation",
        "core_counterfactual_inference",
        "counterfactual_explanation_fairness",
    }

    assert (
        normalize_claim_dimension(
            "counterfactual_explanation_and_fairness",
            "counterfactual fairness and recourse",
            ["counterfactual_explanation_fairness"],
            valid,
        )
        == "counterfactual_explanation_fairness"
    )
    assert (
        normalize_claim_dimension(
            "research_distribution",
            "evidence distribution across research areas",
            ["treatment_effect_estimation", "core_counterfactual_inference"],
            valid,
        )
        == ""
    )
    assert (
        normalize_claim_dimension(
            "unclear_dimension",
            "single-dimension observation",
            ["treatment_effect_estimation", "treatment_effect_estimation"],
            valid,
        )
        == "treatment_effect_estimation"
    )


def test_deterministic_evidence_uses_scientific_topic_dimensions() -> None:
    document = _source_document(
        "Scientific Benchmark",
        (
            "The scientific reasoning benchmark defines multiple scientific problem tasks "
            "and reports evaluation metrics for hypothesis generation. "
            "The workflow uses tool-augmented agents for experiment planning and automated discovery."
        ),
        source_subtype="scientific_reasoning_benchmark",
    )

    evidence = extract_evidence_from_document(
        "run_test",
        document,
        dimensions_for_topic("Scientific Reasoning with LLMs"),
        ["Scientific Reasoning with LLMs"],
    )

    dimensions = {item.dimension for item in evidence}
    assert "other" not in dimensions
    assert "scientific_problem_benchmarking" in dimensions
    assert "tool_augmented_scientific_reasoning" in dimensions
    assert {item.paper for item in evidence} == {"Scientific Benchmark"}
    assert all(item.fact.startswith("Scientific Benchmark 在") for item in evidence)


def test_deterministic_evidence_uses_mathematical_topic_dimensions() -> None:
    document = _source_document(
        "Math Proof Benchmark",
        (
            "The mathematical reasoning benchmark evaluates GSM8K and olympiad word problem tasks. "
            "The method adds theorem proof verification with a step verifier and search-based self-consistency."
        ),
        source_subtype="math_reasoning_benchmark",
    )

    evidence = extract_evidence_from_document(
        "run_test",
        document,
        dimensions_for_topic("Mathematical Reasoning"),
        ["Mathematical Reasoning"],
    )

    dimensions = {item.dimension for item in evidence}
    assert "other" not in dimensions
    assert "math_benchmark_evaluation" in dimensions
    assert "formal_proof_symbolic_reasoning" in dimensions
    assert {item.paper for item in evidence} == {"Math Proof Benchmark"}
    assert all(item.fact.startswith("Math Proof Benchmark 在") for item in evidence)


def _source_document(title: str, content: str, source_subtype: str) -> SourceDocument:
    return SourceDocument(
        source_id="src_test",
        run_id="run_test",
        url=f"/papers/{title.replace(' ', '_')}.pdf",
        title=title,
        content=content,
        excerpt=content[:120],
        source_type="academic_paper",
        http_status=200,
        content_hash="hash",
        fetched_at=datetime.now(timezone.utc),
        parser="test",
        provider="local_papers",
        query="test query",
        content_source="test",
        source_score=0.9,
        relevance_score=0.9,
        relevance_label="high",
        source_subtype=source_subtype,
    )
