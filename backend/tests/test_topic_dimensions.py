"""Topic-family dimension templates for quality-oriented pilots."""

from __future__ import annotations

from cg.agents.research_agents import deterministic_plan, normalize_claim_dimension, normalize_dimension_key
from cg.schemas.research import ResearchRequest, TOPIC_FAMILY_DIMENSIONS, dimensions_for_topic


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
