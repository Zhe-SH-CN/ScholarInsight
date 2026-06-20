"""Red-team model selection behavior."""

from __future__ import annotations

from datetime import datetime, timezone

from cg.orchestrator.pipeline import ResearchPipeline, build_paper_pattern_matrix, select_matrix_papers
from cg.schemas.research import Evidence, ResearchRequest
from cg.settings import Settings


def test_red_team_can_use_gpt55_model_with_gemini_credentials(tmp_path) -> None:
    settings = Settings(
        cg_data_dir=str(tmp_path),
        gemini_api_key="test-key",
        gemini_base_url="https://example.invalid/v1",
        red_team_llm_provider="gpt-5.5",
    )

    pipeline = ResearchPipeline(settings)

    assert pipeline.red_team_llm.provider == "gemini"
    assert pipeline.red_team_llm.model == "gpt-5.5"


def test_red_team_defaults_to_configured_gemini_model(tmp_path) -> None:
    settings = Settings(
        cg_data_dir=str(tmp_path),
        gemini_api_key="test-key",
        gemini_base_url="https://example.invalid/v1",
        gemini_model="gemini-pro-agent",
    )

    pipeline = ResearchPipeline(settings)

    assert pipeline.red_team_llm.provider == "gemini"
    assert pipeline.red_team_llm.model == "gemini-pro-agent"


def test_interim_matrix_uses_real_evidence_papers() -> None:
    request = ResearchRequest(
        project_name="Coverage regression",
        target_topic="RAG with Knowledge Graphs",
        analysis_dimensions=["modular_pipeline_composition"],
    )
    evidence = [
        _evidence("ev_1", "GraphRAG Paper A", "modular_pipeline_composition"),
        _evidence("ev_2", "GraphRAG Paper B", "modular_pipeline_composition"),
    ]

    papers = select_matrix_papers(request, evidence)
    matrix = build_paper_pattern_matrix(evidence, papers, request.analysis_dimensions)

    assert papers == ["GraphRAG Paper A", "GraphRAG Paper B"]
    assert matrix.coverage_by_dimension["modular_pipeline_composition"] > 0


def _evidence(evidence_id: str, paper: str, dimension: str) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        run_id="run_test",
        url=f"/papers/{evidence_id}.pdf",
        title=paper,
        content="Evidence content",
        fetched_at=datetime.now(timezone.utc),
        source_type="academic_paper",
        dimension=dimension,
        dimension_label=dimension,
        paper=paper,
        fact=f"{paper} contains a relevant mechanism.",
        quote="Relevant mechanism.",
        source_title=paper,
        source_url=f"/papers/{evidence_id}.pdf",
        confidence=0.8,
    )
