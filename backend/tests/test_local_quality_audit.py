from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _audit_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "local_quality_audit.py"
    spec = importlib.util.spec_from_file_location("local_quality_audit", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quality_diagnostics_flags_unclassified_sources_and_other_dimensions() -> None:
    audit = _audit_module()
    accepted = [
        SimpleNamespace(source_subtype="core_kg_rag_method"),
        SimpleNamespace(source_subtype="unclassified"),
    ]
    evidence = [
        SimpleNamespace(dimension="other"),
        SimpleNamespace(dimension="other"),
        SimpleNamespace(dimension="inference_time_control"),
    ]
    claims = [
        SimpleNamespace(
            verification_status="verified",
            claim_type="comparative",
            risk_level="low",
            backlog_reason="",
            source_paper_count=2,
            supporting_evidence_ids=["ev_a", "ev_b"],
            evidence_support_level="strong",
            supporting_source_subtypes=["core_kg_rag_method"],
            claim="Two KG-RAG papers independently evaluate graph retrieval grounding.",
            final_wording="",
        )
    ]

    diagnostics = audit.quality_diagnostics(accepted, evidence, claims)

    assert diagnostics["accepted_unclassified_count"] == 1
    assert diagnostics["accepted_non_reportable_count"] == 1
    assert diagnostics["accepted_reportable_count"] == 1
    assert diagnostics["other_dimension_ratio"] == 0.667
    assert diagnostics["audit_verified_synthesis_count"] == 1
    assert diagnostics["report_ready_verified_count"] == 1
    assert diagnostics["report_worthy_verified_count"] == 1
    assert "accepted_sources_include_unclassified_roles" in diagnostics["quality_flags"]
    assert "evidence_dimensions_collapse_to_other" in diagnostics["quality_flags"]
    assert "no_verified_claims" not in diagnostics["quality_flags"]


def test_quality_diagnostics_flags_missing_report_worthy_claims() -> None:
    audit = _audit_module()
    claims = [
        SimpleNamespace(
            verification_status="needs_evidence",
            claim_type="comparative",
            risk_level="medium",
            backlog_reason="red_team_needs_evidence",
        )
    ]

    diagnostics = audit.quality_diagnostics([], [], claims)

    assert diagnostics["report_worthy_verified_count"] == 0
    assert diagnostics["verified_single_paper_observation_count"] == 0
    assert diagnostics["quality_flags"] == [
        "no_audit_verified_synthesis",
        "no_report_ready_verified_synthesis",
        "no_verified_claims",
    ]


def test_quality_diagnostics_separates_audit_verified_from_report_ready() -> None:
    audit = _audit_module()
    claims = [
        SimpleNamespace(
            verification_status="verified",
            claim_type="comparative",
            risk_level="low",
            backlog_reason="",
            source_paper_count=2,
            supporting_evidence_ids=["ev_a", "ev_b"],
            evidence_support_level="strong",
            supporting_source_subtypes=["scientific_reasoning_benchmark"],
            claim="作为跨论文对比性观察，这只说明当前样本内的机制差异，不单独构成领域趋势。",
            final_wording="",
        )
    ]

    diagnostics = audit.quality_diagnostics([], [], claims)

    assert diagnostics["audit_verified_synthesis_count"] == 1
    assert diagnostics["report_ready_verified_count"] == 0
    assert diagnostics["report_ready_rejection_reasons"] == {
        "sample_limited_observation": 1
    }
    assert diagnostics["quality_flags"] == [
        "no_report_ready_verified_synthesis",
        "audit_verified_claims_not_report_ready",
    ]


def test_claim_row_exports_report_ready_reason() -> None:
    audit = _audit_module()
    claim = SimpleNamespace(
        model_dump=lambda **_: {
            "claim_id": "claim_audit_only",
            "claim_type": "comparative",
        },
        verification_status="verified",
        risk_level="low",
        backlog_reason="",
        claim_type="comparative",
        source_paper_count=2,
        supporting_evidence_ids=["ev_a", "ev_b"],
        evidence_support_level="strong",
        supporting_source_subtypes=["scientific_reasoning_benchmark"],
        claim="作为跨论文对比性观察，这只说明当前样本内的机制差异，不单独构成领域趋势。",
        final_wording="",
    )

    row = audit.claim_row(claim)

    assert row["report_ready_rejection_reason"] == "sample_limited_observation"
    assert row["is_report_ready"] is False
