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
        )
    ]

    diagnostics = audit.quality_diagnostics(accepted, evidence, claims)

    assert diagnostics["accepted_unclassified_count"] == 1
    assert diagnostics["accepted_non_reportable_count"] == 1
    assert diagnostics["accepted_reportable_count"] == 1
    assert diagnostics["other_dimension_ratio"] == 0.667
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
        "no_report_worthy_verified_synthesis",
        "no_verified_claims",
    ]
