from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _script_module(name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _fresh_artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "004"
    claim = {
        "claim_id": "c1",
        "run_id": "fresh",
        "dimension": "kg_rag_architecture",
        "dimension_label": "KG-RAG 架构机制",
        "claim": "Four independent papers support a scoped comparative KG-RAG architecture claim.",
        "supporting_evidence_ids": ["ev1", "ev2", "ev3", "ev4"],
        "confidence": 0.82,
        "risk_level": "low",
        "reasoning_summary": "Multi-paper scoped synthesis.",
        "claim_type": "comparative",
        "source_paper_count": 4,
        "evidence_support_level": "strong",
        "supporting_source_subtypes": ["core_kg_rag_method"],
        "supporting_source_subtype_counts": {"core_kg_rag_method": 4},
        "supporting_source_subtype_paper_counts": {"core_kg_rag_method": 4},
        "verification_status": "verified",
        "final_wording": "Four independent KG-RAG papers support a scoped architecture claim.",
    }
    report = "\n\n".join(
        [
            "# Fresh Pilot",
            "## 摘要\n\nScoped claim [1].",
            "## 形式化证据门控与数学背书\n\ng(c)=1 with |E_c|=4 [1].",
            "## 可投稿研究命题与实验化路径\n\nBenchmark perturbation is defined [1].",
            "## 反例与负结果审计\n\nHard-negative coverage: 1/1 [1].",
            "## 可证伪实验设计与负结果记录\n\nfailure_observation and decision are logged [1].",
            "## 可检验研究假设与学术背书\n\nHypothesis and benchmark are explicit [1].",
            "## 参考来源\n\n[1] Paper A. /tmp/a.pdf",
            "## Evidence 附录\n\n[1] Evidence text.",
        ]
    )
    (artifact / "reports").mkdir(parents=True)
    (artifact / "reports" / "report.md").write_text(report, encoding="utf-8")
    (artifact / "reports" / "executive_summary.md").write_text("Report-ready Findings [1].", encoding="utf-8")
    (artifact / "reports" / "methodology.md").write_text("Deterministic methodology [1].", encoding="utf-8")
    _write_json(artifact / "exports" / "claims.json", [claim])
    _write_json(
        artifact / "exports" / "counterexample_audit.json",
        [
            {
                "audit_id": "cex1",
                "target_claim_id": "c1",
                "target_dimension": "kg_rag_architecture",
                "target_dimension_label": "KG-RAG 架构机制",
                "source_title": "Boundary paper",
                "counterexample_type": "hard_negative_boundary",
                "semantic_quality": 0.9,
                "report_visible": True,
                "boundary_challenge": "Tests whether the mechanism holds outside core KG-RAG.",
            }
        ],
    )
    _write_json(
        artifact / "exports" / "falsification_plan.json",
        [
            {
                "plan_id": "fp1",
                "target_claim_id": "c1",
                "target_dimension": "kg_rag_architecture",
                "target_dimension_label": "KG-RAG 架构机制",
                "target_claim_summary": "Scoped architecture claim.",
                "evidence_certificate": "|E_c|=4",
                "falsification_criterion": "Fails if non-KG-RAG baselines match the claimed gain.",
                "benchmark_or_task_perturbation": "Swap KG grounding with text-only retrieval.",
                "expected_failure_mode": "No improvement under perturbation.",
                "negative_result_logging_schema": {"failure_observation": "no gain", "decision": "falsified"},
            }
        ],
    )
    _write_json(artifact / "exports" / "matrix.json", {"dimensions": ["kg_rag_architecture"], "papers": ["Paper A"], "cells": [{"status": "strong"}]})
    _write_json(artifact / "exports" / "observability.json", {"claim_pass_rate": 1.0, "report_confidence": 0.8})
    _write_json(artifact / "exports" / "recommendations.json", [{"next_steps": ["run benchmark", "log negative result"]}])
    _write_json(artifact / "sources" / "accepted_sources.json", [{"title": "Paper A", "source_subtype": "core_kg_rag_method", "relevance_score": 0.9, "reranker_score": 4.0}])
    _write_json(artifact / "sources" / "rejected_sources.json", [{"title": "Drift Paper", "source_subtype": "kg_construction", "rejection_reason": "adjacent"}])
    _write_json(
        artifact / "summary.json",
        {
            "artifact_kind": "fresh_deterministic_local",
            "topic_id": "004",
            "topic": "RAG with Knowledge Graphs",
            "accepted_count": 4,
            "rejected_count": 2,
            "evidence_count": 4,
            "claim_count": 1,
            "verified_claim_count": 1,
            "report_ready_count": 1,
            "accepted_subtypes": {"core_kg_rag_method": 4},
            "counterexample_report_visible_count": 1,
            "falsification_covered_report_ready_claim_count": 1,
            "report_ready_claims": [{"claim_id": "c1"}],
        },
    )
    _write_json(
        artifact / "provenance.json",
        {
            "artifact_kind": "fresh_deterministic_local",
            "git_commit": "abc123",
            "reranker_model": "BAAI/bge-reranker-base",
            "reranker_device_requested": "cuda",
            "reranker_device_loaded": "cuda:0",
            "reranker_error": None,
            "external_llm_calls": False,
        },
    )
    return artifact


def test_pilot_freeze_validator_accepts_fresh_artifact_and_builds_packet(tmp_path: Path) -> None:
    validator = _script_module("pilot_freeze_validator")
    artifact = _fresh_artifact(tmp_path)

    result = validator.validate_artifact(artifact, require_cuda=True)
    packet = validator.mentor_packet(artifact, result)

    assert result["passed"]
    assert result["summary"]["report_ready_count"] == 1
    assert "Mentor Review Packet" in packet
    assert "Human Review Form" in packet


def test_structural_ablation_drops_falsification_score(tmp_path: Path) -> None:
    ablation = _script_module("ablation_quality_table")
    artifact = _fresh_artifact(tmp_path)

    rows = ablation.evaluate_topic(artifact)
    by_variant = {row["variant"]: row for row in rows}

    assert by_variant["full"]["overall_score"] > by_variant["minus_falsification"]["overall_score"]
    assert "missing_falsification_plan" in by_variant["minus_falsification"]["flags"]


def test_runtime_ablation_table_reads_baseline_and_renders(tmp_path: Path) -> None:
    runtime_ablation = _script_module("runtime_ablation_table")
    artifact = _fresh_artifact(tmp_path / "baseline")

    rows = runtime_ablation.baseline_rows(artifact.parent)
    rendered = runtime_ablation.render_markdown(rows)

    assert rows[0]["variant"] == "full"
    assert rows[0]["topic_id"] == "004"
    assert "Runtime Ablation Table" in rendered
    assert "RAG with Knowledge Graphs" in rendered


def test_paper_experiment_packet_summarizes_deltas() -> None:
    packet = _script_module("paper_experiment_packet")

    structural = packet.structural_delta_summary(
        [
            {"topic_id": "004", "variant": "full", "overall_score": 1.0},
            {"topic_id": "004", "variant": "minus_gate", "overall_score": 0.8},
            {"topic_id": "006", "variant": "full", "overall_score": 0.9},
            {"topic_id": "006", "variant": "minus_gate", "overall_score": 0.75},
        ]
    )
    runtime = packet.runtime_delta_summary(
        [
            {"topic_id": "004", "variant": "full", "overall_score": 1.0, "report_ready_count": 4},
            {"topic_id": "004", "variant": "no_reranker", "score_delta_vs_full": -0.1, "report_ready_count": 2, "quality_flags": [], "evaluator_flags": []},
            {"topic_id": "006", "variant": "full", "overall_score": 0.9, "report_ready_count": 3},
            {"topic_id": "006", "variant": "no_reranker", "score_delta_vs_full": -0.2, "report_ready_count": 2, "quality_flags": ["source_drift"], "evaluator_flags": []},
        ]
    )
    gate = packet.formal_gate_note()
    skeleton = packet.paper_skeleton_note()
    intro = packet.introduction_outline_note()

    assert structural == [
        {"variant": "minus_gate", "mean_delta": -0.175, "min_delta": -0.2, "max_delta": -0.15, "topic_count": 2}
    ]
    assert runtime[0]["variant"] == "no_reranker"
    assert runtime[0]["mean_score_delta"] == -0.15
    assert runtime[0]["mean_report_ready_delta"] == -1.5
    assert "source_drift" in runtime[0]["flags"]
    assert "not a guarantee" in " ".join(gate["guarantee"])
    assert skeleton["paper_type"] == "New Problem/Setting paper with a technique contribution"
    assert all(status == "pass" for status in skeleton["self_consistency_checks"].values())
    assert intro["type_positioning"]["type"] == "New Problem/Setting Paper"
    assert intro["flowchart_consistency"]["running_example_loop"] == "pass"
    assert len(intro["paragraphs"][-1]["contributions"]) == 4
