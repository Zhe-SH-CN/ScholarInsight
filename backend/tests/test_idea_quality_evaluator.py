from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _evaluator_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "idea_quality_evaluator.py"
    spec = importlib.util.spec_from_file_location("idea_quality_evaluator", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def test_idea_quality_evaluator_scores_falsification_artifact_above_baseline(tmp_path: Path) -> None:
    evaluator = _evaluator_module()
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"

    (baseline / "reports").mkdir(parents=True)
    (baseline / "claims").mkdir(parents=True)
    (baseline / "reports" / "report.md").write_text(
        "# Baseline\n\nA weak report mentions a possible direction [1].\n\n"
        "## 参考来源\n\n[1] 000. /tmp/paper.pdf\n",
        encoding="utf-8",
    )
    (baseline / "claims" / "_index.jsonl").write_text(
        json.dumps(
            {
                "claim_id": "c_old",
                "claim": "A broad unsupported claim.",
                "verification_status": "challenged",
                "risk_level": "high",
                "supporting_evidence_ids": ["ev_old"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    (candidate / "reports").mkdir(parents=True)
    (candidate / "reports" / "report.md").write_text(
        "# Candidate\n\n## 形式化证据门控与数学背书\n\ng(c)=1.\n\n"
        "## 可证伪实验设计与负结果记录\n\n"
        "Falsification coverage: 1/1. failure_observation and decision are logged [1].\n\n"
        "## 可检验研究假设与学术背书\n\nbenchmark task perturbation is explicit [1].\n\n"
        "## Evidence 附录\n\n[1] Evidence.\n",
        encoding="utf-8",
    )
    _write_json(
        candidate / "exports" / "claims.json",
        [
            {
                "claim_id": "c_new",
                "claim": "Four papers support a scoped comparative claim.",
                "claim_type": "comparative",
                "verification_status": "verified",
                "risk_level": "low",
                "supporting_evidence_ids": ["ev1", "ev2", "ev3", "ev4"],
                "source_paper_count": 4,
                "evidence_support_level": "strong",
                "supporting_source_subtype_paper_counts": {"core_method": 4},
            }
        ],
    )
    _write_json(candidate / "exports" / "counterexample_audit.json", [{"target_claim_id": "c_new", "counterexample_type": "hard_negative_boundary", "report_visible": True}])
    _write_json(candidate / "exports" / "falsification_plan.json", [{"target_claim_id": "c_new", "negative_result_logging_schema": {"failure_observation": "no gain", "decision": "falsified"}}])
    _write_json(candidate / "exports" / "observability.json", {"claim_pass_rate": 1.0, "report_confidence": 0.7})
    _write_json(candidate / "exports" / "matrix.json", {"dimensions": ["d1"], "papers": ["p1"], "cells": [{"status": "strong"}]})
    _write_json(candidate / "exports" / "recommendations.json", [{"next_steps": ["run benchmark", "log negative result"]}])
    _write_json(candidate / "summary.json", {"report_ready_count": 1, "accepted_subtypes": {"core_method": 4}})

    baseline_result = evaluator.load_artifact_dir(baseline, "baseline")
    candidate_result = evaluator.load_artifact_dir(candidate, "candidate")
    pairs = evaluator.build_pairs([baseline_result, candidate_result], ["baseline:candidate"])

    assert candidate_result["overall_score"] > baseline_result["overall_score"]
    assert candidate_result["scores"]["falsifiability"] >= 0.8
    assert "missing_falsification_plan" in baseline_result["flags"]
    assert pairs[0]["winner"] == "candidate"


def test_idea_quality_evaluator_separates_body_and_appendix_citations(tmp_path: Path) -> None:
    evaluator = _evaluator_module()
    artifact = tmp_path / "appendix_heavy"

    (artifact / "reports").mkdir(parents=True)
    (artifact / "reports" / "report.md").write_text(
        "# Candidate\n\n## 形式化证据门控与数学背书\n\ng(c)=1.\n\n"
        "## 可证伪实验设计与负结果记录\n\n"
        "Falsification coverage: 1/1. failure_observation and decision are logged.\n\n"
        "## 可检验研究假设与学术背书\n\nbenchmark task perturbation is explicit.\n\n"
        "## Evidence 附录\n\n"
        "[1] Evidence. [2] Evidence. [3] Evidence. [4] Evidence.\n"
        "- 事实：Paper A 在核心方法相关论文文本中出现了可核验表述：Mechanism evidence.\n",
        encoding="utf-8",
    )
    _write_json(
        artifact / "exports" / "claims.json",
        [
            {
                "claim_id": "c_new",
                "claim": "Four papers support a scoped comparative claim.",
                "claim_type": "comparative",
                "verification_status": "verified",
                "risk_level": "low",
                "supporting_evidence_ids": ["ev1", "ev2", "ev3", "ev4"],
                "source_paper_count": 4,
                "evidence_support_level": "strong",
                "supporting_source_subtype_paper_counts": {"core_method": 4},
            }
        ],
    )
    _write_json(artifact / "exports" / "counterexample_audit.json", [{"target_claim_id": "c_new", "counterexample_type": "hard_negative_boundary", "report_visible": True}])
    _write_json(artifact / "exports" / "falsification_plan.json", [{"target_claim_id": "c_new", "negative_result_logging_schema": {"failure_observation": "no gain", "decision": "falsified"}}])
    _write_json(artifact / "exports" / "observability.json", {"claim_pass_rate": 1.0, "report_confidence": 0.7})
    _write_json(artifact / "exports" / "matrix.json", {"dimensions": ["d1"], "papers": ["p1"], "cells": [{"status": "strong"}]})
    _write_json(artifact / "exports" / "recommendations.json", [{"next_steps": ["run benchmark", "log negative result"]}])
    _write_json(artifact / "summary.json", {"report_ready_count": 1, "accepted_subtypes": {"core_method": 4}})

    result = evaluator.load_artifact_dir(artifact, "appendix_heavy")

    assert result["metrics"]["body_citation_count"] == 0
    assert result["metrics"]["appendix_citation_count"] == 4
    assert result["scores"]["evidence_grounding"] < 1.0
    assert "appendix_heavy_citation_dependency" in result["flags"]
    assert "mechanical_appendix_fact_prefix" in result["flags"]
    assert result["verdict"] == "promising but needs revision"
