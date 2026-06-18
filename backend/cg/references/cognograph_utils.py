"""CognoGraph shared schema constants and graph I/O helpers."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GRAPH_DIR = Path(".cognograph")
GRAPH_FILE = GRAPH_DIR / "graph.json"

REASONING_PATTERNS = [
    "gap_driven_reframing",
    "cross_domain_synthesis",
    "representation_shift",
    "modular_pipeline_composition",
    "data_evaluation_engineering",
    "principled_probabilistic_modeling",
    "formal_experimental_tightening",
    "approximation_engineering",
    "inference_time_control",
    "structural_inductive_bias",
    "multiscale_hierarchical_modeling",
    "mechanistic_decomposition",
    "adversary_modeling",
    "numerics_systems_codesign",
    "data_centric_optimization",
]

PATTERN_DESCRIPTIONS = {
    "gap_driven_reframing": "痛点驱动重构：将失败/局限转化为显式设计约束",
    "cross_domain_synthesis": "跨领域综合：从邻近领域移植方案并添加兼容层",
    "representation_shift": "表征转换：替换基础表征原语",
    "modular_pipeline_composition": "模块化管线：分解为可组合模块",
    "data_evaluation_engineering": "数据评估工程：新数据集/基准/指标",
    "principled_probabilistic_modeling": "概率建模：概率图模型、贝叶斯推理",
    "formal_experimental_tightening": "理论实验迭代：证明/界限与实验验证迭代",
    "approximation_engineering": "近似工程：有原则的近似（量化、低秩）",
    "inference_time_control": "推理时控制：运行时引导采样",
    "structural_inductive_bias": "结构归纳偏置：硬编码领域结构",
    "multiscale_hierarchical_modeling": "多尺度分层：多层级粒度交互",
    "mechanistic_decomposition": "机制分解：分解为可解释机制",
    "adversary_modeling": "对抗建模：建模对手，设计防御",
    "numerics_systems_codesign": "数值系统协同：算法+硬件协同设计",
    "data_centric_optimization": "数据中心优化：优化数据分布",
}


def current_timestamp() -> str:
    """Return the project timestamp format required by the CognoGraph schema."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H_%M")


def ensure_graph_dir(graph_file: Path | str = GRAPH_FILE) -> None:
    """Create the graph directory for the target graph file."""
    Path(graph_file).parent.mkdir(parents=True, exist_ok=True)


def create_empty_graph(topic: str = "", description: str = "") -> dict[str, Any]:
    """Create an empty graph that conforms to the CognoGraph top-level schema."""
    now = current_timestamp()
    return {
        "metadata": {
            "name": "CognoGraph",
            "version": "0.1.0",
            "created_at": now,
            "updated_at": now,
            "description": description,
            "node_count": 0,
            "edge_count": 0,
            "topic": topic,
        },
        "schema": {
            "reasoning_patterns": REASONING_PATTERNS,
            "reasoning_pattern_descriptions": PATTERN_DESCRIPTIONS,
        },
        "nodes": [],
        "edges": [],
    }


def load_graph(graph_file: Path | str = GRAPH_FILE) -> dict[str, Any]:
    """Load a graph, or return an empty graph if the graph file is missing."""
    path = Path(graph_file)
    if not path.exists():
        return create_empty_graph()
    with path.open("r", encoding="utf-8") as file:
        graph = json.load(file)
    validate_graph(graph)
    return graph


def save_graph(graph: dict[str, Any], graph_file: Path | str = GRAPH_FILE) -> None:
    """Save a graph with metadata refresh and atomic replacement."""
    validate_graph(graph)
    path = Path(graph_file)
    ensure_graph_dir(path)
    graph["metadata"]["updated_at"] = current_timestamp()
    graph["metadata"]["node_count"] = len(graph["nodes"])
    graph["metadata"]["edge_count"] = len(graph["edges"])

    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(graph, file, indent=2, ensure_ascii=False)
            file.write("\n")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def next_node_id(graph: dict[str, Any]) -> str:
    """Return the next `paper_XXXX` ID."""
    nums = []
    for node in graph.get("nodes", []):
        node_id = str(node.get("id", ""))
        if node_id.startswith("paper_") and node_id.removeprefix("paper_").isdigit():
            nums.append(int(node_id.removeprefix("paper_")))
    return f"paper_{max(nums, default=-1) + 1:04d}"


def next_edge_id(graph: dict[str, Any]) -> str:
    """Return the next `edge_XXXX` ID."""
    nums = []
    for edge in graph.get("edges", []):
        edge_id = str(edge.get("id", ""))
        if edge_id.startswith("edge_") and edge_id.removeprefix("edge_").isdigit():
            nums.append(int(edge_id.removeprefix("edge_")))
    return f"edge_{max(nums, default=-1) + 1:04d}"


def normalize_title(title: str | None) -> str:
    """Normalize a paper title for de-duplication and fuzzy reference matching."""
    if not title:
        return ""
    normalized = title.casefold()
    normalized = re.sub(r"[\W_]+", " ", normalized, flags=re.UNICODE)
    normalized = re.sub(r"\b(arxiv|preprint|proceedings|conference|journal)\b", " ", normalized)
    return " ".join(normalized.split())


def find_node_by_title(graph: dict[str, Any], title: str) -> dict[str, Any] | None:
    """Find a node by normalized title."""
    wanted = normalize_title(title)
    for node in graph.get("nodes", []):
        if normalize_title(node.get("title")) == wanted:
            return node
    return None


def create_node(
    graph: dict[str, Any],
    *,
    title: str,
    authors: list[str] | None = None,
    year: int | None = None,
    venue: str | None = None,
    abstract: str | None = None,
    methods: list[dict[str, Any]] | None = None,
    pdf_path: str | None = None,
    source: str = "user",
) -> dict[str, Any]:
    """Create a node dictionary without mutating the graph."""
    return {
        "id": next_node_id(graph),
        "title": title,
        "authors": authors or [],
        "year": year,
        "venue": venue,
        "abstract": abstract,
        "methods": methods or [],
        "pdf_path": pdf_path,
        "source": source,
        "added_at": current_timestamp(),
    }


def create_edge(
    graph: dict[str, Any],
    *,
    source: str,
    target: str,
    reasoning_pattern: str,
    bottleneck: str,
    mechanism: str,
    evidence: str,
    confidence: float,
) -> dict[str, Any]:
    """Create an edge dictionary without mutating the graph."""
    if reasoning_pattern not in REASONING_PATTERNS:
        raise ValueError(f"Unknown reasoning pattern: {reasoning_pattern}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")
    return {
        "id": next_edge_id(graph),
        "source": source,
        "target": target,
        "reasoning_pattern": reasoning_pattern,
        "bottleneck": bottleneck,
        "mechanism": mechanism,
        "evidence": evidence,
        "confidence": confidence,
    }


def validate_graph(graph: dict[str, Any]) -> None:
    """Raise ValueError if a graph is missing required top-level fields."""
    for key in ("metadata", "schema", "nodes", "edges"):
        if key not in graph:
            raise ValueError(f"Graph is missing required field: {key}")
    if not isinstance(graph["nodes"], list):
        raise ValueError("Graph field `nodes` must be a list")
    if not isinstance(graph["edges"], list):
        raise ValueError("Graph field `edges` must be a list")
    patterns = graph.get("schema", {}).get("reasoning_patterns", [])
    missing = [pattern for pattern in REASONING_PATTERNS if pattern not in patterns]
    if missing:
        raise ValueError(f"Graph schema is missing reasoning patterns: {missing}")

