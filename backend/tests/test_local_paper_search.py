"""Local paper search metadata correction behavior."""

from __future__ import annotations

from cg.settings import Settings
from cg.tools.local_paper_search import LocalPaperIndex, canonical_paper_title


def _index() -> LocalPaperIndex:
    index = LocalPaperIndex.__new__(LocalPaperIndex)
    index.settings = Settings(scholar_source_gate_enabled=True)
    return index


def test_canonical_title_recovers_iclr_numbered_metadata() -> None:
    paper = {
        "title": "000",
        "abstract": (
            "Graph retrieval-augmented generation uses knowledge graphs to improve "
            "retrieval-augmented generation."
        ),
        "focused_text": (
            "000\n001\nUnder review as a conference paper at ICLR 2026\n"
            "WHEN TO USE GRAPHS IN RAG: A COMPREHENSIVE ANALYSIS FOR GRAPH "
            "RETRIEVAL-AUGMENTED GEN\nERATION\n"
            "Anonymous authors\nPaper under double-blind review\nABSTRACT\n"
        ),
        "pdf_path": (
            "/papers/15574_When_to_use_Graphs_in_RAG_A_Comprehensive_Analysis_for_"
            "Graph_Retrieval-Augmented_Generation.pdf"
        ),
    }

    assert canonical_paper_title(paper) == (
        "WHEN TO USE GRAPHS IN RAG: A COMPREHENSIVE ANALYSIS FOR GRAPH "
        "RETRIEVAL-AUGMENTED GENERATION"
    )
    assert _index()._topic_rejection_reason("RAG with Knowledge Graphs", paper) == ""


def test_canonical_title_recovers_acl_proceedings_metadata() -> None:
    paper = {
        "title": "Findings of the Association for Computational Linguistics: ACL 2025, pages 16652-16670",
        "abstract": (
            "This work studies graph neural retrieval for retrieval-augmented generation "
            "over knowledge graphs."
        ),
        "focused_text": (
            "Findings of the Association for Computational Linguistics: ACL 2025, pages 16652-16670\n"
            "July 27 - August 1, 2025 ©2025 Association for Computational Linguistics\n"
            "GNN-RAG: Graph Neural Retrieval for Efficient Large Language Model\n"
            "Reasoning on Knowledge Graphs\n"
            "Jiangxu Wu1, Cong Wang1, Jun Yang1\nOPPO AI Center\nAbstract\n"
        ),
        "pdf_path": "/papers/2025.findings-acl.856.pdf",
    }

    assert canonical_paper_title(paper) == (
        "GNN-RAG: Graph Neural Retrieval for Efficient Large Language Model "
        "Reasoning on Knowledge Graphs"
    )
    assert _index()._topic_rejection_reason("RAG with Knowledge Graphs", paper) == ""


def test_canonical_title_extends_truncated_title() -> None:
    paper = {
        "title": "REMINDRAG: Low-Cost LLM-Guided Knowledge",
        "abstract": (
            "Knowledge graphs offer a promising avenue for Retrieval Augmented Generation "
            "systems and KG-RAG graph traversal."
        ),
        "focused_text": (
            "REMINDRAG: Low-Cost LLM-Guided Knowledge\n"
            "Graph Traversal for Efficient RAG\n"
            "Yikuan Hu1♢\nJifeng Zhu1♢\nAbstract\n"
        ),
        "pdf_path": "/papers/4043_ReMindRAG_Low-Cost_LLM-Guided_Knowledge_Graph_Traversal_for_Efficient_RAG.pdf",
    }

    assert canonical_paper_title(paper) == (
        "REMINDRAG: Low-Cost LLM-Guided Knowledge Graph Traversal for Efficient RAG"
    )
    assert _index()._topic_rejection_reason("RAG with Knowledge Graphs", paper) == ""


def test_rag_kg_gate_still_rejects_kg_construction_without_title_rag_signal() -> None:
    paper = {
        "title": "KGGen: Extracting Knowledge Graphs from Plain",
        "abstract": (
            "KGGen extracts high-quality graphs from plain text. It is compared with "
            "GraphRAG, but the method is a text-to-knowledge-graph generator."
        ),
        "focused_text": (
            "KGGen: Extracting Knowledge Graphs from Plain\n"
            "Text with Language Models\n"
            "Belinda Mo∗1, Kyssen Yu∗2\nAbstract\n"
        ),
        "pdf_path": "/papers/25968_KGGen_Extracting_Knowledge_Graphs_from_Plain_Text_with_Language_Models.pdf",
    }

    assert canonical_paper_title(paper) == (
        "KGGen: Extracting Knowledge Graphs from Plain Text with Language Models"
    )
    assert (
        _index()._source_subtype_for_topic("RAG with Knowledge Graphs", paper).label
        == "kg_construction"
    )
    assert (
        _index()._topic_rejection_reason("RAG with Knowledge Graphs", paper)
        == "RAG+KG query requires title-level RAG and graph signal"
    )


def test_rag_kg_target_gate_rejects_query_drift_graph_sampling() -> None:
    paper = {
        "title": "Efficient Streaming Algorithms for Graphlet Sampling",
        "abstract": (
            "This paper studies streaming algorithms for estimating graphlets in "
            "large graphs with approximation guarantees."
        ),
        "focused_text": "",
        "pdf_path": "/papers/graphlet_sampling.pdf",
    }

    reason = _index()._topic_rejection_reason(
        "subgraph sampling and approximation algorithms for efficient KG retrieval",
        paper,
        target_topic="RAG with Knowledge Graphs",
    )

    assert reason == "RAG+KG query requires title-level RAG and graph signal"


def test_rag_kg_target_gate_rejects_generic_kg_completion() -> None:
    paper = {
        "title": "Replacing Paths with Connection-Biased Attention for Knowledge Graph Completion",
        "abstract": (
            "Knowledge graph completion predicts missing relations in sparse "
            "knowledge graphs using path-aware attention."
        ),
        "focused_text": "",
        "pdf_path": "/papers/kg_completion.pdf",
    }

    reason = _index()._topic_rejection_reason(
        "relation-aware anchor retrieval over sparse knowledge graphs",
        paper,
        target_topic="RAG with Knowledge Graphs",
    )

    assert reason == "RAG+KG query requires title-level RAG and graph signal"


def test_rag_kg_target_gate_rejects_molecule_gnn() -> None:
    paper = {
        "title": "Pre-Training Graph Neural Networks on Molecules with Context Prediction",
        "abstract": (
            "We pre-train graph neural networks on molecular graphs for downstream "
            "property prediction tasks."
        ),
        "focused_text": "",
        "pdf_path": "/papers/molecule_gnn.pdf",
    }

    reason = _index()._topic_rejection_reason(
        "graph neural retrieval and subgraph representation learning",
        paper,
        target_topic="RAG with Knowledge Graphs",
    )

    assert reason == "RAG+KG query requires title-level RAG and graph signal"


def test_rag_kg_target_gate_accepts_graphrag_paper_under_drift_query() -> None:
    paper = {
        "title": "HyperGraphRAG: Retrieval-Augmented Generation with Hypergraph-Structured Knowledge",
        "abstract": (
            "HyperGraphRAG augments large language models with knowledge graphs "
            "and graph retrieval-augmented generation over hypergraph structure."
        ),
        "focused_text": "",
        "pdf_path": "/papers/hypergraphrag.pdf",
    }

    assert (
        _index()._topic_rejection_reason(
            "subgraph sampling and approximation algorithms for efficient KG retrieval",
            paper,
            target_topic="RAG with Knowledge Graphs",
        )
        == ""
    )


def test_rag_kg_target_gate_accepts_frag_paper_under_drift_query() -> None:
    paper = {
        "title": "FRAG: A Flexible Modular Framework for Retrieval-Augmented Generation based on Knowledge Graphs",
        "abstract": (
            "Knowledge Graph based Retrieval-Augmented Generation uses KGs as "
            "external resources to enhance LLM reasoning."
        ),
        "focused_text": "",
        "pdf_path": "/papers/frag.pdf",
    }

    assert (
        _index()._topic_rejection_reason(
            "modular pipeline composition and inference-time control",
            paper,
            target_topic="RAG with Knowledge Graphs",
        )
        == ""
    )


def test_rag_kg_subtype_rejects_retrieval_augmented_kg_construction() -> None:
    paper = {
        "title": "mRAKL: Multilingual Retrieval-Augmented Knowledge Graph Construction for Low-Resourced Languages",
        "abstract": (
            "Multilingual Knowledge Graph Construction (mKGC) is the task of "
            "constructing or predicting missing entities and links for knowledge graphs. "
            "We reformulate mKGC with retrieval-augmented techniques."
        ),
        "focused_text": "",
        "pdf_path": "/papers/mrakl.pdf",
    }

    subtype = _index()._source_subtype_for_topic(
        "RAG with Knowledge Graphs formal experimental evaluation",
        paper,
        target_topic="RAG with Knowledge Graphs",
    )

    assert subtype.label == "kg_construction"
    assert subtype.rejection_reason == (
        "RAG+KG query excludes pure knowledge-graph construction/extraction papers"
    )
    assert _index()._topic_rejection_reason(
        "RAG with Knowledge Graphs formal experimental evaluation",
        paper,
        target_topic="RAG with Knowledge Graphs",
    ) == subtype.rejection_reason


def test_rag_kg_subtype_keeps_graphrag_with_local_kg_construction() -> None:
    paper = {
        "title": "Query-Driven Multimodal GraphRAG: Dynamic Local Knowledge Graph Construction for Online Reasoning",
        "abstract": (
            "Recent advances include retrieval-augmented generation and "
            "knowledge graph-enhanced RAG. We propose a GraphRAG framework that "
            "constructs a local knowledge graph for online reasoning."
        ),
        "focused_text": "",
        "pdf_path": "/papers/query_driven_multimodal_graphrag.pdf",
    }

    subtype = _index()._source_subtype_for_topic(
        "RAG with Knowledge Graphs multimodal integration",
        paper,
        target_topic="RAG with Knowledge Graphs",
    )

    assert subtype.label == "core_kg_rag_method"
    assert subtype.rejection_reason == ""
    assert _index()._topic_rejection_reason(
        "RAG with Knowledge Graphs multimodal integration",
        paper,
        target_topic="RAG with Knowledge Graphs",
    ) == ""


def test_rag_kg_subtype_marks_application_without_rejecting() -> None:
    paper = {
        "title": "Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation",
        "abstract": (
            "The recommender system uses knowledge graph retrieval-augmented "
            "generation to improve LLM-based recommendation quality."
        ),
        "focused_text": "",
        "pdf_path": "/papers/kg_rag_recommendation.pdf",
    }

    subtype = _index()._source_subtype_for_topic(
        "RAG with Knowledge Graphs limitations",
        paper,
        target_topic="RAG with Knowledge Graphs",
    )

    assert subtype.label == "application_case"
    assert subtype.rejection_reason == ""
    assert _index()._topic_rejection_reason(
        "RAG with Knowledge Graphs limitations",
        paper,
        target_topic="RAG with Knowledge Graphs",
    ) == ""


def test_rag_kg_subtype_separates_kgqa_graph_reasoning() -> None:
    paper = {
        "title": "Fast Think-on-Graph: Wider, Deeper and Faster Reasoning of Large Language Model on Knowledge Graph",
        "abstract": (
            "Graph Retrieval Augmented Generation can integrate knowledge graphs, "
            "but this paper focuses on question answering and reasoning on knowledge graph paths."
        ),
        "focused_text": "",
        "pdf_path": "/papers/fast_tog.pdf",
    }

    subtype = _index()._source_subtype_for_topic(
        "RAG with Knowledge Graphs hallucination mitigation",
        paper,
        target_topic="RAG with Knowledge Graphs",
    )

    assert subtype.label == "kgqa_or_graph_reasoning"
    assert subtype.rejection_reason == ""


def test_causal_reasoning_llm_gate_rejects_autoregressive_causal_inference() -> None:
    paper = {
        "title": "KV-Runahead: Scalable Causal LLM Inference by Parallel Key-Value Cache Generation",
        "abstract": (
            "The paper accelerates autoregressive causal language model inference "
            "with cache generation for efficient decoding."
        ),
        "focused_text": "",
        "pdf_path": "/papers/kv_runahead.pdf",
    }

    assert _index()._topic_rejection_reason(
        "Causal Reasoning with LLMs inference-time control",
        paper,
    ) == "Causal reasoning with LLMs query requires causal/counterfactual reasoning and LLM signal"


def test_causal_reasoning_llm_gate_accepts_core_benchmark() -> None:
    paper = {
        "title": "Ice Cream Doesn't Cause Drowning: Benchmarking LLMs Against Statistical Pitfalls in Causal Inference",
        "abstract": (
            "We evaluate large language models on causal reasoning benchmarks "
            "and statistical pitfalls in causal inference."
        ),
        "focused_text": "",
        "pdf_path": "/papers/ice_cream_causal_llm.pdf",
    }

    subtype = _index()._source_subtype_for_topic(
        "Causal Reasoning with LLMs benchmark",
        paper,
    )

    assert subtype.label == "causal_reasoning_benchmark"
    assert _index()._topic_rejection_reason("Causal Reasoning with LLMs benchmark", paper) == ""


def test_causal_reasoning_llm_gate_rejects_causal_rag_drift() -> None:
    paper = {
        "title": "CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation",
        "abstract": (
            "Large language models use retrieval-augmented generation to integrate "
            "external knowledge. This work proposes a RAG framework that incorporates "
            "causal graphs into retrieval and evaluates answer faithfulness, context "
            "recall, and context precision."
        ),
        "focused_text": "",
        "pdf_path": "/papers/causalrag.pdf",
    }

    subtype = _index()._source_subtype_for_topic(
        "Causal Reasoning with LLMs representation shift",
        paper,
    )

    assert subtype.label == "rag_or_retrieval_adjacent"
    assert subtype.rejection_reason == (
        "Causal reasoning with LLMs query excludes RAG/retrieval/QA papers unless they evaluate LLM causal reasoning"
    )
    assert _index()._topic_rejection_reason(
        "Causal Reasoning with LLMs representation shift",
        paper,
    ) == subtype.rejection_reason


def test_causal_reasoning_target_topic_overrides_counterfactual_inference_terms() -> None:
    paper = {
        "title": "Unveiling Causal Reasoning in Large Language Models: Reality or Mirage?",
        "abstract": (
            "This work evaluates whether large language models have genuine causal "
            "reasoning capability and includes counterfactual reasoning tasks."
        ),
        "focused_text": "",
        "pdf_path": "/papers/reality_or_mirage.pdf",
    }

    subtype = _index()._source_subtype_for_topic(
        "formal evaluation of large language models causal interventions counterfactuals confounders do operator",
        paper,
        target_topic="Causal Reasoning with LLMs",
    )

    assert subtype.label == "causal_reasoning_benchmark"
    assert _index()._topic_rejection_reason(
        "formal evaluation of large language models causal interventions counterfactuals confounders do operator",
        paper,
        target_topic="Causal Reasoning with LLMs",
    ) == ""


def test_counterfactual_inference_gate_rejects_generic_bayes() -> None:
    paper = {
        "title": "Large Language Bayes",
        "abstract": (
            "This paper studies Bayesian posterior inference for language models "
            "and probabilistic program induction."
        ),
        "focused_text": "",
        "pdf_path": "/papers/large_language_bayes.pdf",
    }

    assert _index()._topic_rejection_reason(
        "Counterfactual Inference probabilistic modeling",
        paper,
    ) == "Counterfactual inference query requires counterfactual/treatment-effect/causal-inference signal"


def test_counterfactual_inference_gate_classifies_treatment_effect() -> None:
    paper = {
        "title": "Causal Contrastive Learning for Counterfactual Regression Over Time",
        "abstract": (
            "The method estimates heterogeneous treatment effects and potential "
            "outcomes for counterfactual regression."
        ),
        "focused_text": "",
        "pdf_path": "/papers/counterfactual_regression.pdf",
    }

    subtype = _index()._source_subtype_for_topic(
        "Counterfactual Inference representation learning",
        paper,
    )

    assert subtype.label == "treatment_effect_estimation"
    assert _index()._topic_rejection_reason("Counterfactual Inference representation learning", paper) == ""


def test_counterfactual_inference_gate_handles_hyphenated_counterfactual() -> None:
    paper = {
        "title": "Targeted Estimation of Potential Outcomes",
        "abstract": (
            "The method estimates counter-\n"
            "factual outcomes and heterogeneous treatment effects from observational data."
        ),
        "focused_text": "",
        "pdf_path": "/papers/targeted_counterfactual_estimation.pdf",
    }

    subtype = _index()._source_subtype_for_topic(
        "Counterfactual Inference potential outcomes",
        paper,
    )

    assert subtype.label == "treatment_effect_estimation"
    assert _index()._topic_rejection_reason("Counterfactual Inference potential outcomes", paper) == ""


def test_counterfactual_inference_gate_rejects_recommender_drift() -> None:
    paper = {
        "title": "Counterfactual Implicit Feedback Modeling",
        "abstract": (
            "This recommender system paper models implicit feedback and popularity "
            "bias for recommendation debiasing."
        ),
        "focused_text": "",
        "pdf_path": "/papers/counterfactual_implicit_feedback.pdf",
    }

    subtype = _index()._source_subtype_for_topic(
        "Counterfactual Inference cross-domain synthesis",
        paper,
    )

    assert subtype.label == "application_or_recommender_adjacent"
    assert subtype.rejection_reason == (
        "Counterfactual inference query excludes recommender/debiasing application papers"
    )
    assert _index()._topic_rejection_reason(
        "Counterfactual Inference cross-domain synthesis",
        paper,
    ) == subtype.rejection_reason


def test_counterfactual_inference_gate_rejects_off_policy_eval_drift() -> None:
    paper = {
        "title": "Off-policy Estimation with Adaptively Collected Data",
        "abstract": (
            "The work studies off-policy evaluation and policy mean embeddings "
            "for online learning."
        ),
        "focused_text": "",
        "pdf_path": "/papers/off_policy_estimation.pdf",
    }

    subtype = _index()._source_subtype_for_topic(
        "Counterfactual Inference policy evaluation",
        paper,
    )

    assert subtype.label == "policy_evaluation_adjacent"
    assert subtype.rejection_reason == (
        "Counterfactual inference query excludes off-policy evaluation papers"
    )


def test_counterfactual_inference_gate_rejects_causal_discovery_only() -> None:
    paper = {
        "title": "A Meta-Learning Approach to Bayesian Causal Discovery",
        "abstract": (
            "The paper estimates causal structure and interventional distributions "
            "for graph structure learning under interventions."
        ),
        "focused_text": "",
        "pdf_path": "/papers/bayesian_causal_discovery.pdf",
    }

    subtype = _index()._source_subtype_for_topic(
        "Counterfactual Inference probabilistic modeling",
        paper,
    )

    assert subtype.label == "causal_discovery_adjacent"
    assert subtype.rejection_reason == (
        "Counterfactual inference query excludes causal-discovery-only papers"
    )


def test_multi_hop_graph_gate_rejects_generic_gnn_representation() -> None:
    paper = {
        "title": "GPEN: Global Position Encoding Network for Enhanced Subgraph Representation Learning",
        "abstract": (
            "The paper improves graph neural network representation learning for "
            "subgraphs through global position encodings."
        ),
        "focused_text": "",
        "pdf_path": "/papers/gpen.pdf",
    }

    assert _index()._topic_rejection_reason(
        "Multi-hop Reasoning on Graphs representation learning",
        paper,
    ) == "Multi-hop graph reasoning query excludes generic graph representation learning"


def test_multi_hop_graph_gate_classifies_kgqa_benchmark() -> None:
    paper = {
        "title": "M3GQA: A Multi-Entity Multi-Hop Multi-Setting Graph Question Answering Benchmark",
        "abstract": (
            "This benchmark evaluates multi-hop graph question answering over "
            "knowledge graphs with complex reasoning paths."
        ),
        "focused_text": "",
        "pdf_path": "/papers/m3gqa.pdf",
    }

    subtype = _index()._source_subtype_for_topic(
        "Multi-hop Reasoning on Graphs benchmark dataset",
        paper,
    )

    assert subtype.label == "graph_reasoning_benchmark"
    assert _index()._topic_rejection_reason("Multi-hop Reasoning on Graphs benchmark dataset", paper) == ""


def test_multi_hop_graph_gate_keeps_kgqa_despite_retrieval_context() -> None:
    paper = {
        "title": "GNN-RAG: Graph Neural Retrieval for Efficient Large Language Model Reasoning on Knowledge Graphs",
        "abstract": (
            "Retrieval-augmented generation in Knowledge Graph Question Answering "
            "enhances LLM reasoning on knowledge graphs by retrieving executable "
            "relation paths for multi-hop question answering."
        ),
        "focused_text": "",
        "pdf_path": "/papers/gnn_rag.pdf",
    }

    subtype = _index()._source_subtype_for_topic(
        "GNN-RAG graph neural retrieval efficient LLM reasoning on knowledge graphs",
        paper,
        target_topic="Multi-hop Reasoning on Graphs",
    )

    assert subtype.label == "kgqa_or_graph_reasoning"


def test_multi_hop_target_topic_overrides_rag_kg_classifier() -> None:
    paper = {
        "title": "GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation",
        "abstract": (
            "The method applies retrieval augmented generation on knowledge graphs "
            "but does not focus on multi-hop graph reasoning tasks."
        ),
        "focused_text": "",
        "pdf_path": "/papers/gfm_rag.pdf",
    }

    subtype = _index()._source_subtype_for_topic(
        "retrieval augmented generation knowledge graph traversal efficient RAG multi-hop question answering",
        paper,
        target_topic="Multi-hop Reasoning on Graphs",
    )

    assert subtype.label == "graph_retrieval_rag_adjacent"
    assert _index()._topic_rejection_reason(
        "retrieval augmented generation knowledge graph traversal efficient RAG multi-hop question answering",
        paper,
        target_topic="Multi-hop Reasoning on Graphs",
    ) == ""
