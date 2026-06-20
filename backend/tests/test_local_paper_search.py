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
        _index()._topic_rejection_reason("RAG with Knowledge Graphs", paper)
        == "RAG+KG query requires title-level RAG and graph signal"
    )
