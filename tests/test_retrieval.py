"""Unit tests for Workstream A (Week 2): hybrid retrieval and fallback mechanisms.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from clinlp.schema import ConceptType
from clinlp.linking.retrieval import reciprocal_rank_fusion, RetrievalSystem, hybrid_link


def test_rrf_merging():
    """Verify that reciprocal_rank_fusion merges ranks correctly."""
    # Candidate lists (ordered best to worst)
    sparse_candidates = ["A", "B", "C"]
    dense_candidates = ["B", "D", "A"]
    
    merged = reciprocal_rank_fusion([sparse_candidates, dense_candidates], k=60, top_k=2)
    
    # A has ranks: 1 (sparse), 3 (dense) -> score: 1/(60+1) + 1/(60+3) = 1/61 + 1/63 = 0.03226
    # B has ranks: 2 (sparse), 1 (dense) -> score: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.03252
    # Since B has higher score, B should be #1, A should be #2
    assert merged[0] == "B"
    assert merged[1] == "A"


def test_retrieval_system_initialization_fallback():
    """Test RetrievalSystem basic initialization fallback behavior when libs are missing."""
    choices = {"C1": "đau đầu kinh niên"}
    
    with patch("clinlp.linking.retrieval.HAS_RETRIEVAL_LIBS", False):
        system = RetrievalSystem(choices)
        assert system.choices == choices
        assert system.bm25 is None
        assert system.dense_model is None


@patch("clinlp.linking.retrieval.HAS_RETRIEVAL_LIBS", False)
def test_hybrid_link_fallback_when_dependencies_missing():
    """Verify hybrid_link returns empty list (leading to rapidfuzz fallback) when libraries are missing."""
    choices = {"K21.9": "trào ngược dạ dày"}
    candidates = hybrid_link("trào ngược", choices, top_k=5)
    assert candidates == []


@patch("clinlp.linking.retrieval.HAS_RETRIEVAL_LIBS", True)
def test_hybrid_link_retrieval_flow():
    """Verify hybrid_link successfully runs RetrievalSystem retrieval when libraries are mock-present."""
    choices = {"C1": "đau đầu", "C2": "ho", "C3": "sốt"}
    
    mock_system = MagicMock()
    mock_system.bm25 = MagicMock()
    mock_system.retrieve_bm25.return_value = ["C1", "C2"]
    mock_system.retrieve_dense.return_value = ["C2", "C3"]
    
    with patch("clinlp.linking.retrieval.RetrievalSystem", return_value=mock_system):
        candidates = hybrid_link("ho kéo dài", choices, top_k=3)
        # Merged list should prioritize C2 (appears in both at high rank)
        assert "C2" in candidates
        assert len(candidates) <= 3
