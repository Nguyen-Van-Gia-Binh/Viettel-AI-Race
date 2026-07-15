"""Workstream A (Week 2): hybrid retrieval system for entity linking.
Combines BM25 (sparse retrieval) and dense retrieval (using SentenceTransformers)
with Reciprocal Rank Fusion (RRF) to merge candidate lists.
Falls back gracefully to the rapidfuzz-based matcher if dependencies are missing.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

# Try importing the optional dependencies for dense + sparse retrieval
try:
    import numpy as np
    import torch
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    HAS_RETRIEVAL_LIBS = True
except ImportError:
    HAS_RETRIEVAL_LIBS = False

logger = logging.getLogger(__name__)

class RetrievalSystem:
    def __init__(self, choices: dict[str, str], model_name: str = "intfloat/multilingual-e5-small"):
        self.choices = choices  # key (code/rxcui) -> text (description + aliases)
        self.keys = list(choices.keys())
        self.texts = list(choices.values())
        self.model_name = model_name
        self.bm25 = None
        self.dense_model = None
        self.dense_embeddings = None
        
        if not HAS_RETRIEVAL_LIBS:
            logger.warning("Retrieval libraries missing. Running in fallback mode.")
            return

        try:
            # Initialize BM25
            tokenized_corpus = [text.lower().split() for text in self.texts]
            self.bm25 = BM25Okapi(tokenized_corpus)

            # Initialize Dense (SentenceTransformer)
            # Use CPU by default since local dev has no GPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.dense_model = SentenceTransformer(model_name, device=device)
            
            # E5 models require "query: " or "passage: " prefixes for retrieval
            # We treat the KB texts as passages
            passage_texts = [f"passage: {t}" for t in self.texts]
            self.dense_embeddings = self.dense_model.encode(
                passage_texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(self.dense_embeddings, axis=1, keepdims=True)
            self.dense_embeddings = self.dense_embeddings / np.maximum(norms, 1e-12)
        except Exception as e:
            logger.error("Failed to initialize advanced retrieval system: %s. Falling back.", e)
            self.bm25 = None

    def retrieve_bm25(self, query: str, top_k: int = 50) -> list[str]:
        if not self.bm25:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.keys[idx] for idx in top_indices if scores[idx] > 0]

    def retrieve_dense(self, query: str, top_k: int = 50) -> list[str]:
        if self.dense_model is None or self.dense_embeddings is None:
            return []
        # Query prefix for E5
        query_text = f"query: {query}"
        query_embedding = self.dense_model.encode(query_text, convert_to_numpy=True)
        norm = np.linalg.norm(query_embedding)
        if norm > 1e-12:
            query_embedding = query_embedding / norm
        
        # Calculate cosine similarity
        scores = np.dot(self.dense_embeddings, query_embedding)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.keys[idx] for idx in top_indices]


def reciprocal_rank_fusion(
    lists: list[list[str]],
    k: int = 60,
    top_k: int = 5
) -> list[str]:
    """Merge multiple candidate lists using Reciprocal Rank Fusion (RRF)."""
    rrf_scores: dict[str, float] = {}
    for lst in lists:
        for rank, item in enumerate(lst):
            # rank is 0-indexed, so rank + 1 is 1-based rank
            rrf_scores[item] = rrf_scores.get(item, 0.0) + 1.0 / (k + (rank + 1))
            
    # Sort keys by score descending
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, score in sorted_items[:top_k]]


def hybrid_link(
    entity_text: str,
    choices: dict[str, str],
    top_k: int = 5,
    model_name: str = "intfloat/multilingual-e5-small"
) -> list[str]:
    """Retrieve candidates using BM25 and Dense retrieval, then merge via RRF.
    If libraries are missing, falls back to empty lists (which lets the caller
    degrade to rapidfuzz).
    """
    if not HAS_RETRIEVAL_LIBS or not entity_text or not choices:
        return []
        
    try:
        system = RetrievalSystem(choices, model_name=model_name)
        if not system.bm25:
            return []
            
        bm25_candidates = system.retrieve_bm25(entity_text, top_k=50)
        dense_candidates = system.retrieve_dense(entity_text, top_k=50)
        
        return reciprocal_rank_fusion([bm25_candidates, dense_candidates], top_k=top_k)
    except Exception as e:
        logger.error("Error during hybrid linking: %s", e)
        return []
