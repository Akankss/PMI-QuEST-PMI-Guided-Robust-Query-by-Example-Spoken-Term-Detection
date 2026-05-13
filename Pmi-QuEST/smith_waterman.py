"""
Smith-Waterman local sequence alignment for QbE-STD re-ranking.
"""

import numpy as np
from typing import List, Tuple


def smith_waterman_score(query: List[int],
                         candidate: List[int],
                         match_score: float = 2.0,
                         mismatch_penalty: float = -1.0,
                         gap_penalty: float = -1.0) -> float:
    """
    
    Args:
        query: Query token sequence.
        candidate: Candidate token sequence.
        match_score: Score added for matching tokens.
        mismatch_penalty: Score added for mismatched tokens (negative).
        gap_penalty: Score added for inserting a gap (negative).

    Returns:
        Maximum alignment score (float). Higher = more similar.
    """
    m, n = len(query), len(candidate)
    if m == 0 or n == 0:
        return 0.0

    # Build scoring matrix H of size (m+1) x (n+1)
    H = np.zeros((m + 1, n + 1), dtype=np.float32)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Diagonal: match or mismatch
            diag = H[i - 1, j - 1] + (
                match_score if query[i - 1] == candidate[j - 1]
                else mismatch_penalty
            )
            # Up: gap in candidate
            up = H[i - 1, j] + gap_penalty
            # Left: gap in query
            left = H[i, j - 1] + gap_penalty

            H[i, j] = max(0.0, diag, up, left)

    return float(H.max())


def smith_waterman_score_normalized(query: List[int],
                                    candidate: List[int],
                                    match_score: float = 2.0,
                                    mismatch_penalty: float = -1.0,
                                    gap_penalty: float = -1.0) -> float:
    """
    Normalized Smith-Waterman score in [0, 1].

    Normalizes by the maximum possible score (perfect match of query),
    so scores are comparable across queries of different lengths.

    Args:
        query: Query token sequence.
        candidate: Candidate token sequence.
        match_score, mismatch_penalty, gap_penalty: SW parameters.

    Returns:
        Score in [0, 1].
    """
    raw = smith_waterman_score(query, candidate, match_score,
                               mismatch_penalty, gap_penalty)
    max_possible = match_score * len(query)
    if max_possible == 0:
        return 0.0
    return raw / max_possible


def rerank(query_seq: List[int],
           candidates: List[Tuple[float, int]],
           corpus_sequences: List[List[int]],
           match_score: float = 2.0,
           mismatch_penalty: float = -1.0,
           gap_penalty: float = -1.0) -> List[Tuple[float, int]]:
    """
    Re-rank HNSW candidates using normalized Smith-Waterman scores.

    Replaces the HNSW cosine distance ranking with a more precise
    local alignment score. This is Stage 3 of BPE-MNG H-QuEST.

    Args:
        query_seq: Query sequence.
        candidates: List of (cosine_distance, corpus_idx) from HNSW.
        corpus_sequences: Corpus sequences.
        match_score, mismatch_penalty, gap_penalty: SW parameters.

    Returns:
        Re-ranked list of (sw_score, corpus_idx), sorted by score descending.
    """
    reranked = []
    for _, corpus_idx in candidates:
        candidate_seq = corpus_sequences[corpus_idx]
        score = smith_waterman_score_normalized(
            query_seq, candidate_seq,
            match_score, mismatch_penalty, gap_penalty
        )
        reranked.append((score, corpus_idx))

    # Sort by SW score descending (higher = better)
    reranked.sort(key=lambda x: -x[0])
    return reranked
