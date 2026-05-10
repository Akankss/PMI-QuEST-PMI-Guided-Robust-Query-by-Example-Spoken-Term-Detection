"""
pmiquest_system.py
==================
Self-contained implementation of three QbE-STD systems for TASLP comparison,
corrected to match the paper exactly.

Systems
-------
1. TF-IDF Baseline  (Singh et al. 2024)
       Raw tokens → unigram TF-IDF → brute-force cosine ranking

2. H-QuEST  (Singh et al., Interspeech 2025)
       Raw tokens → unigram TF-IDF
                 → HNSW (M=16, ef_construction=150, ef_search=200, C=200)
                 → Smith-Waterman rerank (match=+2, mismatch=-1, gap=-2)
                 → score normalised by query length

3. PMI-QuEST  (proposed, this paper)
       Raw tokens → PMI-filtered bigram TF-IDF (τ=0.5, α=1.0)
                 → HNSW (same hyperparams)
                 → Smith-Waterman rerank (same, score/query_length)

Fixes applied vs the previous version
--------------------------------------
[1] HNSWIndex now uses hnswlib (real HNSW graph) with M=16,
    ef_construction=150, ef_search=200.  Falls back to sklearn
    NearestNeighbors with a loud warning if hnswlib is not installed.
[2] smith_waterman() returns max(H) / len(query)  (paper Eq. 4).
[3] Default candidate count changed to C=200 everywhere.
[4] Default bigram weight α=1.0 (paper §3.4).
[5] PMITokenDedup, regime-gated SW, and PMI-soft SW removed —
    none of these appear in the paper.

Evaluation
----------
MAP, MRR, P@1, P@5, P@10 — computed over the full ranked corpus.

Usage
-----
    from pmiquest_system import TFIDFBaseline, HQuEST, PMIQuest
    from pmiquest_system import run_comparison

    results = run_comparison(corpus_seqs, query_seqs, ground_truth)
"""
from __future__ import annotations

import math
import time
import warnings
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Set

# ── Optional hnswlib import (required for paper-accurate results) ─────────────
try:
    import hnswlib as _hnswlib
    _HNSWLIB_AVAILABLE = True
except ImportError:
    _HNSWLIB_AVAILABLE = False
    warnings.warn(
        "hnswlib not found. Install with: pip install hnswlib\n"
        "Falling back to sklearn NearestNeighbors (BallTree) — "
        "results will NOT match the paper's HNSW numbers.",
        ImportWarning,
        stacklevel=2,
    )
    from sklearn.neighbors import NearestNeighbors as _NearestNeighbors


# =============================================================================
# ── Section 1: Smith-Waterman alignment  ─────────────────────────────────────
# =============================================================================

def smith_waterman(
    query: List[int],
    doc:   List[int],
    match:    float = 2.0,
    mismatch: float = -1.0,
    gap:      float = -2.0,
) -> float:
    """
    Standard Smith-Waterman local alignment score, normalised by query length.

    Score = max_{i,j} H(i,j) / len(query)          (paper Eq. 4)

    H(i,j) = max(0,
                 H(i-1,j-1) + sigma(q_i, d_j),
                 H(i-1,j)   + g,
                 H(i,j-1)   + g)

    sigma(a,b) = match    if a == b
               = mismatch otherwise
    """
    m = len(query)
    L = len(doc)
    if m == 0 or L == 0:
        return 0.0

    S = np.zeros((m + 1, L + 1), dtype=np.float32)
    for i in range(1, m + 1):
        q_tok = query[i - 1]
        for j in range(1, L + 1):
            s_ij = match if q_tok == doc[j - 1] else mismatch
            S[i, j] = max(
                0.0,
                S[i - 1, j - 1] + s_ij,
                S[i - 1, j]     + gap,
                S[i,     j - 1] + gap,
            )

    # Normalise by query length (paper Eq. 4)
    return float(S.max()) / m


def sw_rerank(
    query:       List[int],
    candidates:  List[Tuple[float, int]],   # (cosine_dist, corpus_idx)
    corpus_seqs: List[List[int]],
    match:    float = 2.0,
    mismatch: float = -1.0,
    gap:      float = -2.0,
) -> List[Tuple[float, int]]:
    """Rerank candidates by SW score (descending)."""
    scored = [
        (smith_waterman(query, corpus_seqs[idx], match, mismatch, gap), idx)
        for _, idx in candidates
    ]
    scored.sort(key=lambda x: -x[0])
    return scored


# =============================================================================
# ── Section 2: HNSW index  ───────────────────────────────────────────────────
# =============================================================================

class HNSWIndex:
    """
    HNSW approximate nearest-neighbour index.

    Uses hnswlib when available (paper-accurate, M=16, ef_construction=150,
    ef_search=200).  Falls back to sklearn BallTree with a warning if hnswlib
    is not installed.

    Parameters
    ----------
    M               : int   HNSW graph degree (paper: 16)
    ef_construction : int   Construction beam width (paper: 150)
    ef_search       : int   Search beam width (paper: 200)
    n_neighbors     : int   Candidate set size C (paper: 200)
    """

    def __init__(
        self,
        M:               int = 16,
        ef_construction: int = 150,
        ef_search:       int = 200,
        n_neighbors:     int = 200,
    ):
        self.M               = M
        self.ef_construction = ef_construction
        self.ef_search       = ef_search
        self.n_neighbors     = n_neighbors
        self._index          = None
        self._matrix         = None   # kept for fallback cosine similarity

    def fit(self, matrix: np.ndarray) -> "HNSWIndex":
        """
        Build the index from L2-normalised document vectors.

        matrix : shape (N, D), rows must be L2-normalised.
        For L2-normalised vectors, cosine distance == 1 - inner product.
        """
        self._matrix = matrix.astype(np.float32)
        N, D = matrix.shape

        if _HNSWLIB_AVAILABLE:
            self._index = _hnswlib.Index(space="cosine", dim=D)
            self._index.init_index(
                max_elements=N,
                ef_construction=self.ef_construction,
                M=self.M,
            )
            self._index.add_items(self._matrix, list(range(N)))
            self._index.set_ef(self.ef_search)
        else:
            # Fallback: sklearn BallTree (euclidean equiv. to cosine on unit vecs)
            k = min(self.n_neighbors, N)
            self._index = _NearestNeighbors(
                n_neighbors=k,
                algorithm="ball_tree",
                metric="euclidean",
                n_jobs=-1,
            ).fit(self._matrix)

        return self

    def search(self, query_vec: np.ndarray, k: int = None) -> List[Tuple[float, int]]:
        """
        Return (cosine_distance, corpus_idx) pairs, sorted ascending by distance.
        """
        if self._index is None:
            raise RuntimeError("HNSWIndex not fitted.")
        k = k or self.n_neighbors
        k = min(k, self._matrix.shape[0])
        qv = query_vec.reshape(1, -1).astype(np.float32)

        if _HNSWLIB_AVAILABLE:
            labels, distances = self._index.knn_query(qv, k=k)
            # hnswlib cosine space: distance = 1 - cosine_similarity
            return list(zip(distances[0].tolist(), labels[0].tolist()))
        else:
            dists, idxs = self._index.kneighbors(qv, n_neighbors=k)
            # euclidean^2 / 2 ~= cosine_distance for unit vectors
            cos_dists = (dists[0] ** 2) / 2.0
            return list(zip(cos_dists.tolist(), idxs[0].tolist()))


# =============================================================================
# ── Section 3: TF-IDF vectoriser  ────────────────────────────────────────────
# =============================================================================

class UnigramTFIDF:
    """
    Unigram TF-IDF vectoriser (paper Eq. 1).

    w_uni(t, s(x)) = [n_t(x) / |s(x)|]  *  log(N / (1 + d_t))

    Vectors are L2-normalised.
    """

    def __init__(self):
        self.vocab:  Dict[int, int]   = {}
        self.idf:    Dict[int, float] = {}
        self.n_docs: int = 0

    def fit(self, sequences: List[List[int]]) -> "UnigramTFIDF":
        self.n_docs = len(sequences)
        doc_freq: Dict[int, int] = defaultdict(int)
        for seq in sequences:
            for tok in set(seq):
                doc_freq[tok] += 1
        self.vocab = {tok: i for i, tok in enumerate(sorted(doc_freq))}
        self.idf   = {
            tok: math.log(self.n_docs / (1 + df))
            for tok, df in doc_freq.items()
        }
        return self

    def transform(self, sequences: List[List[int]]) -> np.ndarray:
        V   = len(self.vocab)
        mat = np.zeros((len(sequences), V), dtype=np.float32)
        for row, seq in enumerate(sequences):
            if not seq:
                continue
            counts = Counter(seq)
            L = len(seq)
            for tok, cnt in counts.items():
                col = self.vocab.get(tok)
                if col is None:
                    continue
                mat[row, col] = (cnt / L) * self.idf.get(tok, 0.0)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def transform_one(self, seq: List[int]) -> np.ndarray:
        V   = len(self.vocab)
        vec = np.zeros(V, dtype=np.float32)
        if not seq:
            return vec
        counts = Counter(seq)
        L = len(seq)
        for tok, cnt in counts.items():
            col = self.vocab.get(tok)
            if col is None:
                continue
            vec[col] = (cnt / L) * self.idf.get(tok, 0.0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def fit_transform(self, sequences: List[List[int]]) -> np.ndarray:
        self.fit(sequences)
        return self.transform(sequences)


# =============================================================================
# ── Section 4: PMI computation  ──────────────────────────────────────────────
# =============================================================================

def compute_pmi(
    sequences:        List[List[int]],
    min_bigram_count: int = 2,
) -> Dict[Tuple[int, int], float]:
    """
    Compute PMI for all observed adjacent token pairs (paper Eq. 2).

    PMI(a, b) = log [ p(a,b) / (p(a) * p(b)) ]

    Only pairs with count >= min_bigram_count are returned.
    """
    uni: Counter = Counter()
    bi:  Counter = Counter()
    N_tok = 0
    N_bi  = 0

    for seq in sequences:
        for tok in seq:
            uni[tok] += 1
            N_tok += 1
        for i in range(len(seq) - 1):
            bi[(seq[i], seq[i + 1])] += 1
            N_bi += 1

    if N_tok == 0 or N_bi == 0:
        return {}

    pmi_scores: Dict[Tuple[int, int], float] = {}
    for (a, b), cnt in bi.items():
        if cnt < min_bigram_count:
            continue
        p_ab = cnt    / N_bi
        p_a  = uni[a] / N_tok
        p_b  = uni[b] / N_tok
        if p_a > 0 and p_b > 0:
            pmi_scores[(a, b)] = math.log(p_ab / (p_a * p_b))

    return pmi_scores


# =============================================================================
# ── Section 5: PMI-TF-IDF vectoriser  ────────────────────────────────────────
# =============================================================================

class PMIFilteredTFIDF:
    """
    Unigrams + PMI-filtered bigrams TF-IDF vectoriser (paper §3.3-3.4).

    Combined vector (Eq. 5):
        v(x) = [ u(x)  ||  alpha * b(x) ] / ||.||_2

    Unigram component (Eq. 1):
        w_uni(t, s(x)) = [n_t(x) / |s(x)|] * log(N / (1 + d_t))

    Bigram component (Eq. 3):
        w_bi((a,b), s(x)) = [n_{ab}(x) / (|s(x)|-1)] * log(N / (1 + d_{ab}))

    PMI bigram vocabulary (Eq. 4 / paper §3.3):
        B_tau = {(a,b) in V^2 : PMI(a,b) > tau}

    Parameters
    ----------
    tau_pmi       : float   PMI threshold tau (paper: 0.5)
    bigram_weight : float   alpha (paper: 1.0)
    min_count     : int     Minimum bigram corpus count
    """

    def __init__(
        self,
        tau_pmi:       float = 0.5,
        bigram_weight: float = 1.0,
        min_count:     int   = 2,
    ):
        self.tau_pmi       = tau_pmi
        self.bigram_weight = bigram_weight
        self.min_count     = min_count

        self.uni_vocab: Dict[int, int]               = {}
        self.uni_idf:   Dict[int, float]             = {}
        self.bi_vocab:  Dict[Tuple[int, int], int]   = {}
        self.bi_idf:    Dict[Tuple[int, int], float] = {}
        self.n_docs:    int  = 0
        self.is_fitted: bool = False

    def fit(self, sequences: List[List[int]]) -> "PMIFilteredTFIDF":
        self.n_docs = len(sequences)

        # ── Unigram IDF ───────────────────────────────────────────────────────
        uni_df: Dict[int, int] = defaultdict(int)
        for seq in sequences:
            for tok in set(seq):
                uni_df[tok] += 1
        self.uni_vocab = {tok: i for i, tok in enumerate(sorted(uni_df))}
        self.uni_idf   = {
            tok: math.log(self.n_docs / (1 + df))
            for tok, df in uni_df.items()
        }

        # ── PMI-filtered bigram vocabulary ────────────────────────────────────
        pmi_scores = compute_pmi(sequences, self.min_count)

        # B_tau: retain only bigrams with PMI > tau  (paper §3.3, Eq. 4)
        selected: Set[Tuple[int, int]] = {
            bg for bg, pmi in pmi_scores.items() if pmi > self.tau_pmi
        }

        # Bigram document frequency (binary: does bigram appear in doc at all)
        bi_df: Dict[Tuple[int, int], int] = defaultdict(int)
        for seq in sequences:
            seen: Set[Tuple[int, int]] = set()
            for i in range(len(seq) - 1):
                bg = (seq[i], seq[i + 1])
                if bg in selected and bg not in seen:
                    bi_df[bg] += 1
                    seen.add(bg)

        self.bi_vocab = {
            bg: i + len(self.uni_vocab)
            for i, bg in enumerate(sorted(bi_df))
        }
        self.bi_idf = {
            bg: math.log(self.n_docs / (1 + df))
            for bg, df in bi_df.items()
        }

        self.is_fitted = True
        print(
            f"  [PMI-TF-IDF] tau={self.tau_pmi:.1f}  alpha={self.bigram_weight:.2f}"
            f"  ->  {len(self.uni_vocab)} unigrams"
            f" + {len(self.bi_vocab)} bigrams"
            f" = {len(self.uni_vocab) + len(self.bi_vocab)} features"
        )
        return self

    def _vectorize(self, seq: List[int]) -> np.ndarray:
        dim = len(self.uni_vocab) + len(self.bi_vocab)
        vec = np.zeros(dim, dtype=np.float32)
        if not seq:
            return vec

        L = len(seq)

        # Unigram TF-IDF  (Eq. 1)
        for tok, cnt in Counter(seq).items():
            col = self.uni_vocab.get(tok)
            if col is not None:
                vec[col] = (cnt / L) * self.uni_idf.get(tok, 0.0)

        # Bigram TF-IDF  (Eq. 3) — denominator is |s(x)| - 1
        if self.bigram_weight > 0 and L > 1 and self.bi_vocab:
            denom = L - 1
            bi_counts: Counter = Counter()
            for i in range(L - 1):
                bg = (seq[i], seq[i + 1])
                if bg in self.bi_vocab:
                    bi_counts[bg] += 1
            for bg, cnt in bi_counts.items():
                col = self.bi_vocab.get(bg)
                if col is not None:
                    vec[col] = (
                        self.bigram_weight
                        * (cnt / denom)
                        * self.bi_idf.get(bg, 0.0)
                    )

        return vec

    def transform(self, sequences: List[List[int]]) -> np.ndarray:
        dim = len(self.uni_vocab) + len(self.bi_vocab)
        mat = np.zeros((len(sequences), dim), dtype=np.float32)
        for i, seq in enumerate(sequences):
            mat[i] = self._vectorize(seq)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def transform_one(self, seq: List[int]) -> np.ndarray:
        vec  = self._vectorize(seq)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def fit_transform(self, sequences: List[List[int]]) -> np.ndarray:
        self.fit(sequences)
        return self.transform(sequences)


# =============================================================================
# ── Section 6: Evaluation metrics  ───────────────────────────────────────────
# =============================================================================

def _ap(ranked_indices: List[int], relevant: Set[int]) -> float:
    """Average Precision for a single query."""
    if not relevant:
        return 0.0
    hits = ap = 0.0
    for rank, idx in enumerate(ranked_indices, 1):
        if idx in relevant:
            hits += 1
            ap   += hits / rank
    return ap / len(relevant)


def _mrr(ranked_lists: List[List[int]], ground_truth: List[Set[int]]) -> float:
    """Mean Reciprocal Rank."""
    rr = []
    for ranked, relevant in zip(ranked_lists, ground_truth):
        score = 0.0
        for rank, idx in enumerate(ranked, 1):
            if idx in relevant:
                score = 1.0 / rank
                break
        rr.append(score)
    return float(np.mean(rr))


def evaluate(
    ranked_lists: List[List[int]],
    ground_truth: List[Set[int]],
) -> Dict[str, float]:
    """
    Compute MAP, MRR, P@1, P@5, P@10.

    ranked_lists[q] must contain ALL corpus indices for MAP to be correct.
    """
    maps = [_ap(rl, gt) for rl, gt in zip(ranked_lists, ground_truth)]

    def precision_at_k(k: int) -> float:
        return float(np.mean([
            len(set(rl[:k]) & gt) / k
            for rl, gt in zip(ranked_lists, ground_truth)
        ]))

    return {
        "MAP":  float(np.mean(maps)),
        "MRR":  _mrr(ranked_lists, ground_truth),
        "P@1":  precision_at_k(1),
        "P@5":  precision_at_k(5),
        "P@10": precision_at_k(10),
    }


# =============================================================================
# ── Section 7: System 1 — TF-IDF Baseline  ───────────────────────────────────
# =============================================================================

class TFIDFBaseline:
    """
    Exact reproduction of Singh et al. (2024) TF-IDF QbE-STD.

    Unigram TF-IDF + brute-force cosine similarity. No HNSW, no SW.
    """

    def __init__(self):
        self.tfidf   = UnigramTFIDF()
        self._matrix: Optional[np.ndarray] = None

    def fit(self, corpus_seqs: List[List[int]]) -> "TFIDFBaseline":
        self._matrix = self.tfidf.fit_transform(corpus_seqs)
        return self

    def rank(self, query_seq: List[int]) -> List[int]:
        q_vec = self.tfidf.transform_one(query_seq)
        sims  = self._matrix @ q_vec
        return list(np.argsort(-sims))

    def run(
        self,
        query_seqs:   List[List[int]],
        ground_truth: List[Set[int]],
    ) -> Dict[str, float]:
        ranked = [self.rank(q) for q in query_seqs]
        return evaluate(ranked, ground_truth)


# =============================================================================
# ── Section 8: System 2 — H-QuEST  ───────────────────────────────────────────
# =============================================================================

class HQuEST:
    """
    Faithful reproduction of H-QuEST (Singh et al., Interspeech 2025).

    Pipeline
    --------
    Raw tokens
        -> Unigram TF-IDF vectorisation
        -> HNSW (M=16, ef_construction=150, ef_search=200, C=200)
        -> Top-C candidates by cosine distance
        -> Smith-Waterman reranking, score normalised by query length
        -> Remaining docs appended by cosine similarity

    Parameters
    ----------
    C            : int   HNSW candidate set size (paper: 200)
    sw_match     : float SW match score     (paper: +2)
    sw_mismatch  : float SW mismatch score  (paper: -1)
    sw_gap       : float SW gap penalty     (paper: -2)
    """

    def __init__(
        self,
        C:           int   = 200,
        sw_match:    float = 2.0,
        sw_mismatch: float = -1.0,
        sw_gap:      float = -2.0,
    ):
        self.C           = C
        self.sw_match    = sw_match
        self.sw_mismatch = sw_mismatch
        self.sw_gap      = sw_gap

        self.tfidf:        UnigramTFIDF         = UnigramTFIDF()
        self.hnsw:         HNSWIndex            = HNSWIndex(n_neighbors=C)
        self._corpus_seqs: List[List[int]]      = []
        self._matrix:      Optional[np.ndarray] = None

    def fit(self, corpus_seqs: List[List[int]]) -> "HQuEST":
        self._corpus_seqs = corpus_seqs
        self._matrix      = self.tfidf.fit_transform(corpus_seqs)
        self.hnsw.fit(self._matrix)
        return self

    def rank(self, query_seq: List[int]) -> List[int]:
        q_vec      = self.tfidf.transform_one(query_seq)
        candidates = self.hnsw.search(q_vec, k=self.C)
        reranked   = sw_rerank(
            query_seq, candidates, self._corpus_seqs,
            self.sw_match, self.sw_mismatch, self.sw_gap,
        )

        # Append non-candidate docs sorted by cosine similarity
        top_k_idxs = {idx for _, idx in reranked}
        sims = self._matrix @ q_vec
        remaining = [
            (float(sims[i]), i)
            for i in np.argsort(-sims)
            if i not in top_k_idxs
        ]

        return [idx for _, idx in reranked] + [idx for _, idx in remaining]

    def run(
        self,
        query_seqs:   List[List[int]],
        ground_truth: List[Set[int]],
    ) -> Dict[str, float]:
        ranked = [self.rank(q) for q in query_seqs]
        return evaluate(ranked, ground_truth)


# =============================================================================
# ── Section 9: System 3 — PMI-QuEST (proposed)  ──────────────────────────────
# =============================================================================

class PMIQuest:
    """
    PMI-QuEST: proposed method (paper §3).

    Pipeline
    --------
    Raw tokens
        -> PMI-TF-IDF (unigrams + PMI-filtered bigrams, tau=0.5, alpha=1.0)
        -> HNSW (M=16, ef_construction=150, ef_search=200, C=200)
        -> Smith-Waterman reranking, score normalised by query length
        -> Remaining docs appended by cosine similarity

    PMI bigram filter B_tau retains only adjacent token pairs whose joint
    probability exceeds the product of marginals by more than tau, capturing
    word-specific phoneme transitions with high discriminative value.

    Parameters
    ----------
    tau_pmi      : float   PMI threshold tau (paper: 0.5)
    alpha        : float   Bigram weight alpha in combined vector (paper: 1.0)
    C            : int     HNSW candidate count (paper: 200)
    sw_match     : float   SW match score   (paper: +2)
    sw_mismatch  : float   SW mismatch score (paper: -1)
    sw_gap       : float   SW gap penalty   (paper: -2)
    min_count    : int     Minimum bigram corpus count for PMI estimation
    """

    def __init__(
        self,
        tau_pmi:     float = 0.5,
        alpha:       float = 1.0,
        C:           int   = 200,
        sw_match:    float = 2.0,
        sw_mismatch: float = -1.0,
        sw_gap:      float = -2.0,
        min_count:   int   = 2,
    ):
        self.tau_pmi     = tau_pmi
        self.alpha       = alpha
        self.C           = C
        self.sw_match    = sw_match
        self.sw_mismatch = sw_mismatch
        self.sw_gap      = sw_gap
        self.min_count   = min_count

        self.pmi_tfidf:    PMIFilteredTFIDF     = PMIFilteredTFIDF(tau_pmi, alpha, min_count)
        self.hnsw:         HNSWIndex            = HNSWIndex(n_neighbors=C)
        self._corpus_seqs: List[List[int]]      = []
        self._matrix:      Optional[np.ndarray] = None

    def fit(self, corpus_seqs: List[List[int]]) -> "PMIQuest":
        self._corpus_seqs = corpus_seqs
        self._matrix      = self.pmi_tfidf.fit_transform(corpus_seqs)
        self.hnsw.fit(self._matrix)
        return self

    def rank(self, query_seq: List[int]) -> List[int]:
        q_vec      = self.pmi_tfidf.transform_one(query_seq)
        candidates = self.hnsw.search(q_vec, k=self.C)
        reranked   = sw_rerank(
            query_seq, candidates, self._corpus_seqs,
            self.sw_match, self.sw_mismatch, self.sw_gap,
        )

        # Append non-candidate docs sorted by cosine similarity
        top_k_idxs = {idx for _, idx in reranked}
        sims = self._matrix @ q_vec
        remaining = [
            (float(sims[i]), i)
            for i in np.argsort(-sims)
            if i not in top_k_idxs
        ]

        return [idx for _, idx in reranked] + [idx for _, idx in remaining]

    def run(
        self,
        query_seqs:   List[List[int]],
        ground_truth: List[Set[int]],
    ) -> Dict[str, float]:
        ranked = [self.rank(q) for q in query_seqs]
        return evaluate(ranked, ground_truth)


# =============================================================================
# ── Section 10: Comparison runner  ───────────────────────────────────────────
# =============================================================================

def _fmt(m: Dict[str, float]) -> str:
    return (
        f"MAP={m['MAP']:.4f}  MRR={m['MRR']:.4f}  "
        f"P@1={m['P@1']:.4f}  P@5={m['P@5']:.4f}  P@10={m['P@10']:.4f}"
    )


def run_comparison(
    corpus_seqs:  List[List[int]],
    query_seqs:   List[List[int]],
    ground_truth: List[Set[int]],
    verbose:      bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Run all three systems and return results dict.

    All systems use paper-default hyperparameters:
        TF-IDF    : brute-force cosine
        H-QuEST   : C=200, SW(+2/-1/-2), score/query_length
        PMI-QuEST : tau=0.5, alpha=1.0, C=200, SW(+2/-1/-2), score/query_length

    Returns
    -------
    {
        "tfidf_baseline" : {"MAP": ..., "MRR": ..., "P@1": ..., ...},
        "hquest"         : {...},
        "pmiquest"       : {...},
    }
    """
    results: Dict[str, Dict[str, float]] = {}
    N = len(corpus_seqs)
    Q = len(query_seqs)

    if verbose:
        print("=" * 70)
        print("PMI-QuEST vs H-QuEST vs TF-IDF  --  paper hyperparameters")
        print("=" * 70)
        print(f"  Corpus: {N} docs  |  Queries: {Q}")
        if not _HNSWLIB_AVAILABLE:
            print("  WARNING: hnswlib not installed; using sklearn fallback.")
        print()

    # ── 1. TF-IDF Baseline ──────────────────────────────────────────────────
    if verbose:
        print("-" * 70)
        print("[1/3] TF-IDF Baseline (Singh et al. 2024)")

    t0 = time.time()
    s1 = TFIDFBaseline()
    s1.fit(corpus_seqs)
    r1 = s1.run(query_seqs, ground_truth)
    r1["time"] = time.time() - t0
    results["tfidf_baseline"] = r1

    if verbose:
        print(f"  {_fmt(r1)}  [{r1['time']:.1f}s]")

    # ── 2. H-QuEST ──────────────────────────────────────────────────────────
    if verbose:
        print("-" * 70)
        print("[2/3] H-QuEST (Singh et al., Interspeech 2025)")
        print("       Unigram TF-IDF -> HNSW(C=200) -> SW(+2/-1/-2)/query_len")

    t0 = time.time()
    s2 = HQuEST(C=200, sw_match=2.0, sw_mismatch=-1.0, sw_gap=-2.0)
    s2.fit(corpus_seqs)
    r2 = s2.run(query_seqs, ground_truth)
    r2["time"] = time.time() - t0
    results["hquest"] = r2

    if verbose:
        print(f"  {_fmt(r2)}  [{r2['time']:.1f}s]")

    # ── 3. PMI-QuEST ────────────────────────────────────────────────────────
    if verbose:
        print("-" * 70)
        print("[3/3] PMI-QuEST (proposed)")
        print("       PMI-TF-IDF(tau=0.5, alpha=1.0) -> HNSW(C=200) -> SW/query_len")

    t0 = time.time()
    s3 = PMIQuest(tau_pmi=0.5, alpha=1.0, C=200)
    s3.fit(corpus_seqs)
    r3 = s3.run(query_seqs, ground_truth)
    r3["time"] = time.time() - t0
    results["pmiquest"] = r3

    if verbose:
        print(f"  {_fmt(r3)}  [{r3['time']:.1f}s]")

    # ── Summary table ────────────────────────────────────────────────────────
    if verbose:
        print()
        print("=" * 70)
        print(f"{'System':<38} {'MAP':>6} {'MRR':>6} {'P@1':>6} {'P@5':>6} {'P@10':>6}")
        print("-" * 70)
        for key, label in [
            ("tfidf_baseline", "TF-IDF Baseline"),
            ("hquest",         "H-QuEST"),
            ("pmiquest",       "PMI-QuEST (proposed)"),
        ]:
            m = results[key]
            print(
                f"  {label:<36} {m['MAP']:>6.4f} {m['MRR']:>6.4f}"
                f" {m['P@1']:>6.4f} {m['P@5']:>6.4f} {m['P@10']:>6.4f}"
            )
        print("=" * 70)

    return results


# =============================================================================
# ── Section 11: Quick self-test with synthetic data  ─────────────────────────
# =============================================================================

if __name__ == "__main__":
    import random

    rng = random.Random(42)
    print("PMI-QuEST self-test -- synthetic data")
    print("Replace with real LibriSpeech token sequences for paper results.\n")

    V = 100   # vocab size
    N = 200   # corpus size
    Q = 20    # queries

    # Synthetic corpus: random token sequences, length 50-300
    corpus_seqs = [
        [rng.randint(0, V - 1) for _ in range(rng.randint(50, 300))]
        for _ in range(N)
    ]

    # Short queries (~12 tokens), relevant = docs where we plant the pattern
    query_seqs:   List[List[int]] = []
    ground_truth: List[Set[int]]  = []

    for _ in range(Q):
        pattern      = [rng.randint(0, V - 1) for _ in range(12)]
        relevant_set = set(rng.sample(range(N), k=rng.randint(3, 8)))
        for idx in relevant_set:
            pos = rng.randint(0, max(0, len(corpus_seqs[idx]) - 12))
            corpus_seqs[idx] = (
                corpus_seqs[idx][:pos] + pattern + corpus_seqs[idx][pos + 12:]
            )
        query_seqs.append(pattern)
        ground_truth.append(relevant_set)

    run_comparison(corpus_seqs, query_seqs, ground_truth)
