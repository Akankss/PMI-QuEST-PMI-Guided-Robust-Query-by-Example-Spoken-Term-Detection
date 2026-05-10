"""
Mixed N-gram TF-IDF vectorizer for discrete token sequences.

Builds sparse TF-IDF vectors combining multiple n-gram orders,
with length-aware weighting w_n = 1/sqrt(n) to balance
bigrams and trigrams.
"""

import math
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
from scipy.sparse import csr_matrix


class MixedNgramTFIDF:
    """
    Mixed N-gram TF-IDF vectorizer.

    For each sequence, extracts n-grams of all orders in N,
    computes TF normalized by number of possible positions,
    weights by IDF, and applies the length penalty w_n = 1/sqrt(n).

    Final weight for n-gram g of order n in sequence d:
        x_g(d) = (1/sqrt(n)) * TF(g,d) * IDF(g)
    where:
        TF(g,d)  = count(g in d) / (L - n + 1)
        IDF(g)   = log(N / DF(g))
    """

    def __init__(self, ngram_orders: Tuple[int, ...] = (1, 2, 3)):
        """
        Args:
            ngram_orders: Which n-gram orders to include.
                          (1,2,3) gives unigrams + bigrams + trigrams.
                          Include 1 for backward compat with original H-QuEST.
        """
        self.ngram_orders = ngram_orders
        self.vocab: Dict[tuple, int] = {}       # ngram tuple -> column index
        self.idf: Dict[tuple, float] = {}       # ngram tuple -> IDF value
        self.n_docs: int = 0
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, sequences: List[List[int]]) -> "MixedNgramTFIDF":
        """
        Build vocabulary and IDF values from a corpus.

        Args:
            sequences: List of (BPE-compressed) token sequences.

        Returns:
            self
        """
        self.n_docs = len(sequences)
        doc_freq: Dict[tuple, int] = defaultdict(int)

        print(f"[TFIDF] Building vocabulary from {self.n_docs} sequences, "
              f"n-gram orders = {self.ngram_orders}")

        for seq in sequences:
            # Collect unique ngrams in this document for DF counting
            seen: Set[tuple] = set()
            for n in self.ngram_orders:
                for ngram in self._extract_ngrams(seq, n):
                    seen.add(ngram)
            for ngram in seen:
                doc_freq[ngram] += 1

        # Build vocab (sorted for determinism)
        all_ngrams = sorted(doc_freq.keys())
        self.vocab = {ng: idx for idx, ng in enumerate(all_ngrams)}

        # Compute IDF: log(N / DF(g))
        self.idf = {
            ng: math.log(self.n_docs / df)
            for ng, df in doc_freq.items()
        }

        self.is_fitted = True
        print(f"[TFIDF] Vocabulary size = {len(self.vocab)} n-grams.")
        return self

    # ------------------------------------------------------------------
    # Transformation
    # ------------------------------------------------------------------

    def transform(self, sequences: List[List[int]]) -> csr_matrix:
        """
        Transform sequences into sparse TF-IDF matrix.

        Args:
            sequences: List of (BPE-compressed) token sequences.

        Returns:
            Sparse matrix of shape (n_sequences, vocab_size).
            Each row is the TF-IDF vector for one sequence.
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before transform().")

        rows, cols, data = [], [], []
        vocab_size = len(self.vocab)

        for doc_idx, seq in enumerate(sequences):
            weights = self._vectorize(seq)
            for col_idx, weight in weights.items():
                rows.append(doc_idx)
                cols.append(col_idx)
                data.append(weight)

        matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(sequences), vocab_size),
            dtype=np.float32
        )
        return matrix

    def fit_transform(self, sequences: List[List[int]]) -> csr_matrix:
        """Fit and transform in one step."""
        self.fit(sequences)
        return self.transform(sequences)

    def transform_one(self, seq: List[int]) -> np.ndarray:
        """
        Transform a single sequence into a dense TF-IDF vector.
        Used for query vectorization at retrieval time.

        Args:
            seq: Single (BPE-compressed) token sequence.

        Returns:
            Dense numpy array of shape (vocab_size,).
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before transform_one().")

        vec = np.zeros(len(self.vocab), dtype=np.float32)
        weights = self._vectorize(seq)
        for col_idx, weight in weights.items():
            vec[col_idx] = weight
        return vec

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_ngrams(self, seq: List[int], n: int) -> List[tuple]:
        """Extract all contiguous n-grams of order n from seq."""
        return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]

    def _vectorize(self, seq: List[int]) -> Dict[int, float]:
        """
        Compute the TF-IDF weight for each vocabulary n-gram in seq.

        Returns:
            Dict mapping column index -> TF-IDF weight.
        """
        weights: Dict[int, float] = {}
        L = len(seq)

        for n in self.ngram_orders:
            if L < n:
                continue

            # Length-aware weight: w_n = 1 / sqrt(n)
            w_n = 1.0 / math.sqrt(n)

            # Count n-gram occurrences
            ngram_counts = Counter(self._extract_ngrams(seq, n))
            n_positions = L - n + 1  # denominator for TF

            for ngram, count in ngram_counts.items():
                if ngram not in self.vocab:
                    continue  # unseen at fit time — skip

                col_idx = self.vocab[ngram]
                idf_val = self.idf.get(ngram, 0.0)

                if idf_val == 0.0:
                    continue  # ngram appears in every doc — useless

                tf = count / n_positions
                weights[col_idx] = w_n * tf * idf_val

        return weights

    # ------------------------------------------------------------------
    # Analysis utilities
    # ------------------------------------------------------------------

    def top_ngrams_by_idf(self, n: int = 20) -> List[Tuple[tuple, float]]:
        """Return the n most discriminative n-grams by IDF score."""
        return sorted(self.idf.items(), key=lambda x: -x[1])[:n]

    def idf_distribution_stats(self) -> Dict[str, float]:
        """Summary statistics of IDF values across vocabulary."""
        vals = np.array(list(self.idf.values()))
        return {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "median": float(np.median(vals)),
        }
