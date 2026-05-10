"""
BPE (Byte Pair Encoding) for discrete audio token sequences.

Learns merge operations on a corpus of token sequences and applies
them to compress sequences by replacing frequent bigrams with new tokens.
"""

from collections import Counter
from typing import List, Tuple, Dict


class AudioBPE:
    """
    Byte Pair Encoding for discrete audio token sequences.

    Learns K merge operations from a corpus, then applies them
    to compress token sequences. Frequent bigrams become atomic tokens,
    leaving only rarer patterns as separate tokens — directly
    complementing TF-IDF's IDF mechanism.
    """

    def __init__(self, num_merges: int = 200):
        """
        Args:
            num_merges: Number of BPE merge operations to learn (K).
                        Typical range: 50–500 depending on vocab size
                        and corpus size.
        """
        self.num_merges = num_merges
        self.merges: List[Tuple] = []          # ordered list of (a, b) merge pairs
        self.merge_map: Dict[Tuple, int] = {}  # (a, b) -> new_token_id
        self.vocab_size: int = 0               # base vocab size before merges
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, sequences: List[List[int]]) -> "AudioBPE":
        """
        Learn BPE merge operations from a corpus of token sequences.

        Args:
            sequences: List of token sequences, each a list of int token IDs.

        Returns:
            self (for chaining)
        """
        if not sequences:
            raise ValueError("Cannot fit BPE on empty corpus.")

        # Determine base vocabulary size from the data
        all_tokens = {t for seq in sequences for t in seq}
        self.vocab_size = max(all_tokens) + 1
        next_token_id = self.vocab_size

        # Work on mutable copies
        corpus = [list(seq) for seq in sequences]

        print(f"[BPE] Fitting on {len(corpus)} sequences, "
              f"base vocab size = {self.vocab_size}, "
              f"num_merges = {self.num_merges}")

        for merge_idx in range(self.num_merges):
            # Count all adjacent pairs across the entire corpus
            pair_counts = Counter()
            for seq in corpus:
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += 1

            if not pair_counts:
                print(f"[BPE] No more pairs to merge at step {merge_idx}. Stopping.")
                break

            # Find most frequent pair
            best_pair, best_count = pair_counts.most_common(1)[0]

            if best_count < 2:
                print(f"[BPE] All remaining pairs occur only once. Stopping at step {merge_idx}.")
                break

            # Record the merge
            self.merges.append(best_pair)
            self.merge_map[best_pair] = next_token_id

            # Apply merge to entire corpus
            corpus = [self._apply_merge(seq, best_pair, next_token_id)
                      for seq in corpus]

            if merge_idx % 50 == 0:
                avg_len = sum(len(s) for s in corpus) / len(corpus)
                print(f"[BPE] Merge {merge_idx:4d}: {best_pair} -> {next_token_id} "
                      f"(count={best_count}), avg seq len = {avg_len:.1f}")

            next_token_id += 1

        self.is_fitted = True
        print(f"[BPE] Done. Learned {len(self.merges)} merges. "
              f"Final vocab size = {next_token_id}.")
        return self

    # ------------------------------------------------------------------
    # Transformation
    # ------------------------------------------------------------------

    def transform(self, sequences: List[List[int]]) -> List[List[int]]:
        """
        Apply learned BPE merges to compress a list of sequences.

        Args:
            sequences: List of token sequences (raw, using base vocab).

        Returns:
            List of compressed sequences using extended BPE vocab.
        """
        if not self.is_fitted:
            raise RuntimeError("BPE must be fitted before calling transform().")
        return [self._compress(seq) for seq in sequences]

    def fit_transform(self, sequences: List[List[int]]) -> List[List[int]]:
        """Fit and transform in one step."""
        self.fit(sequences)
        return self.transform(sequences)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_merge(self,
                     seq: List[int],
                     pair: Tuple[int, int],
                     new_token: int) -> List[int]:
        """Replace all non-overlapping occurrences of pair in seq with new_token."""
        result = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                result.append(new_token)
                i += 2
            else:
                result.append(seq[i])
                i += 1
        return result

    def _compress(self, seq: List[int]) -> List[int]:
        seq = list(seq)
        for pair in self.merges:              # ← use ordered list, not merge_map.items()
            new_token = self.merge_map[pair]
            seq = self._apply_merge(seq, pair, new_token)
        return seq

    # ------------------------------------------------------------------
    # Analysis utilities
    # ------------------------------------------------------------------

    def get_merge_frequencies(self,
                              sequences: List[List[int]]) -> List[Tuple[Tuple, int]]:
        """
        Return how many times each merge was applied across sequences.
        Useful for analysis: top merges = most common acoustic bigrams.
        """
        counts = []
        corpus = [list(seq) for seq in sequences]
        for pair, new_token in self.merge_map.items():
            count = sum(
                sum(1 for i in range(len(s) - 1)
                    if s[i] == pair[0] and s[i + 1] == pair[1])
                for s in corpus
            )
            counts.append((pair, count))
            corpus = [self._apply_merge(s, pair, new_token) for s in corpus]
        return sorted(counts, key=lambda x: -x[1])

    def compression_ratio(self,
                          original: List[List[int]],
                          compressed: List[List[int]]) -> float:
        """Compute mean compression ratio L' / L across the corpus."""
        ratios = [len(c) / len(o) for o, c in zip(original, compressed) if len(o) > 0]
        return sum(ratios) / len(ratios) if ratios else 1.0
