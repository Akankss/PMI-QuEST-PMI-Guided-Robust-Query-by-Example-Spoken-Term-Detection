"""
experiments/run_ablation_bpe.py
================================
Group G — Byte-Pair Encoding ablation for PMI-QuEST.

This file isolates the effect of BPE pre-tokenization on the discrete
acoustic token sequences *before* PMI-QuEST runs.  All PMI-QuEST
hyperparameters are held at their proposed defaults (τ=0.5, α=0.5, K=50).

Sub-groups
----------
  G1 — No BPE (raw k-means token sequences)            [baseline]
  G2 — BPE vocab sizes: 256, 512, 1024, 2048, 4096
  G3 — BPE min-frequency threshold: 2, 5, 10, 20, 50
  G4 — BPE + PMI-bigrams vs BPE-only (no PMI bigrams)
  G5 — BPE applied to corpus-only vs corpus+queries    [coverage study]

Design notes
------------
* BPE is fit on the *corpus* token sequences only (to avoid query leakage),
  then applied to both corpus and queries at inference time.
* The BPE implementation here is a clean, dependency-free version that
  operates on integer token sequences directly (not text bytes), which is
  exactly what PMI-QuEST expects from its upstream tokenizer.
* Each BPE merge produces a new synthetic token ID beyond the original
  k-means vocabulary, keeping the integer-sequence contract intact.

Usage
-----
    python run_ablation_bpe.py \\
        --corpus    corpus_tokens.csv \\
        --queries   query_tokens.csv  \\
        --relevance relevance.json    \\
        --out       results/ablation_bpe.csv \\
        --groups    G1G2G3G4G5
"""

from __future__ import annotations
import argparse
import collections
import csv as _csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloader import load_corpus, load_queries, load_relevance
from pmiquest_system import PMIQuest, evaluate

# ── proposed PMI-QuEST defaults (held fixed throughout) ──────────────────────
DEFAULT_PMI_TAU       = 0.5
DEFAULT_BIGRAM_WEIGHT = 1
DEFAULT_HNSW_K        = 200
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════
# Integer-sequence BPE
# ═══════════════════════════════════════════════════════════════════════════

class IntegerBPE:
    """
    Byte-Pair Encoding over integer token sequences.

    Operates on List[List[int]] rather than text.  Each merge replaces the
    most-frequent adjacent pair (a, b) with a new synthetic token ID
    (base_vocab_size + merge_index).

    Parameters
    ----------
    vocab_size : int
        Target vocabulary size (base + synthetic tokens).  Training stops
        when the vocabulary reaches this size *or* no pair exceeds
        min_frequency.
    min_frequency : int
        A pair must appear at least this many times to be merged.
    fit_on_corpus_only : bool
        If True (default), merge rules are learned from corpus sequences
        only; queries are only *encoded* using the learned rules.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        min_frequency: int = 2,
        fit_on_corpus_only: bool = True,
    ):
        self.vocab_size        = vocab_size
        self.min_frequency     = min_frequency
        self.fit_on_corpus_only = fit_on_corpus_only

        self.merges_: List[Tuple[int, int]] = []   # ordered merge rules
        self.merge_map_: Dict[Tuple[int, int], int] = {}
        self.base_vocab_size_: int = 0
        self.n_merges_applied_: int = 0

    # ── training ────────────────────────────────────────────────────────────

    def fit(self, sequences: List[List[int]]) -> "IntegerBPE":
        """Learn BPE merge rules from sequences."""
        if not sequences:
            return self

        # Infer base vocabulary size from the data
        self.base_vocab_size_ = max(t for seq in sequences for t in seq) + 1
        next_id = self.base_vocab_size_
        current = [list(seq) for seq in sequences]   # working copy

        max_merges = self.vocab_size - self.base_vocab_size_
        if max_merges <= 0:
            # vocab_size already satisfied by base tokens
            return self

        for _ in range(max_merges):
            pair_counts = self._count_pairs(current)
            if not pair_counts:
                break
            best_pair, best_count = max(pair_counts.items(), key=lambda x: x[1])
            if best_count < self.min_frequency:
                break

            self.merges_.append(best_pair)
            self.merge_map_[best_pair] = next_id
            current = self._apply_merge(current, best_pair, next_id)
            next_id += 1

        self.n_merges_applied_ = len(self.merges_)
        return self

    # ── encoding ────────────────────────────────────────────────────────────

    def encode(self, sequences: List[List[int]]) -> List[List[int]]:
        """Apply learned merge rules to sequences."""
        if not self.merges_:
            return [list(s) for s in sequences]

        current = [list(seq) for seq in sequences]
        for pair, new_id in self.merge_map_.items():
            current = self._apply_merge(current, pair, new_id)
        return current

    def fit_encode(
        self,
        corpus_seqs:  List[List[int]],
        query_seqs:   List[List[int]],
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Fit on corpus, encode both corpus and queries.
        Returns (encoded_corpus, encoded_queries).
        """
        self.fit(corpus_seqs)
        enc_corpus  = self.encode(corpus_seqs)
        enc_queries = self.encode(query_seqs)
        return enc_corpus, enc_queries

    # ── internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _count_pairs(sequences: List[List[int]]) -> Dict[Tuple[int, int], int]:
        counts: Dict[Tuple[int, int], int] = collections.defaultdict(int)
        for seq in sequences:
            for a, b in zip(seq, seq[1:]):
                counts[(a, b)] += 1
        return counts

    @staticmethod
    def _apply_merge(
        sequences: List[List[int]],
        pair:      Tuple[int, int],
        new_id:    int,
    ) -> List[List[int]]:
        a, b = pair
        out = []
        for seq in sequences:
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    new_seq.append(new_id)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            out.append(new_seq)
        return out


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def build_groundtruth(
    query_filenames:  List[str],
    corpus_filenames: List[str],
    relevance:        Dict[str, List[str]],
) -> List[Set[int]]:
    corpus_stem_map = {Path(cf).stem: i for i, cf in enumerate(corpus_filenames)}
    suffix_to_key: Dict[str, str] = {}
    for k in relevance:
        suffix_to_key[k.split("_")[-1]] = k

    gt = []
    for qf in query_filenames:
        q_stem = Path(qf).stem
        rel_raw = relevance.get(q_stem)
        if rel_raw is None:
            fallback_key = suffix_to_key.get(q_stem.split("_")[-1], "")
            rel_raw = relevance.get(fallback_key, [])
        rel_idxs: Set[int] = set()
        for r in rel_raw:
            stem = Path(r).stem
            if stem in corpus_stem_map:
                rel_idxs.add(corpus_stem_map[stem])
        gt.append(rel_idxs)
    return gt


def proposed_system(use_pmi_bigrams: bool = True) -> PMIQuest:
    return PMIQuest(
        use_pmitd=False,
        use_pmi_bigrams=use_pmi_bigrams,
        pmi_tau=DEFAULT_PMI_TAU,
        bigram_weight=DEFAULT_BIGRAM_WEIGHT,
        hnsw_k=DEFAULT_HNSW_K,
    )


def seq_stats(seqs: List[List[int]]) -> str:
    lengths = [len(s) for s in seqs]
    return (
        f"len μ={np.mean(lengths):.1f} "
        f"σ={np.std(lengths):.1f} "
        f"vocab={len({t for s in seqs for t in s})}"
    )


def run_variant(
    tag:          str,
    group:        str,
    description:  str,
    corpus_seqs:  List[List[int]],
    query_seqs:   List[List[int]],
    ground_truth: List[Set[int]],
    bpe:          Optional[IntegerBPE],
    use_pmi_bigrams: bool = True,
) -> Dict:
    """
    Optionally apply BPE, fit PMI-QuEST, evaluate, return result row.
    """
    print(f"\n  [{group}] {tag}: {description}")

    t0 = time.time()

    # ── BPE encoding ─────────────────────────────────────────────
    if bpe is not None:
        enc_corpus, enc_queries = bpe.fit_encode(corpus_seqs, query_seqs)
        n_merges = bpe.n_merges_applied_
        bpe_vocab = bpe.base_vocab_size_ + n_merges
    else:
        enc_corpus, enc_queries = corpus_seqs, query_seqs
        n_merges = 0
        bpe_vocab = len({t for s in corpus_seqs for t in s})

    print(f"    corpus  → {seq_stats(enc_corpus)}")
    print(f"    queries → {seq_stats(enc_queries)}")

    # ── PMI-QuEST ────────────────────────────────────────────────
    system = proposed_system(use_pmi_bigrams=use_pmi_bigrams)
    system.fit(enc_corpus)
    ranked  = [system.rank(q) for q in enc_queries]
    metrics = evaluate(ranked, ground_truth)
    elapsed = time.time() - t0

    if hasattr(system, "pmi_tfidf") and getattr(system, "use_pmi_bigrams", False):
        n_pmi_bigrams = len(system.pmi_tfidf.bi_vocab)
    else:
        n_pmi_bigrams = 0

    row = {
        "group":        group,
        "tag":          tag,
        "description":  description,
        "bpe_vocab_size":   bpe.vocab_size   if bpe else 0,
        "bpe_min_freq":     bpe.min_frequency if bpe else 0,
        "bpe_merges_done":  n_merges,
        "effective_vocab":  bpe_vocab,
        "use_pmi_bigrams":  use_pmi_bigrams,
        "n_pmi_bigrams":    n_pmi_bigrams,
        "MAP":  round(metrics["MAP"],  4),
        "P@1":  round(metrics["P@1"],  4),
        "P@5":  round(metrics["P@5"],  4),
        "P@10": round(metrics["P@10"], 4),
        "MRR":  round(metrics.get("MRR", float("nan")), 4),
        "time_s": round(elapsed, 1),
    }
    print(
        f"    MAP={row['MAP']:.4f}  P@1={row['P@1']:.4f}  "
        f"MRR={row['MRR']:.4f}  "
        f"bpe_merges={n_merges}  pmi_bigrams={n_pmi_bigrams}  "
        f"[{elapsed:.1f}s]"
    )
    return row


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Group G — BPE ablation for PMI-QuEST"
    )
    p.add_argument("--corpus",    required=True)
    p.add_argument("--queries",   required=True)
    p.add_argument("--relevance", required=True)
    p.add_argument("--out",       default="results/ablation_bpe.csv")
    p.add_argument(
        "--groups", default="G1G2G3G4G5",
        help="Which sub-groups to run, e.g. --groups G1G2"
    )
    args = p.parse_args()

    print("=" * 65)
    print("PMI-QuEST  —  BPE Ablation (Group G)")
    print("=" * 65)

    corpus_filenames, corpus_seqs = load_corpus(args.corpus)
    query_filenames,  query_seqs  = load_queries(args.queries)
    relevance    = load_relevance(args.relevance)
    ground_truth = build_groundtruth(
        query_filenames, corpus_filenames, relevance
    )

    rows: List[Dict] = []
    G = args.groups.upper()

    # ── G1: No BPE baseline ───────────────────────────────────────
    if "G1" in G:
        print("\n" + "─" * 55)
        print("Group G1: No BPE — raw k-means tokens  [baseline]")
        print("─" * 55)

        rows.append(run_variant(
            tag="G1", group="G1",
            description="No BPE — raw tokens  [baseline]",
            corpus_seqs=corpus_seqs, query_seqs=query_seqs,
            ground_truth=ground_truth,
            bpe=None,
        ))

    # ── G2: BPE vocab size sweep ──────────────────────────────────
    if "G2" in G:
        print("\n" + "─" * 55)
        print("Group G2: BPE Vocabulary Size  (min_freq=2 fixed)")
        print("─" * 55)

        base_vocab = len({t for s in corpus_seqs for t in s})
        for vocab_size in [256, 512, 1024, 2048, 4096, 8192]:
            # Skip vocab sizes smaller than base k-means vocab
            if vocab_size <= base_vocab:
                print(f"  [G2] vocab_size={vocab_size} "
                      f"≤ base_vocab={base_vocab}, skipping.")
                continue
            rows.append(run_variant(
                tag=f"G2_v{vocab_size}", group="G2",
                description=f"BPE vocab_size={vocab_size}  min_freq=2",
                corpus_seqs=corpus_seqs, query_seqs=query_seqs,
                ground_truth=ground_truth,
                bpe=IntegerBPE(vocab_size=vocab_size, min_frequency=2),
            ))

    # ── G3: BPE min-frequency sweep ───────────────────────────────
    if "G3" in G:
        print("\n" + "─" * 55)
        print("Group G3: BPE Min-Frequency Threshold  (vocab_size=2048 fixed)")
        print("─" * 55)

        for min_freq in [2, 5, 10, 20, 50]:
            rows.append(run_variant(
                tag=f"G3_mf{min_freq}", group="G3",
                description=f"BPE min_freq={min_freq}  vocab_size=2048",
                corpus_seqs=corpus_seqs, query_seqs=query_seqs,
                ground_truth=ground_truth,
                bpe=IntegerBPE(vocab_size=2048, min_frequency=min_freq),
            ))

    # ── G4: BPE + PMI-bigrams vs BPE alone ───────────────────────
    if "G4" in G:
        print("\n" + "─" * 55)
        print("Group G4: BPE × PMI-bigrams interaction")
        print("─" * 55)

        for vocab_size in [ 4096, 8192,2048]:
            for use_pmi in [False, True]:
                pmi_label = "PMI-bigrams=ON" if use_pmi else "PMI-bigrams=OFF"
                rows.append(run_variant(
                    tag=f"G4_v{vocab_size}_pmi{int(use_pmi)}", group="G4",
                    description=f"BPE vocab={vocab_size}  {pmi_label}",
                    corpus_seqs=corpus_seqs, query_seqs=query_seqs,
                    ground_truth=ground_truth,
                    bpe=IntegerBPE(vocab_size=vocab_size, min_frequency=2),
                    use_pmi_bigrams=use_pmi,
                ))

    # ── G5: Query coverage — corpus-only BPE vs joint BPE ─────────
    if "G5" in G:
        print("\n" + "─" * 55)
        print("Group G5: BPE Coverage — corpus-only vs corpus+query fit")
        print("─" * 55)

        for vocab_size in [1024, 2048]:
            # Corpus-only fit (standard; queries see OOV synthetic tokens)
            rows.append(run_variant(
                tag=f"G5_v{vocab_size}_corpusonly", group="G5",
                description=f"BPE vocab={vocab_size}  fit=corpus-only",
                corpus_seqs=corpus_seqs, query_seqs=query_seqs,
                ground_truth=ground_truth,
                bpe=IntegerBPE(
                    vocab_size=vocab_size,
                    min_frequency=2,
                    fit_on_corpus_only=True,
                ),
            ))

            # Joint fit (queries included in BPE training — data-leakage
            # risk in practice, but informative upper bound on coverage)
            joint_bpe = IntegerBPE(
                vocab_size=vocab_size,
                min_frequency=2,
                fit_on_corpus_only=False,
            )
            joint_bpe.fit(corpus_seqs + query_seqs)
            enc_corpus  = joint_bpe.encode(corpus_seqs)
            enc_queries = joint_bpe.encode(query_seqs)

            # Run manually so we can pass pre-encoded sequences
            print(f"\n  [G5] G5_v{vocab_size}_joint: "
                  f"BPE vocab={vocab_size}  fit=corpus+queries  "
                  f"(upper-bound coverage)")
            t0 = time.time()
            system = proposed_system()
            system.fit(enc_corpus)
            ranked  = [system.rank(q) for q in enc_queries]
            metrics = evaluate(ranked, ground_truth)
            elapsed = time.time() - t0
            n_pmi = (
                len(system.pmi_tfidf.bi_vocab)
                if hasattr(system, "pmi_tfidf") and system.use_pmi_bigrams
                else 0
            )
            row = {
                "group":           "G5",
                "tag":             f"G5_v{vocab_size}_joint",
                "description":     f"BPE vocab={vocab_size}  fit=corpus+queries",
                "bpe_vocab_size":  vocab_size,
                "bpe_min_freq":    2,
                "bpe_merges_done": joint_bpe.n_merges_applied_,
                "effective_vocab": joint_bpe.base_vocab_size_ + joint_bpe.n_merges_applied_,
                "use_pmi_bigrams": True,
                "n_pmi_bigrams":   n_pmi,
                "MAP":  round(metrics["MAP"],  4),
                "P@1":  round(metrics["P@1"],  4),
                "P@5":  round(metrics["P@5"],  4),
                "P@10": round(metrics["P@10"], 4),
                "MRR":  round(metrics.get("MRR", float("nan")), 4),
                "time_s": round(elapsed, 1),
            }
            rows.append(row)
            print(
                f"    MAP={row['MAP']:.4f}  P@1={row['P@1']:.4f}  "
                f"MRR={row['MRR']:.4f}  [{elapsed:.1f}s]"
            )

    # ── save CSV ──────────────────────────────────────────────────
    if rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  Saved → {args.out}")

    # ── console summary ───────────────────────────────────────────
    print()
    print("=" * 65)
    print(
        f"  {'tag':<26} {'grp':<5}  "
        f"{'MAP':>7} {'P@1':>7} {'MRR':>7} {'merges':>7}"
    )
    print("  " + "-" * 60)
    for r in rows:
        print(
            f"  {r['tag']:<26} {r['group']:<5}  "
            f"{r['MAP']:>7.4f} {r['P@1']:>7.4f} "
            f"{r['MRR']:>7.4f} {r['bpe_merges_done']:>7d}"
        )
    print("=" * 65)
    print("\nDone.")


if __name__ == "__main__":
    main()