"""
experiments/run_ablation.py
============================
Structured ablation sweep for PMI-QuEST.

Groups
------
  A — Bigram presence       (unigrams-only vs PMI-TF-IDF)
  B — PMI threshold τ       (0.0, 0.5, 1.0, 1.5, 2.0)
  C — Bigram weight α       (0.00, 0.25, 0.50, 0.75, 1.00)
  D — HNSW candidates K     (10, 20, 50, 100, 200)
  E — SW reranking          (with vs without)
  F — PMI-TD merges         (pre-computed from prior run)

Usage
-----
    python run_ablation.py \\
        --corpus    corpus_tokens.csv \\
        --queries   query_tokens.csv  \\
        --relevance relevance.json    \\
        --out       results/ablation_structured.csv \\
        --groups    ABCDE
"""

from __future__ import annotations
import argparse
import csv as _csv
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Set
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloader import load_corpus, load_queries, load_relevance
from pmiquest_system import PMIQuest, PMITokenDedup, evaluate

# ── canonical "proposed" hyperparameters ─────────────────────────────────────
# pmi_tau       → PMIQuest(pmi_tau=...)
# bigram_weight → PMIQuest(bigram_weight=...)   [= α in the paper]
# hnsw_k        → PMIQuest(hnsw_k=...)
DEFAULT_PMI_TAU       = 0.5
DEFAULT_BIGRAM_WEIGHT = 1
DEFAULT_HNSW_K        = 200
# ─────────────────────────────────────────────────────────────────────────────


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


class NoSWPMIQuest(PMIQuest):
    """
    PMIQuest subclass that skips SW reranking — returns all corpus docs
    in pure HNSW cosine order. Used to isolate SW's contribution (Group E).
    """
    def rank(self, query_seq: List[int]) -> List[int]:
        q_comp = self._compress_query(query_seq)
        if self.use_pmi_bigrams:
            q_vec = self.pmi_tfidf.transform_one(q_comp)
        else:
            q_vec = self.fallback_tfidf.transform_one(q_comp)
        sims = self._matrix @ q_vec
        return list(np.argsort(-sims))


def run_variant(
    tag:          str,
    group:        str,
    description:  str,
    corpus_seqs:  List[List[int]],
    query_seqs:   List[List[int]],
    ground_truth: List[Set[int]],
    system,
) -> Dict:
    """Fit system, run all queries, evaluate, return result row."""
    print(f"\n  [{group}] {tag}: {description}")

    t0 = time.time()
    system.fit(corpus_seqs)
    ranked = [system.rank(q) for q in query_seqs]
    metrics = evaluate(ranked, ground_truth)
    elapsed = time.time() - t0

    # Count bigrams in the fitted vectoriser
    if hasattr(system, "pmi_tfidf") and getattr(system, "use_pmi_bigrams", False):
        n_bigrams = len(system.pmi_tfidf.bi_vocab)
    else:
        n_bigrams = 0

    row = {
        "group":       group,
        "tag":         tag,
        "description": description,
        "pmi_tau":     getattr(system, "pmi_tau",       DEFAULT_PMI_TAU),
        "alpha":       getattr(system, "bigram_weight", DEFAULT_BIGRAM_WEIGHT),
        "hnsw_k":      getattr(system, "hnsw_k",       DEFAULT_HNSW_K),
        "n_bigrams":   n_bigrams,
        "MAP":         round(metrics["MAP"],  4),
        "MRR":         round(metrics["MRR"],  4),
        "P@1":         round(metrics["P@1"],  4),
        "P@5":         round(metrics["P@5"],  4),
        "P@10":        round(metrics["P@10"], 4),
        "time_s":      round(elapsed, 1),
    }
    print(f"    MAP={row['MAP']:.4f}  MRR={row['MRR']:.4f} P@1={row['P@1']:.4f}  "
          f"P@5={row['P@5']:.4f}  P@10={row['P@10']:.4f}  "
          f"bigrams={n_bigrams}  [{elapsed:.1f}s]")
    return row


def proposed_system(**overrides) -> PMIQuest:
    """Return a PMIQuest instance at proposed defaults, with optional overrides."""
    kwargs = dict(
        use_pmitd=False,
        use_pmi_bigrams=True,
        pmi_tau=DEFAULT_PMI_TAU,
        bigram_weight=DEFAULT_BIGRAM_WEIGHT,
        hnsw_k=DEFAULT_HNSW_K,
    )
    kwargs.update(overrides)
    return PMIQuest(**kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus",    required=True)
    p.add_argument("--queries",   required=True)
    p.add_argument("--relevance", required=True)
    p.add_argument("--out",       default="results/ablation_structured.csv")
    p.add_argument("--groups",    default="ABCDEF",
                   help="Which groups to run, e.g. --groups BC")
    args = p.parse_args()

    print("=" * 65)
    print("PMI-QuEST  —  Structured Ablation Suite")
    print("=" * 65)

    corpus_filenames, corpus_seqs = load_corpus(args.corpus)
    query_filenames,  query_seqs  = load_queries(args.queries)
    relevance    = load_relevance(args.relevance)
    ground_truth = build_groundtruth(
        query_filenames, corpus_filenames, relevance)

    rows = []
    G = args.groups.upper()

    # ── Group A: Bigram presence ──────────────────────────────────
    if "A" in G:
        print("\n" + "─"*55)
        print("Group A: Bigram Presence")
        print("─"*55)

        rows.append(run_variant(
            tag="A1", group="A",
            description="Unigrams only (no PMI bigrams)",
            corpus_seqs=corpus_seqs, query_seqs=query_seqs,
            ground_truth=ground_truth,
            system=PMIQuest(
                use_pmitd=False,
                use_pmi_bigrams=False,
                hnsw_k=DEFAULT_HNSW_K,
            ),
        ))
        rows.append(run_variant(
            tag="A2", group="A",
            description="PMI-TF-IDF (τ=0.5, α=1.0)  [proposed]",
            corpus_seqs=corpus_seqs, query_seqs=query_seqs,
            ground_truth=ground_truth,
            system=proposed_system(),
        ))

    # ── Group B: PMI threshold τ ──────────────────────────────────
    if "B" in G:
        print("\n" + "─"*55)
        print("Group B: PMI Threshold τ  (α=1.0 fixed)")
        print("─"*55)

        for tau in [0.0, 0.5, 1.0, 1.5, 2.0]:
            star = "  [proposed]" if tau == DEFAULT_PMI_TAU else ""
            rows.append(run_variant(
                tag=f"B_tau{tau}", group="B",
                description=f"tau={tau}{star}",
                corpus_seqs=corpus_seqs, query_seqs=query_seqs,
                ground_truth=ground_truth,
                system=proposed_system(pmi_tau=tau),
            ))

    # ── Group C: Bigram weight α ──────────────────────────────────
    if "C" in G:
        print("\n" + "─"*55)
        print("Group C: Bigram Weight alpha  (tau=0.5 fixed)")
        print("─"*55)

        for alpha in [ 0.25, 0.50, 0.75, 1.00, 2.00, 5.00, 10.00]:
            star = "  [proposed]" if alpha == DEFAULT_BIGRAM_WEIGHT else ""
            # α=0 is functionally identical to no bigrams — skip bigram
            # fitting to save time and avoid division by zero in bi-TF
            rows.append(run_variant(
                tag=f"C_a{alpha}", group="C",
                description=f"alpha={alpha}{star}",
                corpus_seqs=corpus_seqs, query_seqs=query_seqs,
                ground_truth=ground_truth,
                system=proposed_system(
                    bigram_weight=alpha,
                    use_pmi_bigrams=(alpha > 0.0),
                ),
            ))

    # ── Group D: HNSW candidates K ───────────────────────────────
    if "D" in G:
        print("\n" + "─"*55)
        print("Group D: HNSW Candidate Count K  (tau=0.5, alpha=0.5)")
        print("─"*55)

        for K in [10, 20, 50, 100, 200]:
            star = "  [proposed]" if K == DEFAULT_HNSW_K else ""
            rows.append(run_variant(
                tag=f"D_K{K}", group="D",
                description=f"K={K}{star}",
                corpus_seqs=corpus_seqs, query_seqs=query_seqs,
                ground_truth=ground_truth,
                system=proposed_system(hnsw_k=K),
            ))

    # ── Group E: SW reranking ─────────────────────────────────────
    if "E" in G:
        print("\n" + "─"*55)
        print("Group E: Smith-Waterman Reranking")
        print("─"*55)

        rows.append(run_variant(
            tag="E1", group="E",
            description="HNSW cosine only (no SW reranking)",
            corpus_seqs=corpus_seqs, query_seqs=query_seqs,
            ground_truth=ground_truth,
            system=NoSWPMIQuest(
                use_pmitd=False,
                use_pmi_bigrams=True,
                pmi_tau=DEFAULT_PMI_TAU,
                bigram_weight=DEFAULT_BIGRAM_WEIGHT,
                hnsw_k=DEFAULT_HNSW_K,
            ),
        ))
        rows.append(run_variant(
            tag="E2", group="E",
            description="HNSW + SW reranking  [proposed]",
            corpus_seqs=corpus_seqs, query_seqs=query_seqs,
            ground_truth=ground_truth,
            system=proposed_system(),
        ))

    # ── Group F: PMI-TD merges (pre-computed) ─────────────────────
    if "F" in G:
        print("\n" + "─"*55)
        print("Group F: PMI-TD Merge Rules  (logged from prior run)")
        print("─"*55)

        known_f = [
            ("F1", "no PMI-TD (raw tokens)  [proposed]",
             0,   "1024",      0.5107, 0.7426, 0.1881, 0.1069, 3917.5),
            ("F2", "PMI-TD max_merges=20",
             20,  "1024→1044", 0.5069, 0.7327, 0.1842, 0.1059, 8867.4),
            ("F3", "PMI-TD max_merges=50",
             50,  "1024→1074", 0.4964, 0.7327, 0.1822, 0.1050, 4009.0),
            ("F4", "PMI-TD max_merges=100",
             100, "1024→1124", 0.4850, 0.7228, 0.1743, 0.1050, 9724.6),
        ]
        for tag, desc, merges, vocab, MAP, P1, P5, P10, t in known_f:
            rows.append({
                "group": "F", "tag": tag, "description": desc,
                "pmi_tau": 1.5, "alpha": 0.5,
                "hnsw_k": DEFAULT_HNSW_K,
                "n_bigrams": 37715,
                "MAP": MAP, "P@1": P1, "P@5": P5, "P@10": P10,
                "time_s": t,
            })
            print(f"    [{tag}] merges={merges:3d}  MAP={MAP:.4f}  (pre-computed)")

    # ── save CSV ──────────────────────────────────────────────────
    if rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  Saved -> {args.out}")

    # ── console summary ───────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"  {'tag':<14} {'grp':<4}  {'MAP':>7} {'P@1':>7} {'P@5':>7} {'P@10':>7}")
    print("  " + "-" * 55)
    for r in rows:
        print(f"  {r['tag']:<14} {r['group']:<4}  "
              f"{r['MAP']:>7.4f} {r['P@1']:>7.4f} "
              f"{r['P@5']:>7.4f} {r['P@10']:>7.4f}")
    print("=" * 65)
    print("\nDone.")


if __name__ == "__main__":
    main()