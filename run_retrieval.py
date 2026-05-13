"""
run_retrieval.py
================
Command-line entry point for PMI-QuEST, H-QuEST, and TF-IDF retrieval.

Usage
-----
# Run PMI-QuEST (proposed, paper defaults)
python run_retrieval.py \\
    --corpus    data/sample/corpus.csv \\
    --queries   data/sample/queries.csv \\
    --relevance data/sample/relevance.json \\
    --system    pmiquest

# Compare all three systems
python run_retrieval.py \\
    --corpus    data/sample/corpus.csv \\
    --queries   data/sample/queries.csv \\
    --relevance data/sample/relevance.json \\
    --system    all

# Override paper hyperparameters
python run_retrieval.py \\
    --corpus    corpus.csv \\
    --queries   queries.csv \\
    --relevance relevance.json \\
    --system    pmiquest \\
    --tau       0.5 \\
    --alpha     1.0 \\
    --candidates 200 \\
    --out       results/my_run.csv

Systems
-------
  tfidf    : Unigram TF-IDF + brute-force cosine  (Singh et al. 2024)
  hquest   : TF-IDF + HNSW(C=200) + SW reranking  (Singh et al. 2025)
  pmiquest : PMI-TF-IDF + HNSW(C=200) + SW        (this paper)
  all      : Run all three and print comparison table

Output
------
Prints MAP, MRR, P@1, P@5, P@10 to stdout.
Optionally writes a CSV row to --out.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

_HERE = Path(__file__).resolve().parent
for _candidate in [_HERE, _HERE / "Pmi-QuEST"]:
    if (_candidate / "pmiquest_system.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from dataloader import load_corpus, load_queries, load_relevance
from pmiquest_system import TFIDFBaseline, HQuEST, PMIQuest, evaluate




def build_ground_truth(
    query_filenames:  List[str],
    corpus_filenames: List[str],
    relevance:        Dict[str, List[str]],
) -> List[Set[int]]:
    """Map query filenames to sets of relevant corpus indices."""
    corpus_stem_map = {Path(cf).stem: i for i, cf in enumerate(corpus_filenames)}

    # Build a suffix-based fallback index (handles THE_word_00 -> word_00 mismatch)
    suffix_to_key: Dict[str, str] = {}
    for k in relevance:
        suffix_to_key[k.split("_")[-1]] = k

    gt = []
    for qf in query_filenames:
        q_stem  = Path(qf).stem
        rel_raw = relevance.get(q_stem)
        if rel_raw is None:
            rel_raw = relevance.get(suffix_to_key.get(q_stem.split("_")[-1], ""), [])
        rel_idxs: Set[int] = set()
        for r in rel_raw:
            stem = Path(r).stem
            if stem in corpus_stem_map:
                rel_idxs.add(corpus_stem_map[stem])
        gt.append(rel_idxs)
    return gt



def run_system(name: str, system, corpus_seqs, query_seqs, ground_truth):
    t0 = time.time()
    system.fit(corpus_seqs)
    ranked  = [system.rank(q) for q in query_seqs]
    metrics = evaluate(ranked, ground_truth)
    metrics["time_s"] = round(time.time() - t0, 2)
    metrics["system"] = name
    return metrics


def _fmt(m: Dict) -> str:
    return (
        f"MAP={m['MAP']:.4f}  MRR={m['MRR']:.4f}  "
        f"P@1={m['P@1']:.4f}  P@5={m['P@5']:.4f}  P@10={m['P@10']:.4f}"
        f"  [{m['time_s']:.1f}s]"
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="PMI-QuEST retrieval system runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    p.add_argument("--corpus",    required=True,
                   help="Path to corpus token CSV  (filename, tokens columns)")
    p.add_argument("--queries",   required=True,
                   help="Path to queries token CSV  (filename, tokens columns)")
    p.add_argument("--relevance", required=True,
                   help="Path to relevance JSON  ({query.wav: {relevant: [...]}})")

    # System selection
    p.add_argument("--system", default="pmiquest",
                   choices=["tfidf", "hquest", "pmiquest", "all"],
                   help="Which system to run")

    # PMI-QuEST hyperparameters (paper defaults)
    p.add_argument("--tau",        type=float, default=0.5,
                   help="PMI threshold tau  (paper: 0.5)")
    p.add_argument("--alpha",      type=float, default=1.0,
                   help="Bigram weight alpha  (paper: 1.0)")
    p.add_argument("--candidates", type=int,   default=200,
                   help="HNSW candidate count C  (paper: 200)")

    # SW parameters
    p.add_argument("--sw_match",    type=float, default=2.0)
    p.add_argument("--sw_mismatch", type=float, default=-1.0)
    p.add_argument("--sw_gap",      type=float, default=-2.0)

    # Output
    p.add_argument("--out", default=None,
                   help="Optional CSV path to append result row")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-step progress output")

    return p.parse_args()


def main():
    args = parse_args()


    if not args.quiet:
        print("Loading data ...")
    corpus_filenames, corpus_seqs = load_corpus(args.corpus)
    query_filenames,  query_seqs  = load_queries(args.queries)
    relevance = load_relevance(args.relevance)
    ground_truth = build_ground_truth(
        query_filenames, corpus_filenames, relevance
    )

    n_with_rel = sum(1 for g in ground_truth if g)
    if not args.quiet:
        print(
            f"  Corpus: {len(corpus_seqs)} utterances  |  "
            f"Queries: {len(query_seqs)} ({n_with_rel} with relevant docs)\n"
        )


    sw_kw = dict(
        sw_match=args.sw_match,
        sw_mismatch=args.sw_mismatch,
        sw_gap=args.sw_gap,
    )

    systems_to_run = []
    if args.system in ("tfidf", "all"):
        systems_to_run.append(("TF-IDF",    TFIDFBaseline()))
    if args.system in ("hquest", "all"):
        systems_to_run.append(("H-QuEST",   HQuEST(C=args.candidates, **sw_kw)))
    if args.system in ("pmiquest", "all"):
        systems_to_run.append(("PMI-QuEST", PMIQuest(
            tau_pmi=args.tau,
            alpha=args.alpha,
            C=args.candidates,
            **sw_kw,
        )))


    results = []
    print("=" * 68)
    for name, system in systems_to_run:
        if not args.quiet:
            print(f"Running {name} ...")
        m = run_system(name, system, corpus_seqs, query_seqs, ground_truth)
        results.append(m)
        print(f"  {name:<12}  {_fmt(m)}")
    print("=" * 68)


    if len(results) > 1:
        print()
        print(f"  {'System':<14} {'MAP':>7} {'MRR':>7} {'P@1':>7} {'P@5':>7} {'P@10':>7}")
        print("  " + "-" * 52)
        for m in results:
            print(
                f"  {m['system']:<14} "
                f"{m['MAP']:>7.4f} {m['MRR']:>7.4f} "
                f"{m['P@1']:>7.4f} {m['P@5']:>7.4f} {m['P@10']:>7.4f}"
            )
        print()


    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        write_header = not Path(args.out).exists()
        fields = ["system", "tau", "alpha", "candidates",
                  "MAP", "MRR", "P@1", "P@5", "P@10", "time_s",
                  "corpus", "queries"]
        with open(args.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if write_header:
                w.writeheader()
            for m in results:
                m.update(dict(
                    tau=args.tau, alpha=args.alpha,
                    candidates=args.candidates,
                    corpus=args.corpus, queries=args.queries,
                ))
                w.writerow(m)
        print(f"  Results appended -> {args.out}")


if __name__ == "__main__":
    main()
