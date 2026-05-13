"""
Multi-Tokeniser PMI-QuEST Comparison
======================================
Runs TF-IDF, Token-DTW, H-QuEST, and PMI-QuEST across all tokeniser CSVs
in a directory, producing a single comparison table.

Token-DTW baseline
------------------
Follows Hazen et al. (2009) "Query-by-Example Spoken Term Detection using
Phonetic Posteriorgram Templates":
  - Binary substitution cost:  c(i,j) = 0 if q[i]==d[j] else 1
  - Standard unconstrained DTW (no Sakoe-Chiba window)
  - Normalized score:          score(q,d) = DTW(q,d) / (|q| + |d|)
  - Ranking: ascending by score (lower = better match)

Complexity note: O(N · |q| · |d|) per query. For large corpora use
--dtw_max_docs to cap the number of corpus documents compared per query
(random subset drawn once per config); default = full corpus.

Usage
-----
python run_multi_tokeniser.py \\
    --tokeniser_dir  tokenised/ \\
    --relevance      relevance.json \\
    --out            results/multi_tokeniser.csv

python run_multi_tokeniser.py \\
    --configs \\
        "wav2vec2-base l6 k100:tokenised/wav2vec2-base_l6_k100/corpus.csv:tokenised/wav2vec2-base_l6_k100/queries.csv" \\
        "hubert-base l6 k100:tokenised/hubert-base_l6_k100/corpus.csv:tokenised/hubert-base_l6_k100/queries.csv" \\
        "wavlm-base l7 k512:tokenised/wavlm-base_l7_k512/corpus.csv:tokenised/wavlm-base_l7_k512/queries.csv" \\
    --relevance relevance.json \\
    --out results/multi_tokeniser.csv

"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

csv.field_size_limit(sys.maxsize)

try:
    from pmiquest_system import TFIDFBaseline, HQuEST, PMIQuest, evaluate
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from pmiquest_system import TFIDFBaseline, HQuEST, PMIQuest, evaluate


# ===========================================================================

class TokenDTW:
    """
    Hazen et al. (2009) style DTW over discrete token sequences.

    Cost function  : binary substitution  c(i,j) = 0 if a==b else 1
    DTW variant    : standard unconstrained DP (no Sakoe-Chiba window)
    Score          : DTW(q,d) / (|q| + |d|)   ← length-normalised distance
    Ranking        : ascending (lower score = better match)

    Parameters
    ----------
    max_docs : int or None
        If set, compare the query against a random subset of this many corpus
        documents (sampled once in fit()). Useful for large corpora where
        brute-force DTW is too slow. None = full corpus.
    seed : int
        RNG seed for reproducible subsampling.
    """

    def __init__(self, max_docs: Optional[int] = None, seed: int = 42):
        self.max_docs  = max_docs
        self.seed      = seed
        self._corpus:  List[np.ndarray] = []   # token arrays
        self._indices: List[int]        = []   # corpus positions exposed to rank()

    # ------------------------------------------------------------------
    def fit(self, corpus: List[List[int]]) -> "TokenDTW":
        """Store corpus (as numpy arrays for speed)."""
        self._corpus_full = [np.asarray(s, dtype=np.int32) for s in corpus]
        N = len(self._corpus_full)

        if self.max_docs is not None and self.max_docs < N:
            rng = np.random.default_rng(self.seed)
            self._indices = sorted(rng.choice(N, size=self.max_docs, replace=False).tolist())
        else:
            self._indices = list(range(N))

        self._corpus = [self._corpus_full[i] for i in self._indices]
        return self

    # ------------------------------------------------------------------
    @staticmethod
    def _dtw_binary(q: np.ndarray, d: np.ndarray) -> float:
        """
        Unconstrained DTW with binary cost.

        Uses numpy row-by-row DP to avoid a Python loop over every cell.
        Time  O(|q| · |d|),  Memory O(|d|)  (two-row rolling buffer).
        """
        nq, nd = len(q), len(d)
        INF = 1e18

        # cost row for current query step (initialised to infinity)
        prev = np.full(nd, INF, dtype=np.float64)
        curr = np.empty(nd, dtype=np.float64)

        for i in range(nq):
            # binary substitution cost for row i: shape (nd,)
            c = (q[i] != d).astype(np.float64)   # 0 where match, 1 otherwise

            # DTW recurrence:  D[i,j] = c[i,j] + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
            # We unroll column 0 separately (no left or diagonal neighbour):
            if i == 0:
                curr[0] = c[0]
                for j in range(1, nd):
                    curr[j] = c[j] + curr[j - 1]   # only left neighbour
            else:
                # column 0: only top neighbour
                curr[0] = c[0] + prev[0]
                # columns 1..nd-1: vectorised minimum of top, top-left, left
                # top      = prev[1:]
                # top-left = prev[:-1]
                # left     = curr[j-1]  ← must stay serial (data dependency)
                for j in range(1, nd):
                    curr[j] = c[j] + min(prev[j], prev[j - 1], curr[j - 1])

            prev, curr = curr, prev   # swap buffers

        return float(prev[nd - 1])

    # ------------------------------------------------------------------
    def rank(self, query: List[int]) -> List[int]:
        """
        Return corpus indices (into the *original* corpus passed to fit())
        sorted by ascending normalised DTW distance.

        Documents not in the active subset (when max_docs is set) are
        appended at the end in their original order with score=inf,
        so the returned list always covers all N corpus positions —
        matching the contract expected by run_one_config().
        """
        q = np.asarray(query, dtype=np.int32)
        nq = len(q)

        scores: Dict[int, float] = {}
        for orig_idx, d in zip(self._indices, self._corpus):
            nd = len(d)
            if nd == 0 or nq == 0:
                scores[orig_idx] = 1.0   # maximum normalised cost
                continue
            dtw_val = self._dtw_binary(q, d)
            scores[orig_idx] = dtw_val / (nq + nd)

        # Positions not in the subset get score=inf (ranked last)
        N_full = len(self._corpus_full)
        ranked_subset  = sorted(self._indices, key=lambda i: scores[i])
        ranked_missing = [i for i in range(N_full) if i not in scores]

        return ranked_subset + ranked_missing




def load_csv(csv_path: str) -> Dict[str, List[int]]:
    """
    Load a Filename,Data CSV produced by audio_tokenizer_v2.py.
    Returns {utt_id: [tok, ...]}  where utt_id = Path(filename).stem.
    """
    # Ensure the field-size limit is set before reading
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = max_int // 10

    tokens: Dict[str, List[int]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])

        fname_col = next(
            (c for c in cols if c.strip().lower() in
             ("filename", "file", "name", "id", "utt_id", "query_id")),
            cols[0],
        )
        data_col = next(
            (c for c in cols if c.strip().lower() in
             ("data", "tokens", "token", "sequence")),
            cols[1] if len(cols) > 1 else cols[0],
        )

        for row in reader:
            fname  = row[fname_col].strip()
            utt_id = Path(fname).stem
            raw    = row[data_col].strip()

            if "," in raw:
                toks = [int(t) for t in raw.split(",")
                        if t.strip().lstrip("-").isdigit()]
            else:
                toks = [int(t) for t in raw.split()
                        if t.strip().lstrip("-").isdigit()]

            if toks:
                tokens[utt_id] = toks

    if not tokens:
        raise ValueError(f"No tokens loaded from {csv_path} — columns: {cols}")

    all_lens = [len(v) for v in tokens.values()]
    print(f"  {len(tokens):,} seqs  "
          f"len={min(all_lens)}–{max(all_lens)}  "
          f"mean={np.mean(all_lens):.1f}   [{csv_path}]")
    return tokens


def load_relevance(path: str) -> Dict[str, List[str]]:
    with open(path) as f:
        raw = json.load(f)
    result = {}
    for key, val in raw.items():
        query_stem = Path(key).stem                          # "THE_4446-2275-0000"
        rel_stems  = [Path(r).stem for r in val.get("relevant", [])]
        result[query_stem] = rel_stems
    return result



def _eval(ranked_lists: Dict[str, List[str]],
          relevance:    Dict[str, List[str]]) -> Dict[str, float]:
    """
    ranked_lists : {query_id: [doc_id, ...]}   full ranked list of string IDs
    relevance    : {query_id: [relevant_doc_ids]}
    Returns MAP, MRR, P@1, P@5, P@10.
    """
    aps, rrs, p1s, p5s, p10s = [], [], [], [], []
    for qid, ranked in ranked_lists.items():
        if qid not in relevance:
            continue
        rel = set(relevance[qid])

        n_rel, ap, rr = 0, 0.0, 0.0
        for rank, did in enumerate(ranked, 1):
            if did in rel:
                n_rel += 1
                ap += n_rel / rank
                if rr == 0.0:
                    rr = 1.0 / rank
        aps.append(ap / max(len(rel), 1))
        rrs.append(rr)

        p1s.append(1.0 if ranked and ranked[0] in rel else 0.0)
        p5s.append(sum(1 for d in ranked[:5]  if d in rel) / 5)
        p10s.append(sum(1 for d in ranked[:10] if d in rel) / 10)

    return {
        "map": float(np.mean(aps)),
        "mrr": float(np.mean(rrs)),
        "p1":  float(np.mean(p1s)),
        "p5":  float(np.mean(p5s)),
        "p10": float(np.mean(p10s)),
    }



def run_one_config(
    label:        str,
    corpus:       Dict[str, List[int]],
    queries:      Dict[str, List[int]],
    relevance:    Dict[str, List[str]],
    dtw_max_docs: Optional[int] = None,
    verbose:      bool = True,
) -> dict:
    """
    Run TF-IDF, Token-DTW, H-QuEST, and PMI-QuEST on one tokeniser config.

    Result keys
    -----------
    label, vocab_size, mean_query_len, mean_corpus_len, rho
    tfidf_{map,mrr,p1,p5,p10}
    dtw_{map,mrr,p1,p5,p10}
    hquest_{map,mrr,p1,p5,p10}
    pmi_{map,mrr,p1,p5,p10}
    pmi_{map,mrr,p1}_gain_vs_hquest
    """
    corpus_list = list(corpus.values())
    query_list  = list(queries.values())
    query_ids   = list(queries.keys())
    corpus_ids  = list(corpus.keys())        # position i → doc_id string

    vocab_size = len({t for s in corpus_list for t in s})
    mean_q_len = float(np.mean([len(s) for s in query_list]))
    mean_c_len = float(np.mean([len(s) for s in corpus_list]))
    rho        = mean_q_len / mean_c_len

    if verbose:
        print(f"\n{'─'*65}")
        print(f"Config : {label}")
        print(f"  Corpus={len(corpus_list)}  Queries={len(query_ids)}")
        print(f"  V={vocab_size}  mean_q={mean_q_len:.1f}  "
              f"mean_c={mean_c_len:.1f}  rho={rho:.3f}")
        if dtw_max_docs:
            print(f"  DTW cap: {dtw_max_docs} docs/query")

    results = {
        "label": label, "vocab_size": vocab_size,
        "mean_query_len": mean_q_len, "mean_corpus_len": mean_c_len,
        "rho": rho,
    }

    def _ranked_dict(model):
        """Build {qid: [doc_id, ...]} from a fitted model with .rank()."""
        out = {}
        for qid in query_ids:
            if qid not in relevance:
                continue
            ranked_idx = model.rank(queries[qid])
            out[qid] = [corpus_ids[i] for i in ranked_idx
                        if i < len(corpus_ids)]
        return out

    # ── TF-IDF Baseline ──────────────────────────────────────────────────────
    t0 = time.time()
    baseline = TFIDFBaseline()
    baseline.fit(corpus_list)
    m = _eval(_ranked_dict(baseline), relevance)
    results.update({
        "tfidf_map": m["map"], "tfidf_mrr": m["mrr"],
        "tfidf_p1":  m["p1"],  "tfidf_p5":  m["p5"], "tfidf_p10": m["p10"],
    })
    if verbose:
        print(f"  TF-IDF    : MAP={m['map']:.4f}  MRR={m['mrr']:.4f}  "
              f"P@1={m['p1']:.4f}  [{time.time()-t0:.1f}s]")

    # ── Token-DTW Baseline (Hazen et al. 2009) ────────────────────────────────
    t0 = time.time()
    dtw_model = TokenDTW(max_docs=dtw_max_docs)
    dtw_model.fit(corpus_list)
    m = _eval(_ranked_dict(dtw_model), relevance)
    results.update({
        "dtw_map": m["map"], "dtw_mrr": m["mrr"],
        "dtw_p1":  m["p1"],  "dtw_p5":  m["p5"], "dtw_p10": m["p10"],
    })
    if verbose:
        cap_note = (f" [capped @ {dtw_max_docs}]" if dtw_max_docs else "")
        print(f"  Token-DTW : MAP={m['map']:.4f}  MRR={m['mrr']:.4f}  "
              f"P@1={m['p1']:.4f}  [{time.time()-t0:.1f}s]{cap_note}")

    # ── H-QuEST ───────────────────────────────────────────────────────────────
    t0 = time.time()
    hquest = HQuEST(hnsw_k=50)
    hquest.fit(corpus_list)
    m = _eval(_ranked_dict(hquest), relevance)
    results.update({
        "hquest_map": m["map"], "hquest_mrr": m["mrr"],
        "hquest_p1":  m["p1"],  "hquest_p5":  m["p5"], "hquest_p10": m["p10"],
    })
    if verbose:
        print(f"  H-QuEST   : MAP={m['map']:.4f}  MRR={m['mrr']:.4f}  "
              f"P@1={m['p1']:.4f}  [{time.time()-t0:.1f}s]")

    # ── PMI-QuEST (best config: τ=0.5, α=1) ────────────────────
    t0 = time.time()
    pmiquest = PMIQuest(
        pmi_tau       = 0.5,
        bigram_weight = 1,
        use_pmitd     = False,
        hnsw_k        = 200,
    )
    pmiquest.fit(corpus_list)
    m = _eval(_ranked_dict(pmiquest), relevance)
    results.update({
        "pmi_map": m["map"], "pmi_mrr": m["mrr"],
        "pmi_p1":  m["p1"],  "pmi_p5":  m["p5"], "pmi_p10": m["p10"],
    })
    if verbose:
        print(f"  PMI-QuEST : MAP={m['map']:.4f}  MRR={m['mrr']:.4f}  "
              f"P@1={m['p1']:.4f}  [{time.time()-t0:.1f}s]")

    # ── Gains: PMI-QuEST vs H-QuEST ──────────────────────────────────────────
    for metric in ("map", "mrr", "p1"):
        hq_val = results[f"hquest_{metric}"]
        pm_val = results[f"pmi_{metric}"]
        results[f"pmi_{metric}_gain_vs_hquest"] = \
            (pm_val - hq_val) / max(hq_val, 1e-9)

    return results



def print_table(all_results: list):
    W = 95
    hdr = (f"{'Tokeniser':<28} {'V':>5}  {'rho':>5}  "
           f"{'MAP':>6}  {'MRR':>6}  {'P@1':>6}  {'P@5':>6}  {'P@10':>6}")

    def _row(r, prefix):
        return (f"  {r['label']:<26} {r['vocab_size']:>5}  {r['rho']:>5.3f}"
                f"  {r[prefix+'_map']:>6.4f}  {r[prefix+'_mrr']:>6.4f}"
                f"  {r[prefix+'_p1']:>6.4f}  {r[prefix+'_p5']:>6.4f}"
                f"  {r[prefix+'_p10']:>6.4f}")

    print(f"\n{'='*W}")
    print("MULTI-TOKENISER COMPARISON")
    print(f"{'='*W}")

    for section, prefix, title in [
        ("tfidf",  "tfidf",  "TF-IDF Baseline"),
        ("dtw",    "dtw",    "Token-DTW Baseline  (Hazen et al. 2009 — binary cost, unconstrained)"),
        ("hquest", "hquest", "H-QuEST"),
    ]:
        print(f"\n── {title} " + "─" * (W - len(title) - 4))
        print(hdr)
        print("─" * W)
        for r in all_results:
            print(_row(r, prefix))

    gain_hdr = hdr + f"  {'ΔMAP':>8}  {'ΔMRR':>8}  {'ΔP@1':>8}"
    print(f"\n── PMI-QuEST  (τ=0.5, α=0.5, no PMI-TD) " + "─" * (W - 42))
    print(gain_hdr)
    print("─" * W)
    for r in all_results:
        gmap = r["pmi_map_gain_vs_hquest"] * 100
        gmrr = r["pmi_mrr_gain_vs_hquest"] * 100
        gp1  = r["pmi_p1_gain_vs_hquest"]  * 100
        print(_row(r, "pmi") +
              f"  {gmap:>+7.1f}%  {gmrr:>+7.1f}%  {gp1:>+7.1f}%")
    print("─" * W)



def save_csv(all_results: list, path: str):
    fields = [
        "label", "vocab_size", "rho",
        "tfidf_map", "tfidf_mrr", "tfidf_p1", "tfidf_p5", "tfidf_p10",
        "dtw_map",   "dtw_mrr",   "dtw_p1",   "dtw_p5",   "dtw_p10",
        "hquest_map","hquest_mrr","hquest_p1","hquest_p5","hquest_p10",
        "pmi_map",   "pmi_mrr",   "pmi_p1",   "pmi_p5",   "pmi_p10",
        "pmi_map_gain_vs_hquest", "pmi_mrr_gain_vs_hquest",
        "pmi_p1_gain_vs_hquest",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)
    print(f"\n  Saved → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run TF-IDF / Token-DTW / H-QuEST / PMI-QuEST "
                    "across multiple tokeniser configurations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tokeniser_dir", default=None,
        help="Directory with subdirectories each containing corpus.csv + queries.csv.",
    )
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help="Explicit configs: 'label:corpus.csv:queries.csv'.",
    )
    parser.add_argument(
        "--relevance", required=True,
        help="Path to relevance.json.",
    )
    parser.add_argument(
        "--out", default="results/multi_tokeniser.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--dtw_max_docs", type=int, default=None,
        help="Cap DTW to N corpus docs per query (random subset). "
             "Default: full corpus. Recommended for >10k docs.",
    )
    args = parser.parse_args()

    relevance = load_relevance(args.relevance)

    # Collect configs
    run_configs: List[tuple] = []

    if args.tokeniser_dir:
        base = Path(args.tokeniser_dir)
        for subdir in sorted(base.iterdir()):
            c = subdir / "corpus.csv"
            q = subdir / "queries.csv"
            if c.exists() and q.exists():
                run_configs.append((subdir.name, str(c), str(q)))
        if not run_configs:
            raise FileNotFoundError(
                f"No subdirs with corpus.csv + queries.csv in {args.tokeniser_dir}"
            )

    elif args.configs:
        for spec in args.configs:
            parts = spec.split(":")
            if len(parts) != 3:
                raise ValueError(
                    f"Expected 'label:corpus_csv:queries_csv', got '{spec}'"
                )
            run_configs.append(tuple(parts))

    else:
        parser.error("Provide --tokeniser_dir or --configs.")

    print(f"\nRunning {len(run_configs)} tokeniser config(s) …")

    all_results = []
    for label, corpus_csv, query_csv in run_configs:
        corpus  = load_csv(corpus_csv)
        queries = load_csv(query_csv)
        result  = run_one_config(
            label, corpus, queries, relevance,
            dtw_max_docs=args.dtw_max_docs,
            verbose=True,
        )
        all_results.append(result)

    print_table(all_results)
    save_csv(all_results, args.out)
