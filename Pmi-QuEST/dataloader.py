"""
data_loader.py — CSV + relevance loader for BPE-MNG H-QuEST

Public API
----------
load_corpus(csv_path)   -> (filenames: List[str], sequences: List[List[int]])
load_queries(csv_path)  -> (filenames: List[str], sequences: List[List[int]])
load_relevance(path)    -> Dict[str, List[str]]   # query_stem -> [corpus_stems]
"""

import sys
import json
import _csv as _csv_module
import csv as _csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _raise_field_size_limit() -> None:
    """Set csv.field_size_limit to the largest value the platform supports."""
    max_int = sys.maxsize
    while True:
        try:
            _csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = max_int // 10


def _parse_token_string(raw: str) -> List[int]:
    """
    Parse a token string in either comma-separated ("1,2,3") or
    space-separated ("1 2 3") format.  Non-numeric tokens are silently
    skipped so that stray headers / labels don't crash the loader.
    """
    parts = raw.split(",") if "," in raw else raw.split()
    return [int(t.strip()) for t in parts if t.strip().lstrip("-").isdigit()]


def _detect_columns(fieldnames: List[str]) -> Tuple[str, str]:
    """
    Return (filename_column, data_column) by matching common header names
    case-insensitively.  Falls back to positional (col0, col1).
    """
    cols = list(fieldnames)
    lower = [c.strip().lower() for c in cols]

    fname_candidates = {"filename", "file", "name", "id", "utt_id", "query_id"}
    data_candidates  = {"data", "tokens", "token", "sequence"}

    fname_col = next((cols[i] for i, c in enumerate(lower) if c in fname_candidates),
                     cols[0])
    data_col  = next((cols[i] for i, c in enumerate(lower) if c in data_candidates),
                     cols[1] if len(cols) > 1 else cols[0])

    return fname_col, data_col


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_csv(csv_path: str) -> Dict[str, List[int]]:
    """
    Load a token CSV and return a dict mapping utterance stem -> token list.

    Accepted CSV formats
    --------------------
    filename,tokens
    THE_4446-2275-0000.wav,"1 2 3 4 5"
    THE_4446-2275-0000.wav,"1,2,3,4,5"

    The filename column header may be any of:
        filename | file | name | id | utt_id | query_id  (case-insensitive)
    The token column header may be any of:
        data | tokens | token | sequence                 (case-insensitive)

    Returns
    -------
    Dict[stem, List[int]]
        Keys are Path(filename).stem (extension stripped).
    """
    _raise_field_size_limit()

    tokens: Dict[str, List[int]] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {csv_path}")

        fname_col, data_col = _detect_columns(list(reader.fieldnames))

        for row in reader:
            fname  = row[fname_col].strip()
            utt_id = Path(fname).stem          # drop extension
            raw    = row[data_col].strip()
            toks   = _parse_token_string(raw)
            if toks:
                tokens[utt_id] = toks

    if not tokens:
        cols = list(reader.fieldnames) if reader.fieldnames else []
        raise ValueError(
            f"No tokens loaded from {csv_path!r} — detected columns: {cols}\n"
            f"  filename column used: {fname_col!r}\n"
            f"  data column used:     {data_col!r}"
        )

    all_lens = [len(v) for v in tokens.values()]
    print(
        f"  {len(tokens):,} seqs  "
        f"len={min(all_lens)}–{max(all_lens)}  "
        f"mean={np.mean(all_lens):.1f}   "
        f"[{csv_path}]"
    )
    return tokens


# ---------------------------------------------------------------------------
# Public corpus / query loaders
# ---------------------------------------------------------------------------

def _dict_to_parallel(token_dict: Dict[str, List[int]]) -> Tuple[List[str], List[List[int]]]:
    """
    Convert {stem: tokens} dict to two parallel lists (filenames, sequences).
    Filenames are returned as "<stem>.wav" to match the convention used by
    infer_ground_truth / load_relevance (which strip extensions anyway).
    """
    filenames  = [f"{stem}.wav" for stem in token_dict]
    sequences  = list(token_dict.values())
    return filenames, sequences


def load_corpus(csv_path: str) -> Tuple[List[str], List[List[int]]]:
    """
    Load corpus CSV.

    Returns
    -------
    filenames : List[str]   — e.g. ["THE_4446-2275-0000.wav", ...]
    sequences : List[List[int]]
    """
    token_dict = load_csv(csv_path)
    return _dict_to_parallel(token_dict)


def load_queries(csv_path: str) -> Tuple[List[str], List[List[int]]]:
    """
    Load query CSV.

    Returns
    -------
    filenames : List[str]
    sequences : List[List[int]]
    """
    token_dict = load_csv(csv_path)
    return _dict_to_parallel(token_dict)


# ---------------------------------------------------------------------------
# Relevance loader
# ---------------------------------------------------------------------------

def load_relevance(path: str) -> Dict[str, List[str]]:
    """
    Load a relevance JSON file.

    Expected format
    ---------------
    {
        "THE_4446-2275-0000.wav": {
            "relevant": ["4446-2275-0000.flac", ...]
        },
        ...
    }

    Returns
    -------
    Dict[query_stem, List[corpus_stem]]
        Both keys and values have file extensions stripped so they can be
        matched against the stems produced by load_corpus / load_queries.

    Example
    -------
    "THE_4446-2275-0000.wav"  →  query_stem  = "THE_4446-2275-0000"
    "4446-2275-0000.flac"     →  corpus_stem = "4446-2275-0000"
    """
    with open(path, encoding="utf-8") as f:
        raw: dict = json.load(f)

    result: Dict[str, List[str]] = {}
    for key, val in raw.items():
        query_stem = Path(key).stem
        rel_stems  = [Path(r).stem for r in val.get("relevant", [])]
        result[query_stem] = rel_stems

    n_with_rel = sum(1 for v in result.values() if v)
    print(
        f"  {len(result):,} queries in relevance file  "
        f"({n_with_rel:,} have ≥1 relevant document)   [{path}]"
    )
    return result