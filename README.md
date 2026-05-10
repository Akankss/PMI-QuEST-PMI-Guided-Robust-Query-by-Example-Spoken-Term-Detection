# PMI-QuEST: PMI-Guided Robust Query-by-Example Spoken Term Detection

> **IEEE TASLP** (under review) · Akanksha Singh ( IIT Kanpur and La Trobe University) · Yi-Ping Phoebe Chen (La Trobe University) · Vipul Arora (IIT Kanpur / KU Leuven)

---

## 🔴 Live Demo
   👉 [**Interactive pipeline demo**](https://akankss.github.io/PMI-QuEST-PMI-Guided-Robust-Query-by-Example-Spoken-Term-Detection/)


---

## 📋 Abstract

QbE-STD locates spoken queries in untranscribed audio without ASR. We propose **PMI-QuEST**, which augments the unigram TF-IDF pre-filter of H-QuEST with PMI-selected token bigrams, addressing order-blindness without incurring the full O(K²) cost of exhaustive bigram indexing.

- **LibriSpeech test-clean**: MAP = 0.7867, MRR = 0.9809, P@1 = 0.9775 (BEST-STD tokeniser, K=1024)
- **+16.4% relative MAP** over H-QuEST; **+6.6%** over standalone BEST-STD retrieval
- **Zero-shot cross-lingual**: Statistically significant gains on all 12 IndicSUPERB Kathbath languages (paired Wilcoxon, p < 0.05)

---

## 🏗️ Pipeline

```
Speech corpus / query
       │
       ▼
SSL encoder + k-means → token sequence s
       │
       ▼
PMI bigram filtering (τ = 0.5) → 𝒷_τ
       │
       ▼
PMI-TF-IDF vector  v = [u ‖ α·b] / ‖·‖₂
       │
       ▼
HNSW index (offline) → top-C candidates   O(log N)
       │
       ▼
Smith-Waterman reranking → ranked list     O(C·m·n̄)
```

---
## Repository Structure

```
PMI-QuEST/
├── Pmi-QuEST/
│   ├── pmiquest_system.py      # TFIDFBaseline, HQuEST, PMIQuest classes
│   ├── dataloader.py           # CSV + JSON data loaders
│   ├── smith_waterman.py       # Smith-Waterman alignment module
│   ├── hnsw.py                 # HNSW index wrapper
│   ├── tfidf.py                # TF-IDF vectoriser
│   └── bpe.py                  # BPE baseline (for ablation comparison)
├── Tokenizer/
│   └── audio_tokenizer_v2.py   # SSL encoder + k-means tokenisation
├── Experiments/
│   ├── run_ablation.py         # Groups A–E ablation sweep
│   ├── run_multi_tokeniser.py  # Table I: cross-tokeniser evaluation
│   ├── run_layer_sweep.py      # Optimal layer selection per SSL model
│   ├── bpe_abl.py              # PMI vs BPE bigram comparison
│   ├── dtw_final.py            # DTW baseline
│   └── plot_search_times.py    # Figure: per-query search time bar chart
├── Figures/                    # Paper figures (PDFs)
├── configs/                    # Hyperparameter configs per tokeniser
├── data/
│   └── sample/                 # Toy corpus + queries + relevance for smoke test
│       ├── corpus.csv
│       ├── queries.csv
│       └── relevance.json
├── run_retrieval.py            
├── requirements.txt
├── setup.py
└── LICENSE
```

--

## Results

### LibriSpeech test-clean (Q = 200 queries, BEST-STD tokeniser, K = 1024)

| System | MAP | MRR | P@1 | P@5 | P@10 |
|---|---|---|---|---|---|
| DTW | 0.4474 | 0.4790 | 0.4591 | 0.1036 | 0.0559 |
| TF-IDF | 0.5974 | 0.5970 | 0.5891 | 0.1360 | 0.0734 |
| H-QuEST | 0.6760 | 0.8621 | 0.8539 | 0.2215 | 0.1163 |
| BEST-STD | 0.7383 | 0.8720 | 0.8586 | 0.2334 | 0.1260 |
| **PMI-QuEST** | **0.7867** | **0.9809** | **0.9775** | **0.2629** | **0.1427** |

PMI-QuEST achieves **+16.4% relative MAP** over H-QuEST and **+6.6%** over standalone BEST-STD retrieval.

### IndicSUPERB Kathbath — zero-shot cross-lingual (XLS-R-300M, K = 1024)

| Language | TF-IDF MAP | H-QuEST MAP | PMI-QuEST MAP |
|---|---|---|---|
| Bengali | 0.5250 | 0.6399 | **0.7190** † |
| Gujarati | 0.5490 | 0.6349 | **0.7536** † |
| Hindi | 0.5300 | 0.6377 | **0.7547** † |
| Kannada | 0.5203 | 0.6364 | **0.7627** † |
| Malayalam | 0.5299 | 0.6300 | **0.7465** † |
| Marathi | 0.5357 | 0.6364 | **0.7595** † |
| Odia | 0.5351 | 0.6297 | **0.7478** † |
| Punjabi | 0.5352 | 0.6409 | **0.6966** † |
| Sanskrit | 0.5409 | 0.6420 | **0.7601** † |
| Tamil | 0.5353 | 0.6367 | **0.7503** † |
| Telugu | 0.5299 | 0.6320 | **0.7487** † |
| Urdu | 0.5368 | 0.6391 | **0.7697** † |

† Statistically significant over H-QuEST (paired Wilcoxon, p < 0.05, N = 40).

### Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| τ (PMI threshold) | 0.5 | Stable across τ ∈ [0, 2] |
| α (bigram weight) | 1.0 | Saturates at α ≥ 1.0 |
| C (HNSW candidates) | 200 | MAP grows up to C = 200 |
| SW: m₊ / m₋ / g | +2 / −1 / −2 | Match / mismatch / gap |
| HNSW M | 16 | ef_construction = 150, ef_search = 200 |

---

## Installation

```bash
git clone https://github.com/Akankss/PMI-QuEST-PMI-Guided-Robust-Query-by-Example-Spoken-Term-Detection.git
cd PMI-QuEST-PMI-Guided-Robust-Query-by-Example-Spoken-Term-Detection

pip install -r requirements.txt
```
```bash
pip install hnswlib
```

For GPU-accelerated tokenisation (optional, needed only for `audio_tokenizer_v2.py`):

```bash
pip install torch torchaudio transformers  # follow pytorch.org for CUDA version
```

---

## Quick Start

### 1. Verify installation on sample data 

```bash
python run_retrieval.py \
    --corpus    data/sample/corpus.csv \
    --queries   data/sample/queries.csv \
    --relevance data/sample/relevance.json \
    --system    all
```

Expected output:

```
  TF-IDF        MAP=...  MRR=...  P@1=...
  H-QuEST       MAP=...  MRR=...  P@1=...
  PMI-QuEST     MAP=...  MRR=...  P@1=...
```

### 2. Run on your own tokenised data

```bash
python run_retrieval.py \
    --corpus    /path/to/corpus.csv \
    --queries   /path/to/queries.csv \
    --relevance /path/to/relevance.json \
    --system    pmiquest \
    --out       results/run.csv
```

### 3. Tokenise raw audio first (optional)

```bash
python Tokenizer/audio_tokenizer_v2.py \
    --audio_dir  /path/to/wavs \
    --model      wavlm-base \
    --layer      9 \
    --K          1024 \
    --out_dir    tokenised/wavlm_l9_k1024
```

### 4. Run ablation study (reproduces paper Figure 4)

```bash
python Experiments/run_ablation.py \
    --corpus    corpus.csv \
    --queries   queries.csv \
    --relevance relevance.json \
    --out       results/ablation.csv \
    --groups    ABCDE
```

---

## Data Format

### corpus.csv / queries.csv

```csv
filename,tokens
1089-134686-0000.wav,42 17 83 5 61 23 ...
1089-134686-0001.wav,9 55 12 77 34 ...
```

- `filename`: utterance name (extension stripped internally)
- `tokens`: space- or comma-separated integer token IDs

### relevance.json

```json
{
  "THE_word_00.wav": {
    "relevant": ["1089-134686-0000.wav", "1320-122612-0003.wav"]
  },
  "THE_word_01.wav": {
    "relevant": ["2961-960-0000.wav"]
  }
}
```

---


## Citation

If you use PMI-QuEST in your research, please cite:

```bibtex
@article{singh2026pmiquest,
  title     = {{PMI-QuEST}: {PMI}-Guided Robust Query-by-Example Spoken Term Detection},
  author    = {Singh, Akanksha and Chen, Yi-Ping Phoebe and Arora, Vipul},
  journal   = {IEEE Transactions on Audio, Speech, and Language Processing},
  year      = {2026},
  note      = {Under review}
}
```

Related works this builds on:

```bibtex
@inproceedings{singh2025hquest,
  title     = {{H-QuEST}: {HNSW}-Accelerated Query-by-Example Spoken Term Detection},
  author    = {Singh, Akanksha and Chen, Yi-Ping Phoebe and Arora, Vipul},
  booktitle = {Proc. Interspeech},
  year      = {2025}
}

@inproceedings{singh2024efficient,
  title     = {Efficient Query-by-Example Spoken Term Detection with Discrete Tokens},
  author    = {Singh, Akanksha and Arora, Vipul},
  booktitle = {Proc. Interspeech},
  year      = {2024}
}

@inproceedings{singh2025best,
  title     = {{BEST-STD}: Bidirectional Mamba Tokeniser for Spoken Term Detection},
  author    = {Singh, Akanksha and Chen, Yi-Ping Phoebe and Arora, Vipul},
  booktitle = {Proc. ICASSP},
  year      = {2025}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

