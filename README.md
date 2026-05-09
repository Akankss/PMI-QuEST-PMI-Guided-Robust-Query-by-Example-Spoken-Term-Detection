# PMI-QuEST: PMI-Guided Robust Query-by-Example Spoken Term Detection

> **IEEE TASLP** (under review) · Akanksha Singh · Vipul Arora (IIT Kanpur / KU Leuven) · Yi-Ping Phoebe Chen (La Trobe University)

---

## 🎯 Interactive Demo

**Open `demo/pmiquest_demo.html` in any browser** 

The demo walks through the full pipeline:

| Tab | Content |
|-----|---------|
| ① Pipeline Walkthrough | Step-by-step trace of a single query through tokenise → PMI-TF-IDF vector → HNSW search → SW reranking |
| ② PMI Bigram Discovery | The PMI formula, why frequency (BPE) fails, vocabulary compactness |
| ③ PMI vs BPE | Side-by-side MAP/MRR/P@1 comparison |
| ④ Results | Full baseline table + cross-tokeniser results + ablation bars |
| ⑤ Cross-Lingual | All 12 IndicSUPERB Kathbath languages, per-language MAP |
| ⑥ Algorithm | Pseudocode + complexity + hyperparameter table |

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


## ⚙️ Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| τ (PMI threshold) | 0.5 | Stable across τ ∈ [0, 2] |
| α (bigram weight) | 1.0 | Saturates at α ≥ 1.0 |
| C (HNSW candidates) | 200 | MAP grows up to C=200 |
| SW: m₊ / m₋ / g | +2 / −1 / −2 | Match / mismatch / gap |
| HNSW M | 16 | ef_construction=150, ef_search=200 |
