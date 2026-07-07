# Prompt Injection Firewall

> ⚠️ **Work in progress.** This repo is an active research sandbox for detecting and
> categorizing prompt-injection attacks against LLM applications. APIs, scripts, and
> results are still changing.

A firewall/classifier that flags **prompt-injection and jailbreak attempts** before they
reach an LLM. The project benchmarks several detection approaches — from a lightweight
bag-of-words classifier to a fine-tuned transformer — and uses unsupervised clustering to
build a taxonomy of attack techniques.

## Overview

The work is split into two threads:

1. **Detection** — binary classification of a prompt as *benign* (`0`) or *malicious* (`1`).
   Two families are compared:
   - **Multinomial Naive Bayes** over bag-of-words / n-gram features (fast, CPU-only baseline).
   - **DeBERTa** (`ProtectAI/deberta-v3-base-prompt-injection-v2`), run via ONNX Runtime.
2. **Attack taxonomy** — embedding malicious prompts with a sentence-transformer, reducing
   dimensionality with PCA, and clustering with K-Means to discover recurring attack
   patterns (encoding tricks, authority spoofing, roleplay jailbreaks, etc.).

## Results so far

Measured on the held-out test split (see `metrics/test_metrics.csv`):

| Model | Features | Accuracy | Precision | Recall | F1 | Notes |
|-------|----------|----------|-----------|--------|----|-------|
| Multinomial NB (tuned) | word 1–2 grams | 0.997 | 1.00 | 0.994 | 0.997 | ~6k samples/sec, CPU |
| DeBERTa (ONNX) | transformer | 0.92 | 1.00 | 0.871 | 0.931 | ~1 sec/sample |

The NB baseline is both faster and more accurate on this dataset. The transformer's misses
are concentrated in **obfuscated** attacks (Base64, ROT13, leetspeak, zero-width Unicode,
Morse, ASCII art) — see `metrics/deberta.md` for the specific failure cases that motivated
the taxonomy work.

### Discovered attack clusters

K-Means (k=5) over PCA-reduced embeddings of malicious prompts surfaced these groupings
(`experiements/k_means.py`):

- **C0** — Cross-lingual override attacks
- **C1** — Obfuscated instruction injection
- **C2** — Encoded payload & execution attacks
- **C3** — Social-engineering / pretextual jailbreaks
- **C4** — Authority impersonation & privilege escalation

## Dataset

Sourced from the [`wambosec/prompt-injections`](https://huggingface.co/datasets/wambosec/prompt-injections)
dataset on Hugging Face and split into train/val/test under `data/`. Each row has:

```
prompt, is_malicious, category, goal, length_type, label
```

Derived variants live alongside the base splits:

- `*_malicious.csv` — malicious rows only (for clustering).
- `*_normalized.csv` — attack `category` collapsed into a consistent taxonomy.
- `*_top_level.csv` — categories rolled up to *Instruction Manipulation* vs *Obfuscation*.

## Repository layout

```
data/                 Train/val/test splits and derived variants
experiements/         Model + clustering experiments (note: intentional spelling)
  load_data.py          Load benign+malicious splits
  load_malicious.py     Load malicious-only splits
  mnb.py                Multinomial NB with GridSearchCV tuning
  multinomial_nb.py     Simpler NB baseline
  deberta_base.py       DeBERTa (ONNX) classifier
  vectorize.py          Sentence-transformer embeddings
  reduce_dim.py         PCA dimensionality reduction
  k_means.py            K-Means clustering of attack embeddings
  evaluate.py           Metrics + appends to metrics/test_metrics.csv
scripts/
  use_dataset.py        Download dataset from HF and build splits
  extract_malicious.py  Filter malicious rows
  extract_obfuscation.py Normalize attack categories
metrics/              Benchmark results and misclassification logs
main.py               Entry point (empty — WIP)
```

## Getting started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) rebuild the dataset splits from Hugging Face
python scripts/use_dataset.py

# 3. Run an experiment — scripts execute top-to-bottom on import.
#    Run from inside experiements/ since the scripts use local imports.
cd experiements
python mnb.py            # tuned Naive Bayes
python deberta_base.py   # DeBERTa via ONNX Runtime
```

The clustering pipeline runs in sequence:

```bash
cd experiements
python vectorize.py    # -> train_embeddings.pt
python reduce_dim.py   # -> train_embeddings_pca.npy, pca.pkl
python k_means.py      # cluster + inspect centroids
```

## Requirements

`transformers`, `torch`, `pandas`, `scikit-learn`, `optimum[onnxruntime]`,
`sentence-transformers` (see `requirements.txt`).

## Roadmap / TODO

- [ ] Flesh out `main.py` into an actual firewall interface / inference entry point.
- [ ] Add a dedicated obfuscation-normalization pre-processing stage before classification.
- [ ] Combine the NB detector with the cluster-based taxonomy to label *why* a prompt was flagged.
- [ ] Evaluate on out-of-distribution / adversarial obfuscated prompts.
