# Peptide Design: Generative Modelling and MIC Prediction

An exploratory computational research project for **antimicrobial peptide sequence generation and potency prediction**.

The pipeline trains a lightweight **PyTorch autoregressive GRU** on canonical peptide sequences and uses a separate **ExtraTrees regression model** to rank generated sequences by predicted *Escherichia coli* minimum inhibitory concentration (MIC).

## Research workflow

1. Read peptide sequences and MIC values (µM) from `EC.csv`.
2. Remove noncanonical sequences and restrict peptide length to 8–45 residues.
3. Aggregate duplicate sequence measurements using the median MIC.
4. Train an ExtraTrees model on amino-acid composition, length, charge proxy and hydrophobicity proxy.
5. Train a PyTorch GRU on training sequences only.
6. Sample candidate sequences and exclude exact matches to the source dataset.
7. Rank candidates by predicted MIC; export tables and evaluation figures.

## Reproduce

Python 3.10+ recommended.

```bash
python -m pip install -r requirements.txt
python train_generate.py --data EC.csv --output results --epochs 4 --samples 350
python make_figures.py
```

The source data file `EC.csv` is **not redistributed** here because its redistribution rights have not been confirmed. To reproduce the run, provide a CSV containing `SEQUENCE` and `EC_MIC` (MIC in µM).

## Exploratory results (seed 42)

| Metric | Value |
| --- | ---: |
| Curated unique peptides | 3,834 |
| Training peptides | 2,453 |
| Predictor test peptides | 767 |
| Predictor MAE, log10(µM) | 0.435 |
| Predictor test R² | 0.531 |
| Generated samples | 350 |
| Unique valid sequences absent by exact match | 343 |

See [results/summary.json](results/summary.json) and [results/training_history.csv](results/training_history.csv).

## Figures

Run `python make_figures.py` to regenerate:

- `results/training_curve.png` — generator token cross-entropy across epochs
- `results/predictor_test.png` — observed versus predicted log10 MIC
- `results/candidate_scores.png` — predicted MIC distribution for generated candidates

## Interpretation and limitations

**Generated sequences are computational hypotheses, not experimentally validated antimicrobial peptides.** Predicted MIC does not establish antibacterial efficacy, toxicity, stability, permeability or developability.

The random train/test split prevents exact sequence duplication but does not guarantee homology independence. The GRU is unconditional, not a pretrained protein language model or property-conditioned generator. The predictor uses simple sequence features. Novelty was checked by exact matching only; structural novelty, homology novelty and experimental activity remain unverified. Hyperparameter optimisation and external validation are not claimed.

This repository is a reproducible research prototype, not a validated drug-discovery platform.

## Repository contents

- `train_generate.py` — model training, generation, candidate screening
- `make_figures.py` — plot generation
- `requirements.txt` — Python dependencies
- `results/` — run summary and evaluation artifacts

## Research integrity

Dataset attribution, license compliance, robust sequence-cluster evaluation and independent laboratory testing are prerequisites for stronger scientific claims.