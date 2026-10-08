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

## Inputs and observed outcomes

The project has **two distinct input-to-output workflows**. The numbers below are from one exploratory run (random seed 42), not experimental laboratory validation.

| Workflow | Input | Processing | Output / observed outcome |
| --- | --- | --- | --- |
| Dataset curation | `EC.csv` containing `SEQUENCE` and `EC_MIC` (MIC in µM) | Canonical amino acids only, 8–45 residues; median MIC for duplicate sequences | **3,834 unique eligible peptide sequences** |
| MIC prediction | Peptide sequence represented by amino-acid frequencies, length, charge and hydrophobicity proxies | ExtraTrees regression predicts log10(MIC in µM) | On **767 held-out sequences**: **MAE 0.435 log10(µM)** and **R² 0.531** |
| Generative sequence modelling | **2,453 training peptide sequences** from the curated dataset; no desired property specified | PyTorch autoregressive GRU, four training epochs | **350 sampled sequences**, of which **343 were unique, valid and absent from the curated dataset by exact match** |
| Candidate screening | Newly generated valid peptide sequences | Trained ExtraTrees predictor estimates MIC | Candidate sequences are ranked by **predicted MIC (µM)**; lower values indicate stronger *predicted* antibacterial potency |

### Input example and output interpretation

**Predicting activity for an existing peptide**

- **Input:** A peptide sequence containing standard one-letter amino-acid codes, for example `KWKLFKKIGAVLKVL`. This is an *illustrative input format*, **not** a verified prediction from the saved experiment.
- **Output:** A predicted log10 MIC value and its corresponding MIC in µM. For example, a hypothetical prediction of **0.30 log10(µM)** corresponds to **about 2.0 µM** (`10 ** 0.30`). This is a calculation example, **not a measured or previously generated result**.

**Generating and prioritising new peptides**

- **Input:** A trained generator and sampling parameters (`--samples 350`, random seed 42). The generator is **unconditional**: it does not accept a target MIC, desired affinity, or toxicity constraint.
- **Output:** Candidate amino-acid sequences, their lengths, predicted log10 MIC and predicted MIC in µM, ordered from lower to higher predicted MIC.
- **Observed outcome:** 343 unique valid generated sequences were not exact matches to any curated input sequence. This demonstrates sequence generation and exact-match novelty, **not confirmed antimicrobial activity**.

### What the metrics mean

- **MAE 0.435 log10(µM):** The predictor's average absolute error on held-out sequences in log10 MIC units. It should not be interpreted as an error of 0.435 µM.
- **R² 0.531:** The model explained approximately 53.1% of the variation in the held-out log10 MIC values in this split.
- **343 novel sequences:** Novelty is limited to **exact sequence matching** against this dataset. Similarity-controlled novelty, toxicity, synthesis feasibility and laboratory MIC remain untested.

The experimental outputs are summarised in [results/summary.json](results/summary.json) and [results/RESULTS.md](results/RESULTS.md). The [training curve](results/training_curve.svg) shows the GRU's training loss over four epochs.

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