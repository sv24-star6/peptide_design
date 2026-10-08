"""Create evaluation figures from local results CSV files."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

out = Path("results")
history = pd.read_csv(out / "training_history.csv")
pred = pd.read_csv(out / "heldout_predictor_predictions.csv")
candidates = pd.read_csv(out / "generated_candidates.csv")

plt.figure(figsize=(6, 4))
plt.plot(history.epoch, history.train_token_cross_entropy, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training token cross-entropy")
plt.title("Peptide GRU training")
plt.tight_layout()
plt.savefig(out / "training_curve.png", dpi=180)
plt.close()

plt.figure(figsize=(5, 5))
plt.scatter(np.log10(pred.observed_mic_uM), pred.predicted_log10_mic_uM, s=7, alpha=0.35)
plt.xlabel("Observed log10 MIC (µM)")
plt.ylabel("Predicted log10 MIC (µM)")
plt.title("Held-out predictor evaluation")
plt.tight_layout()
plt.savefig(out / "predictor_test.png", dpi=180)
plt.close()

plt.figure(figsize=(6, 4))
plt.hist(candidates.predicted_MIC_uM, bins=30)
plt.xlabel("Predicted MIC (µM)")
plt.ylabel("Generated sequences")
plt.title("Computational candidate scores")
plt.tight_layout()
plt.savefig(out / "candidate_scores.png", dpi=180)
plt.close()
