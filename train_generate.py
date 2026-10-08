"""Train a small autoregressive peptide GRU and a separate MIC predictor.
Input: EC.csv with SEQUENCE and EC_MIC (micromolar). Exploratory research only.
Run: python train_generate.py --data EC.csv --output results
"""
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

AA = "ACDEFGHIKLMNPQRSTVWY"
TOK = ["<pad>", "<bos>", "<eos>"] + list(AA)
ID = {token: i for i, token in enumerate(TOK)}
MAXLEN = 45

def features(seq):
    n = len(seq)
    return [seq.count(a) / n for a in AA] + [
        n,
        (sum(seq.count(a) for a in "KRH") - sum(seq.count(a) for a in "DE")) / n,
        sum(seq.count(a) for a in "AILMFWV") / n,
    ]

def encode(seq):
    tokens = [ID["<bos>"]] + [ID[a] for a in seq] + [ID["<eos>"]]
    return tokens + [0] * (MAXLEN + 2 - len(tokens))

class Generator(nn.Module):
    def __init__(self, emb=24, hidden=64):
        super().__init__()
        self.embedding = nn.Embedding(len(TOK), emb, padding_idx=0)
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.head = nn.Linear(hidden, len(TOK))

    def forward(self, x, hidden=None):
        out, hidden = self.gru(self.embedding(x), hidden)
        return self.head(out), hidden

@torch.no_grad()
def sample(model, count, seed=52, temperature=0.9):
    torch.manual_seed(seed)
    model.eval()
    sequences = []
    for _ in range(count):
        token = torch.tensor([[ID["<bos>"]]])
        hidden, letters = None, []
        for step in range(MAXLEN):
            logits, hidden = model(token, hidden)
            scores = logits[0, -1].clone() / temperature
            scores[ID["<pad>"]] = -1e9
            scores[ID["<bos>"]] = -1e9
            if step < 7:
                scores[ID["<eos>"]] = -1e9
            next_id = int(torch.multinomial(torch.softmax(scores, dim=0), 1))
            if next_id == ID["<eos>"]:
                break
            letters.append(TOK[next_id])
            token = torch.tensor([[next_id]])
        sequences.append("".join(letters))
    return sequences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="EC.csv")
    parser.add_argument("--output", default="results")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--samples", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.data)
    data["SEQUENCE"] = data["SEQUENCE"].astype(str).str.upper().str.strip()
    data["EC_MIC"] = pd.to_numeric(data["EC_MIC"], errors="coerce")
    data = data[
        data.SEQUENCE.str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]+")
        & data.SEQUENCE.str.len().between(8, MAXLEN)
        & (data.EC_MIC > 0)
        & np.isfinite(data.EC_MIC)
    ].copy()
    data = data.groupby("SEQUENCE", as_index=False).agg(EC_MIC=("EC_MIC", "median"))
    data["y"] = np.log10(data.EC_MIC)

    train, test = train_test_split(data, test_size=0.2, random_state=args.seed)
    train, validation = train_test_split(train, test_size=0.2, random_state=args.seed)
    model = ExtraTreesRegressor(
        n_estimators=120, min_samples_leaf=2, n_jobs=2, random_state=args.seed
    )
    model.fit(np.asarray([features(s) for s in train.SEQUENCE]), train.y)
    predictions = model.predict(np.asarray([features(s) for s in test.SEQUENCE]))
    pd.DataFrame({
        "sequence": test.SEQUENCE,
        "observed_mic_uM": test.EC_MIC,
        "predicted_log10_mic_uM": predictions,
    }).to_csv(out / "heldout_predictor_predictions.csv", index=False)

    encoded = torch.tensor([encode(s) for s in train.SEQUENCE], dtype=torch.long)
    generator = Generator()
    optimizer = torch.optim.Adam(generator.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    history = []
    for epoch in range(args.epochs):
        generator.train()
        losses = []
        for indices in torch.randperm(len(encoded)).split(128):
            batch = encoded[indices]
            logits, _ = generator(batch[:, :-1])
            loss = criterion(
                logits.reshape(-1, len(TOK)), batch[:, 1:].reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        history.append({"epoch": epoch + 1, "train_token_cross_entropy": float(np.mean(losses))})
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    torch.save({
        "state_dict": generator.state_dict(), "vocabulary": TOK,
        "max_length": MAXLEN, "seed": args.seed,
    }, out / "generator_weights.pt")

    generated = sample(generator, args.samples, seed=args.seed + 10)
    valid = [s for s in generated if 8 <= len(s) <= MAXLEN and set(s) <= set(AA)]
    novel = [s for s in dict.fromkeys(valid) if s not in set(data.SEQUENCE)]
    scores = model.predict(np.asarray([features(s) for s in novel])) if novel else []
    candidates = pd.DataFrame({
        "sequence": novel, "length": [len(s) for s in novel],
        "predicted_log10_MIC_uM": scores,
    })
    if len(candidates):
        candidates["predicted_MIC_uM"] = 10 ** candidates.predicted_log10_MIC_uM
        candidates = candidates.sort_values("predicted_MIC_uM").reset_index(drop=True)
        candidates["rank"] = np.arange(1, len(candidates) + 1)
    candidates.to_csv(out / "generated_candidates.csv", index=False)
    summary = {
        "n_total": len(data), "n_predictor_train": len(train),
        "n_predictor_validation": len(validation), "n_predictor_test": len(test),
        "predictor_test_MAE_log10_uM": float(mean_absolute_error(test.y, predictions)),
        "predictor_test_R2": float(r2_score(test.y, predictions)),
        "samples_requested": args.samples, "valid_samples": len(valid),
        "novel_unique_samples": len(novel),
        "generation_method": "autoregressive PyTorch GRU",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
