#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LSTM baseline (PyTorch) for UIT-ViSFD (SA2SL paper-style evaluation)

- Hardcoded dataset paths:
  /Users/hatrungkien/my-sentiment/data/raw/UIT-ViSFD/{Train,Dev,Test}.csv
- 2 tasks:
  1) Aspect Detection: multi-label aspects (includes OTHERS)
  2) Sentiment Detection: multi-label ASPECT#POLARITY (excludes OTHERS)
- Model: Embedding -> BiLSTM -> Pooling (max+mean) -> MLP -> logits
- Loss: BCEWithLogitsLoss
- Threshold tuning on DEV for macro-F1
- Outputs saved next to this script:
  - lstm_output.txt
  - lstm_reports.json

Run:
python /Users/hatrungkien/my-sentiment/baseline/lstm_torch.py
"""

import json
import os
import re
import random
from datetime import datetime
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# HARD-CODED DATA PATHS
# =========================
DATA_DIR = "/Users/hatrungkien/my-sentiment/data/raw/UIT-ViSFD"
TRAIN_CSV = os.path.join(DATA_DIR, "Train.csv")
DEV_CSV   = os.path.join(DATA_DIR, "Dev.csv")
TEST_CSV  = os.path.join(DATA_DIR, "Test.csv")


# =========================
# REPRODUCIBILITY
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================
# CONFIG
# =========================
VOCAB_SIZE = 60000
SEQ_LEN = 160

EMB_DIM = 200
HIDDEN = 256
NUM_LAYERS = 1
BIDIR = True
DROPOUT = 0.3

MLP_HIDDEN = 256

BATCH_SIZE = 64
EPOCHS = 8
LR = 1e-3
WEIGHT_DECAY = 0.0
CLIP_NORM = 1.0

THR_GRID = np.concatenate([
    np.arange(0.10, 0.31, 0.01),
    np.arange(0.32, 0.91, 0.02),
])

PAD = "<pad>"
UNK = "<unk>"

LABEL_RE = re.compile(r"\{([^{}]+)\}")


def device_auto():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_label_field(label_str: str) -> Tuple[List[str], List[str]]:
    if not isinstance(label_str, str) or not label_str.strip():
        return [], []

    chunks = LABEL_RE.findall(label_str)
    aspects: List[str] = []
    aspect_pols: List[str] = []

    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue

        if ch.upper() == "OTHERS":
            aspects.append("OTHERS")
            continue

        if "#" in ch:
            asp, pol = ch.split("#", 1)
            asp = asp.strip()
            pol = pol.strip()

            if asp:
                aspects.append(asp)
            if asp.upper() != "OTHERS" and pol:
                aspect_pols.append(f"{asp}#{pol}")
            continue

        aspects.append(ch)

    aspects = list(dict.fromkeys(aspects))
    aspect_pols = list(dict.fromkeys(aspect_pols))
    return aspects, aspect_pols


def load_split(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "comment" not in df.columns or "label" not in df.columns:
        raise ValueError(f"CSV {csv_path} must have columns: comment, label")

    df["comment"] = df["comment"].astype(str).fillna("")
    df["label"] = df["label"].astype(str).fillna("")

    parsed = df["label"].apply(parse_label_field)
    df["aspects"] = parsed.apply(lambda x: x[0])
    df["aspect_pols"] = parsed.apply(lambda x: x[1])
    return df


def simple_tokenize(text: str) -> List[str]:
    return text.lower().split()


def build_vocab(texts: List[str], max_vocab: int) -> Dict[str, int]:
    from collections import Counter
    cnt = Counter()
    for t in texts:
        cnt.update(simple_tokenize(t))

    vocab = {PAD: 0, UNK: 1}
    for w, _ in cnt.most_common(max_vocab - len(vocab)):
        if w not in vocab:
            vocab[w] = len(vocab)
    return vocab


def encode(text: str, vocab: Dict[str, int], seq_len: int) -> np.ndarray:
    ids = [vocab.get(tok, vocab[UNK]) for tok in simple_tokenize(text)]
    ids = ids[:seq_len]
    if len(ids) < seq_len:
        ids += [vocab[PAD]] * (seq_len - len(ids))
    return np.array(ids, dtype=np.int64)


def macro_prf(y_true, y_pred) -> Tuple[float, float, float]:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return float(p), float(r), float(f1)


def find_best_threshold(y_true_dev: np.ndarray, y_prob_dev: np.ndarray) -> Tuple[float, float]:
    best_thr, best_f1 = 0.5, -1.0
    for thr in THR_GRID:
        y_pred = (y_prob_dev >= thr).astype(int)
        _, _, f1 = macro_prf(y_true_dev, y_pred)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr), float(best_f1)


class TextDataset(Dataset):
    def __init__(self, x_int: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x_int)              # (N, L)
        self.y = torch.from_numpy(y).float()          # (N, C)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden: int, num_layers: int,
                 bidir: bool, dropout: float, mlp_hidden: int, num_labels: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidir,
            dropout=lstm_dropout
        )

        out_dim = hidden * (2 if bidir else 1)

        # Pooling: concat(max_pool, mean_pool)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim * 2, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_labels)
        )

    def forward(self, x):  # x: (B, L)
        e = self.emb(x)  # (B, L, E)
        h, _ = self.lstm(e)  # (B, L, H*)

        # max pooling over time
        max_pool = torch.max(h, dim=1).values
        # mean pooling over time
        mean_pool = torch.mean(h, dim=1)

        z = torch.cat([max_pool, mean_pool], dim=1)
        z = self.dropout(z)
        logits = self.mlp(z)
        return logits


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    all_probs = []
    for xb, _ in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


def train_model(model: nn.Module, train_loader: DataLoader, dev_loader: DataLoader,
                device: torch.device) -> None:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    best_dev_loss = float("inf")
    patience, bad = 2, 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            opt.step()

        # dev loss for early stopping
        model.eval()
        dev_loss = 0.0
        with torch.no_grad():
            for xb, yb in dev_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                dev_loss += float(loss.item())

        dev_loss /= max(1, len(dev_loader))
        if dev_loss < best_dev_loss - 1e-4:
            best_dev_loss = dev_loss
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)


def run_task(task_name: str,
             train_texts: List[str], dev_texts: List[str], test_texts: List[str],
             y_train_list: List[List[str]], y_dev_list: List[List[str]], y_test_list: List[List[str]],
             vocab: Dict[str, int],
             device: torch.device) -> Dict[str, Any]:

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train_list)
    y_dev = mlb.transform(y_dev_list)
    y_test = mlb.transform(y_test_list)
    labels = list(mlb.classes_)

    x_train = np.stack([encode(t, vocab, SEQ_LEN) for t in train_texts])
    x_dev = np.stack([encode(t, vocab, SEQ_LEN) for t in dev_texts])
    x_test = np.stack([encode(t, vocab, SEQ_LEN) for t in test_texts])

    train_ds = TextDataset(x_train, y_train)
    dev_ds = TextDataset(x_dev, y_dev)
    test_ds = TextDataset(x_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        emb_dim=EMB_DIM,
        hidden=HIDDEN,
        num_layers=NUM_LAYERS,
        bidir=BIDIR,
        dropout=DROPOUT,
        mlp_hidden=MLP_HIDDEN,
        num_labels=len(labels)
    )

    train_model(model, train_loader, dev_loader, device)

    y_prob_dev = predict_probs(model, dev_loader, device)
    y_prob_test = predict_probs(model, test_loader, device)

    best_thr, best_dev_f1_grid = find_best_threshold(y_dev, y_prob_dev)

    y_pred_dev = (y_prob_dev >= best_thr).astype(int)
    y_pred_test = (y_prob_test >= best_thr).astype(int)

    p_dev, r_dev, f1_dev = macro_prf(y_dev, y_pred_dev)
    p_test, r_test, f1_test = macro_prf(y_test, y_pred_test)

    rep_dev = classification_report(y_dev, y_pred_dev, target_names=labels, zero_division=0, output_dict=True)
    rep_test = classification_report(y_test, y_pred_test, target_names=labels, zero_division=0, output_dict=True)

    return {
        "task": task_name,
        "label_count": len(labels),
        "labels": labels,
        "best_threshold": best_thr,
        "best_dev_f1_on_grid": best_dev_f1_grid,
        "dev": {"macro_precision": p_dev, "macro_recall": r_dev, "macro_f1": f1_dev, "report": rep_dev},
        "test": {"macro_precision": p_test, "macro_recall": r_test, "macro_f1": f1_test, "report": rep_test},
    }


def format_block(title: str, m: Dict[str, float]) -> str:
    return (
        f"{title}\n"
        f"Macro Precision: {m['macro_precision']*100:.2f}%\n"
        f"Macro Recall:    {m['macro_recall']*100:.2f}%\n"
        f"Macro F1:        {m['macro_f1']*100:.2f}%\n"
    )


def main():
    train_df = load_split(TRAIN_CSV)
    dev_df = load_split(DEV_CSV)
    test_df = load_split(TEST_CSV)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_txt = os.path.join(out_dir, "lstm_output.txt")
    out_json = os.path.join(out_dir, "lstm_reports.json")

    train_texts = train_df["comment"].tolist()
    dev_texts = dev_df["comment"].tolist()
    test_texts = test_df["comment"].tolist()

    vocab = build_vocab(train_texts, VOCAB_SIZE)
    dev = device_auto()

    aspect_res = run_task(
        "Aspect Detection",
        train_texts, dev_texts, test_texts,
        train_df["aspects"].tolist(), dev_df["aspects"].tolist(), test_df["aspects"].tolist(),
        vocab, dev
    )

    sent_res = run_task(
        "Sentiment Detection (ASPECT#POLARITY)",
        train_texts, dev_texts, test_texts,
        train_df["aspect_pols"].tolist(), dev_df["aspect_pols"].tolist(), test_df["aspect_pols"].tolist(),
        vocab, dev
    )

    lines = []
    lines.append("LSTM baseline (PyTorch) - UIT-ViSFD")
    lines.append(f"Run time: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"DATA_DIR: {DATA_DIR}")
    lines.append(f"Device: {dev}")
    lines.append(f"Vocab: {len(vocab)} | SeqLen: {SEQ_LEN} | Emb: {EMB_DIM}")
    lines.append(f"BiLSTM hidden: {HIDDEN} | layers: {NUM_LAYERS} | bidir: {BIDIR} | dropout: {DROPOUT}")
    lines.append(f"Pooling: max + mean | MLP hidden: {MLP_HIDDEN}")
    lines.append(f"Train: batch={BATCH_SIZE}, epochs={EPOCHS}, lr={LR}")
    lines.append("Threshold tuning on Dev (grid).")
    lines.append("")

    lines.append(f"[{aspect_res['task']}] best_thr={aspect_res['best_threshold']:.2f} (best_dev_f1_on_grid={aspect_res['best_dev_f1_on_grid']*100:.2f}%)")
    lines.append(format_block("Aspect Detection (Dev)", aspect_res["dev"]))
    lines.append(format_block("Aspect Detection (Test)", aspect_res["test"]))

    lines.append(f"[{sent_res['task']}] best_thr={sent_res['best_threshold']:.2f} (best_dev_f1_on_grid={sent_res['best_dev_f1_on_grid']*100:.2f}%)")
    lines.append(format_block("Sentiment Detection (Dev) [ASPECT#POLARITY]", sent_res["dev"]))
    lines.append(format_block("Sentiment Detection (Test) [ASPECT#POLARITY]", sent_res["test"]))

    text_out = "\n".join(lines)
    print(text_out)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text_out)

    payload = {
        "config": {
            "DATA_DIR": DATA_DIR,
            "SEED": SEED,
            "VOCAB_SIZE": VOCAB_SIZE,
            "SEQ_LEN": SEQ_LEN,
            "EMB_DIM": EMB_DIM,
            "HIDDEN": HIDDEN,
            "NUM_LAYERS": NUM_LAYERS,
            "BIDIR": BIDIR,
            "DROPOUT": DROPOUT,
            "MLP_HIDDEN": MLP_HIDDEN,
            "BATCH_SIZE": BATCH_SIZE,
            "EPOCHS": EPOCHS,
            "LR": LR,
            "device": str(dev),
        },
        "aspect": aspect_res,
        "sentiment": sent_res,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\nSaved outputs next to lstm_torch.py:")
    print(f"- {out_txt}")
    print(f"- {out_json}")


if __name__ == "__main__":
    main()
