#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CNN baseline for UIT-ViSFD (SA2SL paper-style evaluation)

- Hardcoded dataset paths:
  /Users/hatrungkien/my-sentiment/data/raw/UIT-ViSFD/{Train,Dev,Test}.csv
- Aspect Detection: multi-label aspects (includes OTHERS)
- Sentiment Detection: multi-label ASPECT#POLARITY (excludes OTHERS)
- Model: Embedding -> Conv1D -> GlobalMaxPooling1D -> Dense(sigmoid)
- Loss: binary_crossentropy
- Threshold tuning on DEV for macro-F1
- Metrics: Macro Precision/Recall/F1 (macro avg)
- Outputs saved next to this script:
  - cnn_output.txt
  - cnn_reports.json

Run:
python /Users/hatrungkien/my-sentiment/baseline/cnn.py

Deps:
pip install tensorflow scikit-learn pandas numpy
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

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


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
tf.random.set_seed(SEED)

# =========================
# CNN CONFIG
# =========================
VOCAB_SIZE = 60000          # vocab cap
SEQ_LEN = 120               # max tokens kept
EMB_DIM = 200
FILTERS = 256
KERNEL_SIZE = 3
DROPOUT = 0.3
DENSE_UNITS = 128
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3

# Threshold grid for DEV tuning
THR_GRID = np.concatenate([
    np.arange(0.10, 0.31, 0.01),  # fine around typical
    np.arange(0.32, 0.91, 0.02)
])

LABEL_RE = re.compile(r"\{([^{}]+)\}")


def parse_label_field(label_str: str) -> Tuple[List[str], List[str]]:
    """
    Returns:
      aspects: ["CAMERA","BATTERY","OTHERS",...]
      aspect_polarities: ["CAMERA#Positive","BATTERY#Negative", ...] (NO OTHERS)
    """
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

            # Exclude OTHERS sentiment
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


def build_text_vectorizer(train_texts: List[str]) -> layers.TextVectorization:
    vec = layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQ_LEN,
        standardize="lower_and_strip_punctuation",
        split="whitespace",
    )
    # adapt on training texts only
    vec.adapt(tf.data.Dataset.from_tensor_slices(train_texts).batch(256))
    return vec


def build_cnn_model(num_labels: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(SEQ_LEN,), dtype=tf.int32)
    x = layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMB_DIM, mask_zero=False)(inputs)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, padding="same", activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(DENSE_UNITS, activation="relu")(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(num_labels, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="binary_crossentropy"
    )
    return model


def make_tf_dataset(x_int: np.ndarray, y: np.ndarray, batch_size: int, training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x_int, y))
    if training:
        ds = ds.shuffle(20000, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def evaluate_with_threshold(
    name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thr: float,
    target_names: List[str]
) -> Dict[str, Any]:
    y_pred = (y_prob >= thr).astype(int)
    p, r, f1 = macro_prf(y_true, y_pred)
    rep = classification_report(
        y_true, y_pred,
        target_names=target_names,
        zero_division=0,
        output_dict=True
    )
    return {
        "name": name,
        "macro_precision": p,
        "macro_recall": r,
        "macro_f1": f1,
        "report": rep
    }


def format_block(title: str, p: float, r: float, f1: float) -> str:
    return (
        f"{title}\n"
        f"Macro Precision: {p*100:.2f}%\n"
        f"Macro Recall:    {r*100:.2f}%\n"
        f"Macro F1:        {f1*100:.2f}%\n"
    )


def run_task(
    task_name: str,
    train_texts: List[str],
    dev_texts: List[str],
    test_texts: List[str],
    y_train_list: List[List[str]],
    y_dev_list: List[List[str]],
    y_test_list: List[List[str]],
    text_vectorizer: layers.TextVectorization,
) -> Dict[str, Any]:
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train_list)
    y_dev = mlb.transform(y_dev_list)
    y_test = mlb.transform(y_test_list)
    labels = list(mlb.classes_)

    # Vectorize texts -> int sequences
    x_train_int = text_vectorizer(tf.constant(train_texts)).numpy()
    x_dev_int = text_vectorizer(tf.constant(dev_texts)).numpy()
    x_test_int = text_vectorizer(tf.constant(test_texts)).numpy()

    model = build_cnn_model(num_labels=len(labels))

    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    train_ds = make_tf_dataset(x_train_int, y_train.astype(np.float32), BATCH_SIZE, training=True)
    dev_ds = make_tf_dataset(x_dev_int, y_dev.astype(np.float32), BATCH_SIZE, training=False)

    model.fit(
        train_ds,
        validation_data=dev_ds,
        epochs=EPOCHS,
        callbacks=[es],
        verbose=2
    )

    # Predict probabilities
    y_prob_dev = model.predict(x_dev_int, batch_size=BATCH_SIZE, verbose=0)
    y_prob_test = model.predict(x_test_int, batch_size=BATCH_SIZE, verbose=0)

    best_thr, best_dev_f1_grid = find_best_threshold(y_dev, y_prob_dev)

    dev_metrics = evaluate_with_threshold(
        name=f"{task_name} (Dev)",
        y_true=y_dev,
        y_prob=y_prob_dev,
        thr=best_thr,
        target_names=labels
    )
    test_metrics = evaluate_with_threshold(
        name=f"{task_name} (Test)",
        y_true=y_test,
        y_prob=y_prob_test,
        thr=best_thr,
        target_names=labels
    )

    return {
        "task": task_name,
        "label_count": len(labels),
        "labels": labels,
        "best_threshold": best_thr,
        "best_dev_f1_on_grid": best_dev_f1_grid,
        "dev": dev_metrics,
        "test": test_metrics,
    }


def main():
    train_df = load_split(TRAIN_CSV)
    dev_df = load_split(DEV_CSV)
    test_df = load_split(TEST_CSV)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_txt = os.path.join(out_dir, "cnn_output.txt")
    out_json = os.path.join(out_dir, "cnn_reports.json")

    train_texts = train_df["comment"].tolist()
    dev_texts = dev_df["comment"].tolist()
    test_texts = test_df["comment"].tolist()

    # Build one shared vectorizer from TRAIN only (same input space for both tasks)
    text_vectorizer = build_text_vectorizer(train_texts)

    # ===== TASK 1: Aspect Detection (includes OTHERS) =====
    aspect_res = run_task(
        task_name="Aspect Detection",
        train_texts=train_texts,
        dev_texts=dev_texts,
        test_texts=test_texts,
        y_train_list=train_df["aspects"].tolist(),
        y_dev_list=dev_df["aspects"].tolist(),
        y_test_list=test_df["aspects"].tolist(),
        text_vectorizer=text_vectorizer,
    )

    # ===== TASK 2: Sentiment Detection (ASPECT#POLARITY) =====
    sent_res = run_task(
        task_name="Sentiment Detection (ASPECT#POLARITY)",
        train_texts=train_texts,
        dev_texts=dev_texts,
        test_texts=test_texts,
        y_train_list=train_df["aspect_pols"].tolist(),
        y_dev_list=dev_df["aspect_pols"].tolist(),
        y_test_list=test_df["aspect_pols"].tolist(),
        text_vectorizer=text_vectorizer,
    )

    lines = []
    lines.append("CNN baseline - UIT-ViSFD")
    lines.append(f"Run time: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"DATA_DIR: {DATA_DIR}")
    lines.append(f"Files: Train={os.path.basename(TRAIN_CSV)}, Dev={os.path.basename(DEV_CSV)}, Test={os.path.basename(TEST_CSV)}")
    lines.append(f"Vectorizer: Keras TextVectorization vocab={VOCAB_SIZE}, seq_len={SEQ_LEN}")
    lines.append(f"Model: Emb({EMB_DIM}) -> Conv1D(filters={FILTERS}, k={KERNEL_SIZE}) -> GlobalMaxPool -> Dense({DENSE_UNITS}) -> sigmoid")
    lines.append(f"Train: batch={BATCH_SIZE}, epochs={EPOCHS}, lr={LR}, dropout={DROPOUT}")
    lines.append("Threshold tuning on Dev (grid).")
    lines.append("")

    lines.append(f"[{aspect_res['task']}] best_thr={aspect_res['best_threshold']:.2f} (best_dev_f1_on_grid={aspect_res['best_dev_f1_on_grid']*100:.2f}%)")
    lines.append(format_block(
        "Aspect Detection (Dev)",
        aspect_res["dev"]["macro_precision"],
        aspect_res["dev"]["macro_recall"],
        aspect_res["dev"]["macro_f1"],
    ))
    lines.append(format_block(
        "Aspect Detection (Test)",
        aspect_res["test"]["macro_precision"],
        aspect_res["test"]["macro_recall"],
        aspect_res["test"]["macro_f1"],
    ))

    lines.append(f"[{sent_res['task']}] best_thr={sent_res['best_threshold']:.2f} (best_dev_f1_on_grid={sent_res['best_dev_f1_on_grid']*100:.2f}%)")
    lines.append(format_block(
        "Sentiment Detection (Dev) [ASPECT#POLARITY]",
        sent_res["dev"]["macro_precision"],
        sent_res["dev"]["macro_recall"],
        sent_res["dev"]["macro_f1"],
    ))
    lines.append(format_block(
        "Sentiment Detection (Test) [ASPECT#POLARITY]",
        sent_res["test"]["macro_precision"],
        sent_res["test"]["macro_recall"],
        sent_res["test"]["macro_f1"],
    ))

    text_out = "\n".join(lines)
    print(text_out)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text_out)

    payload = {
        "config": {
            "DATA_DIR": DATA_DIR,
            "TRAIN_CSV": TRAIN_CSV,
            "DEV_CSV": DEV_CSV,
            "TEST_CSV": TEST_CSV,
            "SEED": SEED,
            "VOCAB_SIZE": VOCAB_SIZE,
            "SEQ_LEN": SEQ_LEN,
            "EMB_DIM": EMB_DIM,
            "FILTERS": FILTERS,
            "KERNEL_SIZE": KERNEL_SIZE,
            "DROPOUT": DROPOUT,
            "DENSE_UNITS": DENSE_UNITS,
            "BATCH_SIZE": BATCH_SIZE,
            "EPOCHS": EPOCHS,
            "LR": LR,
            "THR_GRID_MIN": float(THR_GRID.min()),
            "THR_GRID_MAX": float(THR_GRID.max()),
            "THR_GRID_SIZE": int(len(THR_GRID)),
        },
        "aspect": aspect_res,
        "sentiment": sent_res,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\nSaved outputs next to cnn.py:")
    print(f"- {out_txt}")
    print(f"- {out_json}")


if __name__ == "__main__":
    main()
