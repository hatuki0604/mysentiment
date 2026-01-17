#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SVM baseline (DEFAULT) for UIT-ViSFD — paper-style evaluation WITHOUT threshold tuning.

This version is "default" in the sense:
- Uses standard Pipeline + OneVsRestClassifier(LinearSVC)
- Uses model.predict() directly (NO threshold tuning)
- Hardcoded dataset paths (no CLI args)
- Macro Precision/Recall/F1 (macro avg)
- Saves outputs next to this script:
  - svm_output.txt
  - svm_reports.json

Tasks:
1) Aspect Detection: multi-label aspects (includes OTHERS)
2) Sentiment Detection: multi-label ASPECT#POLARITY (excludes OTHERS)

Run:
python /Users/hatrungkien/my-sentiment/baseline/svm.py
"""

import json
import os
import re
from datetime import datetime
from typing import List, Tuple, Dict, Any

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC


# =========================
# HARD-CODED DATA PATHS
# =========================
DATA_DIR = "/Users/hatrungkien/my-sentiment/data/raw/UIT-ViSFD"
TRAIN_CSV = os.path.join(DATA_DIR, "Train.csv")
DEV_CSV   = os.path.join(DATA_DIR, "Dev.csv")
TEST_CSV  = os.path.join(DATA_DIR, "Test.csv")

# =========================
# DEFAULT CONFIG
# =========================
NGRAM_RANGE = (1, 2)        # common default for TF-IDF SVM
MAX_FEATURES = 50000        # cap vocabulary size; set None to disable
C = 1.0                    # LinearSVC regularization
CLASS_WEIGHT = None        # set to "balanced" if needed


LABEL_RE = re.compile(r"\{([^{}]+)\}")  # captures inside {...}


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

            # Exclude OTHERS sentiment (paper: NaN / not evaluated)
            if asp.upper() != "OTHERS" and pol:
                aspect_pols.append(f"{asp}#{pol}")
            continue

        aspects.append(ch)

    aspects = list(dict.fromkeys(aspects))
    aspect_pols = list(dict.fromkeys(aspect_pols))
    return aspects, aspect_pols


def macro_prf(y_true, y_pred) -> Tuple[float, float, float]:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return float(p), float(r), float(f1)


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


def build_svm_pipeline() -> Pipeline:
    vec = TfidfVectorizer(
        ngram_range=NGRAM_RANGE,
        max_features=MAX_FEATURES,
        lowercase=True
    )
    svm = LinearSVC(C=C, class_weight=CLASS_WEIGHT)
    return Pipeline([
        ("vec", vec),
        ("clf", OneVsRestClassifier(svm))
    ])


def evaluate_task_default_predict(
    task_name: str,
    model: Pipeline,
    mlb: MultiLabelBinarizer,
    x_train: pd.Series,
    y_train_list: List[List[str]],
    x_dev: pd.Series,
    y_dev_list: List[List[str]],
    x_test: pd.Series,
    y_test_list: List[List[str]],
) -> Dict[str, Any]:
    # Fit label space on TRAIN
    y_train = mlb.fit_transform(y_train_list)
    y_dev = mlb.transform(y_dev_list)
    y_test = mlb.transform(y_test_list)
    target_names = list(mlb.classes_)

    # Train
    model.fit(x_train, y_train)

    # DEFAULT predict (no threshold tuning)
    y_pred_dev = model.predict(x_dev)
    y_pred_test = model.predict(x_test)

    p_dev, r_dev, f1_dev = macro_prf(y_dev, y_pred_dev)
    p_test, r_test, f1_test = macro_prf(y_test, y_pred_test)

    rep_dev = classification_report(y_dev, y_pred_dev, target_names=target_names, zero_division=0, output_dict=True)
    rep_test = classification_report(y_test, y_pred_test, target_names=target_names, zero_division=0, output_dict=True)

    return {
        "task": task_name,
        "label_count": len(target_names),
        "dev": {"macro_precision": p_dev, "macro_recall": r_dev, "macro_f1": f1_dev, "report": rep_dev},
        "test": {"macro_precision": p_test, "macro_recall": r_test, "macro_f1": f1_test, "report": rep_test},
        "labels": target_names,
    }


def format_block(title: str, metrics: Dict[str, float]) -> str:
    return (
        f"{title}\n"
        f"Macro Precision: {metrics['macro_precision']*100:.2f}%\n"
        f"Macro Recall:    {metrics['macro_recall']*100:.2f}%\n"
        f"Macro F1:        {metrics['macro_f1']*100:.2f}%\n"
    )


def main():
    train_df = load_split(TRAIN_CSV)
    dev_df = load_split(DEV_CSV)
    test_df = load_split(TEST_CSV)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_txt = os.path.join(out_dir, "svm_output.txt")
    out_json = os.path.join(out_dir, "svm_reports.json")

    # ===== TASK 1: Aspect Detection (includes OTHERS) =====
    aspect_model = build_svm_pipeline()
    aspect_mlb = MultiLabelBinarizer()
    aspect_res = evaluate_task_default_predict(
        task_name="Aspect Detection",
        model=aspect_model,
        mlb=aspect_mlb,
        x_train=train_df["comment"],
        y_train_list=train_df["aspects"].tolist(),
        x_dev=dev_df["comment"],
        y_dev_list=dev_df["aspects"].tolist(),
        x_test=test_df["comment"],
        y_test_list=test_df["aspects"].tolist(),
    )

    # ===== TASK 2: Sentiment Detection (ASPECT#POLARITY, excludes OTHERS) =====
    sent_model = build_svm_pipeline()
    sent_mlb = MultiLabelBinarizer()
    sent_res = evaluate_task_default_predict(
        task_name="Sentiment Detection (ASPECT#POLARITY)",
        model=sent_model,
        mlb=sent_mlb,
        x_train=train_df["comment"],
        y_train_list=train_df["aspect_pols"].tolist(),
        x_dev=dev_df["comment"],
        y_dev_list=dev_df["aspect_pols"].tolist(),
        x_test=test_df["comment"],
        y_test_list=test_df["aspect_pols"].tolist(),
    )

    lines = []
    lines.append("SVM baseline (DEFAULT predict) - UIT-ViSFD")
    lines.append(f"Run time: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"DATA_DIR: {DATA_DIR}")
    lines.append(f"Files: Train={os.path.basename(TRAIN_CSV)}, Dev={os.path.basename(DEV_CSV)}, Test={os.path.basename(TEST_CSV)}")
    lines.append(f"Vectorizer: TF-IDF ngram={NGRAM_RANGE}, max_features={MAX_FEATURES}")
    lines.append(f"Model: OneVsRest(LinearSVC(C={C}, class_weight={CLASS_WEIGHT}))")
    lines.append("Note: using model.predict() (no threshold tuning)")
    lines.append("")

    lines.append(format_block("Aspect Detection (Dev)", aspect_res["dev"]))
    lines.append(format_block("Aspect Detection (Test)", aspect_res["test"]))

    lines.append(format_block("Sentiment Detection (Dev) [ASPECT#POLARITY]", sent_res["dev"]))
    lines.append(format_block("Sentiment Detection (Test) [ASPECT#POLARITY]", sent_res["test"]))

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
            "vectorizer": "TfidfVectorizer",
            "ngram_range": NGRAM_RANGE,
            "max_features": MAX_FEATURES,
            "svm_C": C,
            "class_weight": CLASS_WEIGHT,
            "predict_mode": "default_predict_no_threshold_tuning"
        },
        "aspect": aspect_res,
        "sentiment": sent_res,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\nSaved outputs next to svm.py:")
    print(f"- {out_txt}")
    print(f"- {out_json}")


if __name__ == "__main__":
    main()
