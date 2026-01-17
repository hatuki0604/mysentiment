#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Random Forest baseline for UIT-ViSFD (SA2SL paper-style evaluation)

Goal: "giống bài báo cũ" (classic ML baseline)
- Hardcoded dataset paths:
  /Users/hatrungkien/my-sentiment/data/raw/UIT-ViSFD/{Train,Dev,Test}.csv
- Aspect Detection: multi-label aspects (includes OTHERS)
- Sentiment Detection: multi-label ASPECT#POLARITY (excludes OTHERS)
- Multi-label via OneVsRest
- Vectorizer: Bag-of-Words (CountVectorizer) unigram (classic baseline)
- Classifier: RandomForestClassifier
- Predict: DEFAULT model.predict() (NO threshold tuning) to stay baseline-ish
- Metrics: Macro Precision/Recall/F1 (macro avg)
- Outputs saved next to this script:
  - rf_output.txt
  - rf_reports.json

Run:
python /Users/hatrungkien/my-sentiment/baseline/random_forest.py
"""

import json
import os
import re
from datetime import datetime
from typing import List, Tuple, Dict, Any

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier


# =========================
# HARD-CODED DATA PATHS
# =========================
DATA_DIR = "/Users/hatrungkien/my-sentiment/data/raw/UIT-ViSFD"
TRAIN_CSV = os.path.join(DATA_DIR, "Train.csv")
DEV_CSV   = os.path.join(DATA_DIR, "Dev.csv")
TEST_CSV  = os.path.join(DATA_DIR, "Test.csv")

# =========================
# DEFAULT CONFIG (classic baseline)
# =========================
NGRAM_RANGE = (1, 1)        # BoW unigram
MAX_FEATURES = None         # keep all vocab (can set 50000 if you want)

# Random Forest params (reasonable baseline)
N_ESTIMATORS = 300
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1
N_JOBS = -1
RANDOM_STATE = 42


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


def build_rf_pipeline() -> Pipeline:
    vec = CountVectorizer(
        ngram_range=NGRAM_RANGE,
        max_features=MAX_FEATURES,
        lowercase=True
    )

    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE
    )

    return Pipeline([
        ("vec", vec),
        ("clf", OneVsRestClassifier(rf))
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
    y_train = mlb.fit_transform(y_train_list)
    y_dev = mlb.transform(y_dev_list)
    y_test = mlb.transform(y_test_list)
    target_names = list(mlb.classes_)

    model.fit(x_train, y_train)

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
    out_txt = os.path.join(out_dir, "rf_output.txt")
    out_json = os.path.join(out_dir, "rf_reports.json")

    # ===== TASK 1: Aspect Detection (includes OTHERS) =====
    aspect_model = build_rf_pipeline()
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
    sent_model = build_rf_pipeline()
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
    lines.append("Random Forest baseline (DEFAULT predict) - UIT-ViSFD")
    lines.append(f"Run time: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"DATA_DIR: {DATA_DIR}")
    lines.append(f"Files: Train={os.path.basename(TRAIN_CSV)}, Dev={os.path.basename(DEV_CSV)}, Test={os.path.basename(TEST_CSV)}")
    lines.append(f"Vectorizer: BoW (CountVectorizer) ngram={NGRAM_RANGE}, max_features={MAX_FEATURES}")
    lines.append(
        "Model: OneVsRest(RandomForestClassifier("
        f"n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, "
        f"min_samples_split={MIN_SAMPLES_SPLIT}, min_samples_leaf={MIN_SAMPLES_LEAF}, "
        f"random_state={RANDOM_STATE}))"
    )
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
            "vectorizer": "CountVectorizer",
            "ngram_range": NGRAM_RANGE,
            "max_features": MAX_FEATURES,
            "rf_n_estimators": N_ESTIMATORS,
            "rf_max_depth": MAX_DEPTH,
            "rf_min_samples_split": MIN_SAMPLES_SPLIT,
            "rf_min_samples_leaf": MIN_SAMPLES_LEAF,
            "rf_random_state": RANDOM_STATE,
            "predict_mode": "default_predict_no_threshold_tuning",
        },
        "aspect": aspect_res,
        "sentiment": sent_res,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\nSaved outputs next to random_forest.py:")
    print(f"- {out_txt}")
    print(f"- {out_json}")


if __name__ == "__main__":
    main()
