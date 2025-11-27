# ===========================
# Imports & Configuration
# ===========================
import sys
from pathlib import Path
from typing import Tuple, List, Dict

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# --- Config: dùng Excel VnEmoLex.xlsx
LEXICON_XLSX = Path("/Users/hatrungkien/my-sentiment/CopyOfVnEmoLex.xlsx")  # đường dẫn từ điển cảm xúc VN (Excel)
SHEET_NAME = 0                        # 0 = sheet đầu tiên; đổi tên/ chỉ số nếu cần

EMO_COLUMNS = [
    "Positive", "Negative",
    "Anger", "Anticipation", "Disgust", "Fear", "Joy", "Sadness", "Surprise", "Trust"
]
# POSITIVE_EMOS = ["Positive", "Joy", "Trust", "Anticipation", "Surprise"]
# NEGATIVE_EMOS = ["Negative", "Anger", "Disgust", "Fear", "Sadness"]

POSITIVE_EMOS = ["Positive"]
NEGATIVE_EMOS = ["Negative"]

# --- thresholds: compound [-1..1] -> sao 1..5
STAR_THRESHOLDS = (-0.1, 0, 0.15, 0.25)  # <=-0.6→1, <=-0.2→2, <=0.2→3, <=0.6→4, else 5
GENERATE_HEATMAP = True
TOP_PAD_FRAC = 0.15

# --- shared menus (choose_dataset / choose_sample_size)
THIS_ROOT = Path(__file__).resolve().parents[1]  # .../app_reviews_pipeline
if str(THIS_ROOT) not in sys.path:
    sys.path.insert(0, str(THIS_ROOT))
from user_selection import choose_dataset, choose_sample_size  # type: ignore


# ===========================
# Lexicon Loader & Scorer (Excel)
# ===========================

class VnEmoLexScorer:
    """
    Tính điểm sentiment dựa trên từ điển VnEmoLex (Excel).
    - Với mỗi review: tìm tất cả mục 'Vietnamese' (từ/cụm từ) xuất hiện trong text.
    - Tính trung bình từng chỉ số (10 cột EMO_COLUMNS) trên các mục khớp.
    - compound = mean(positives) - mean(negatives) ∈ [-1, 1].
    """
    def __init__(self, xlsx_path: Path, sheet=SHEET_NAME):
        if not xlsx_path.exists():
            raise FileNotFoundError(f"Lexicon not found at {xlsx_path}")

        # Đọc Excel (pandas tự xử lý unicode)
        try:
            df = pd.read_excel(xlsx_path, sheet_name=sheet)
        except Exception as e:
            raise RuntimeError(f"Cannot read Excel '{xlsx_path}': {e}")

        # Kiểm tra cột bắt buộc
        missing = [c for c in ["Vietnamese"] + EMO_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in lexicon: {missing}")

        # Chuẩn hóa
        df = df[["Vietnamese"] + EMO_COLUMNS].copy()
        df["Vietnamese"] = df["Vietnamese"].astype(str).str.strip().str.lower()

        for c in EMO_COLUMNS:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(0, 1).astype(int)

        # Loại trùng
        df = df.drop_duplicates(subset=["Vietnamese"]).reset_index(drop=True)

        # Phân tách theo số từ
        df["is_multi"] = df["Vietnamese"].str.contains(r"\s")
        self.df_single = df[~df["is_multi"]].reset_index(drop=True)
        self.df_multi  = df[df["is_multi"]].reset_index(drop=True)

        # Regex word-boundary cho đơn từ
        if len(self.df_single):
            escaped = [re.escape(w) for w in self.df_single["Vietnamese"].tolist()]
            pattern = r"\b(" + "|".join(sorted(escaped, key=len, reverse=True)) + r")\b"
            self.single_re = re.compile(pattern, flags=re.IGNORECASE)
        else:
            self.single_re = None

        # Multi-word: dò substring không phân biệt hoa/thường
        self.multi_terms = self.df_multi["Vietnamese"].tolist()

    def score_text(self, text: str) -> Dict[str, float]:
        """
        Kết quả:
          'means': dict trung bình cho 10 cảm xúc (0..1),
          'compound': float [-1, 1],
          'n_matches': số mục khớp.
        """
        t = (text or "").lower()
        matched_rows = []

        # 1) match multi-words (substring)
        for term, row in zip(self.df_multi["Vietnamese"], self.df_multi[EMO_COLUMNS].values):
            if term in t:
                matched_rows.append(row)

        # 2) match single-words (word boundary)
        if self.single_re is not None:
            for m in self.single_re.findall(t):
                row = self.df_single.loc[self.df_single["Vietnamese"] == m.lower(), EMO_COLUMNS]
                if not row.empty:
                    matched_rows.append(row.values[0])

        if not matched_rows:
            means = {emo: 0.0 for emo in EMO_COLUMNS}
            return {"means": means, "compound": 0.0, "n_matches": 0}

        mat = np.vstack(matched_rows).astype(float)   # (k, 10)
        col_means = mat.mean(axis=0)
        means = {emo: float(val) for emo, val in zip(EMO_COLUMNS, col_means)}

        pos_mean = float(np.mean([means[e] for e in POSITIVE_EMOS])) if POSITIVE_EMOS else 0.0
        neg_mean = float(np.mean([means[e] for e in NEGATIVE_EMOS])) if NEGATIVE_EMOS else 0.0
        compound = pos_mean - neg_mean
        compound = max(-1.0, min(1.0, compound))      # clamp

        return {"means": means, "compound": compound, "n_matches": len(matched_rows)}


# ===========================
# Utility & Plotting
# ===========================

def compound_to_star(v: float) -> int:
    """Map compound [-1..1] to star rating (1-5) using STAR_THRESHOLDS."""
    a, b, c, d = STAR_THRESHOLDS
    if v <= a: return 1
    if v <= b: return 2
    if v <= c: return 3
    if v <= d: return 4
    return 5

def _counts_1to5(series: pd.Series) -> np.ndarray:
    vc = series.value_counts().reindex([1, 2, 3, 4, 5]).fillna(0).astype(int)
    return vc.values

def _annotate_counts(ax, x, heights, total, y_max):
    offset = max(6, y_max * 0.02)
    for xi, h in zip(x, heights):
        pct = (h / max(total, 1)) * 100.0
        ax.text(xi, h + offset, f"{h}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

def _make_bar_chart(ds: str,
                    actual_counts: np.ndarray,
                    pred_counts: np.ndarray,
                    total_n: int,
                    out_png: Path,
                    out_pdf: Path):
    labels = np.array([1, 2, 3, 4, 5])
    x = np.arange(len(labels), dtype=float)
    width = 0.35

    tallest = max(int(actual_counts.max()), int(pred_counts.max()))
    y_max = int(np.ceil(tallest * (1.0 + TOP_PAD_FRAC)))

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.bar(x - width/2, actual_counts, width, label="Actual ratings")
    ax.bar(x + width/2, pred_counts,  width, label="Sentiment ratings (VnEmoLex)")

    ax.set_title(f"{ds.capitalize()} App Reviews: Actual vs. Sentiment Ratings (N={total_n})", pad=14)
    ax.set_xlabel("Rating bucket")
    ax.set_ylabel("Review count")
    ax.set_xticks(x, labels)

    ax.set_ylim(0, y_max)
    ax.margins(y=0.02)
    ax.legend()

    _annotate_counts(ax, x - width/2, actual_counts, actual_counts.sum(), y_max)
    _annotate_counts(ax, x + width/2, pred_counts,  pred_counts.sum(),  y_max)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

def _make_heatmap(ds: str,
                ctab: pd.DataFrame,
                out_png: Path,
                out_pdf: Path):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(ctab.values, cmap="Blues")

    ax.set_title(f"{ds.capitalize()} — User Stars vs. VnEmoLex Stars (Counts)", pad=12)
    ax.set_xlabel("VnEmoLex stars")
    ax.set_ylabel("Actual stars")
    ax.set_xticks(range(5), [1,2,3,4,5])
    ax.set_yticks(range(5), [1,2,3,4,5])

    for i in range(5):
        for j in range(5):
            ax.text(j, i, int(ctab.values[i, j]), ha="center", va="center", fontsize=9, color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ===========================
# Main Pipeline
# ===========================

def main():
    # 1) Pick dataset và load
    ds, csv_path, topic = choose_dataset()
    df = pd.read_csv(csv_path)  # nếu file dataset không UTF-8, đổi sang hàm đọc linh hoạt của bạn

    # Detect star column
    star_col = next((c for c in ["score", "rating", "stars", "user_rating"] if c in df.columns), None)
    if not star_col:
        raise ValueError(
            f"No star/rating column found in {csv_path}. "
            f"Expected one of: score/rating/stars/user_rating"
        )
    if "review_text" not in df.columns:
        raise ValueError("Expected a 'review_text' column in processed CSV.")

    # 2) Optional sampling
    n = choose_sample_size(len(df))
    if n:
        df = df.sample(n=n, random_state=42).reset_index(drop=True)

    # 3) VnEmoLex → compound ∈ [-1,1] → stars 1..5
    scorer = VnEmoLexScorer(LEXICON_XLSX, sheet=SHEET_NAME)

    def _score_row(s: str) -> Tuple[float, int, int]:
        """
        return (compound, stars, n_matches)
        """
        res = scorer.score_text(str(s))
        comp = res["compound"]                # [-1, 1]
        stars = compound_to_star(comp)
        return comp, stars, res["n_matches"]

    scored = df["review_text"].map(_score_row)
    df[["compound", "pred_stars", "lex_matches"]] = pd.DataFrame(scored.tolist(), index=df.index)

    # Normalize user stars: numeric, clip 1..5
    actual_stars = pd.to_numeric(df[star_col], errors="coerce").round().clip(1, 5).astype("Int64")
    pred_stars = pd.to_numeric(df["pred_stars"], errors="coerce").clip(1, 5).astype("Int64")
    df["pred_stars"] = pred_stars
    df["actual_stars"] = actual_stars
    df["star_discrepancy"] = (actual_stars - pred_stars).abs()

    # 4) Exports
    out_dir = Path(f"outputs/discrepancy/{ds}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = out_dir / f"{ds}_vnemo_discrepancy.csv"
    df_out = df.copy()
    df_out.to_csv(rows_csv, index=False)

    # Summary discrepancy
    summary = (
        df_out["star_discrepancy"]
        .value_counts()
        .sort_index()
        .rename_axis("abs_diff")
        .reset_index(name="count")
    )
    summary_csv = out_dir / f"{ds}_discrepancy_summary.csv"
    summary.to_csv(summary_csv, index=False)

    # 5) Console quick stats
    mad = float(df_out["star_discrepancy"].mean())
    prop_ge1 = float((df_out["star_discrepancy"] >= 1).mean())
    prop_ge2 = float((df_out["star_discrepancy"] >= 2).mean())
    print(
        f"\nSummary ({ds}, N={len(df_out)}): "
        f"MAD={mad:.3f}, |Δ|≥1: {prop_ge1*100:.1f}%, |Δ|≥2: {prop_ge2*100:.1f}%"
    )

    # 6) Plot: bar chart
    actual_counts = _counts_1to5(actual_stars.dropna().astype(int))
    pred_counts = _counts_1to5(pred_stars.dropna().astype(int))
    bar_png = out_dir / f"{ds}_user_vs_vnemo_bars.png"
    bar_pdf = out_dir / f"{ds}_user_vs_vnemo_bars.pdf"
    _make_bar_chart(ds, actual_counts, pred_counts, len(df_out), bar_png, bar_pdf)

    # 7) Optional heatmap
    if GENERATE_HEATMAP:
        ctab = (
            pd.crosstab(actual_stars, pred_stars)
            .reindex(index=[1,2,3,4,5], columns=[1,2,3,4,5])
            .fillna(0)
            .astype(int)
        )
        heat_png = out_dir / f"{ds}_user_vs_vnemo_heatmap.png"
        heat_pdf = out_dir / f"{ds}_user_vs_vnemo_heatmap.pdf"
        _make_heatmap(ds, ctab, heat_png, heat_pdf)

    print("\nSaved:")
    print(f"  {rows_csv}")
    print(f"  {summary_csv}")
    print(f"  {bar_png}")
    if GENERATE_HEATMAP:
        print(f"  {heat_png}")


# ===========================
# Entry Point
# ===========================

if __name__ == "__main__":
    main()
