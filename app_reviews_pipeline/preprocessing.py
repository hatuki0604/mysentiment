"""
Step 1 — Data Collection & Preprocessing
Chú thích từng phần
"""

# ===========================
# Imports & Configuration
# ===========================
from pathlib import Path
import re
import json
import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm
import logging
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

# ===========================
# Language Utilities
# ===========================
def is_english(text: str) -> bool:
    """Check if text is in English"""
    if not isinstance(text, str) or not text.strip():
        return False
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False
    
def is_vietnamese(text: str) -> bool:
    """Check if text is in Vietnamese - Có dấu"""
    if not isinstance(text, str) or not text.strip():
        return False
    vi_chars = regex.findall(r"[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệ"
                             r"ìíỉĩịòóỏõọôốồổỗộơớờởỡợ"
                             r"ùúủũụưứừửữựỳýỷỹỵđ]", text)
    return len(vi_chars) > 0

# ===========================
# Cache & Logging Setup
# ===========================
CACHE_DIR = Path("./preprocessing_cache")
CACHE_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)

# ===========================
# Source Configuration
# ===========================
@dataclass
class SourceConfig:
    """Configuration for data sources"""
    path: Path
    text_column: str
    rating_column: Optional[str] = None
    name: Optional[str] = None
    min_text_length: int = 5
    def __post_init__(self):
        self.name = self.name or self.path.stem
    language: str = "en" 

# ===========================
# CSV Reading
# ===========================
def load_reviews_csv(
        filepath: Path,
        text_column: str,
        nrows: Optional[int] = None,
        sample_n: Optional[int] = None,
        random_state: int = 42
    ) -> pd.DataFrame:
    """Load reviews with optional sampling (no row dropping here)."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath, nrows=nrows)
    if text_column not in df.columns:
        available_cols = ", ".join(df.columns)
        raise KeyError(f"Column '{text_column}' not found. Available columns: {available_cols}")
    if sample_n is not None:
        df = df.sample(n=sample_n, random_state=random_state)
    return df.reset_index(drop=True)

# ===========================
# Text Cleaning
# ===========================
def clean_text(text: str) -> str:
    """Enhanced text cleaning aligned with RAG QA approach"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^\w\s.,!?]', ' ', text)  # Keep some punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text.split()) < 2:
        return ""
    return text

def clean_text_vi(text: str) -> str:
    """Làm sạch văn bản tiếng Việt (giữ dấu, xóa ký tự đặc biệt)"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", " ", text)
    text = re.sub(r"[^\p{L}\p{N}\s.,!?]", " ", text)  # Giữ lại chữ có dấu, số và một số dấu câu
    text = re.sub(r"\s+", " ", text).strip()
    # Tuỳ chọn: tách từ
    text = word_tokenize(text, format="text")
    return text

# ===========================
# Review ID Normalization
# ===========================
def ensure_unique_review_ids(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Preserve existing review_id values.
    - If review_id column is missing: generate fresh IDs.
    - If some IDs are blank/NaN: fill just those.
    - If duplicates remain: append a stable '-n' suffix.
    """
    df = df.copy()
    if "review_id" not in df.columns:
        df["review_id"] = [f"{source_name}_{i:07d}" for i in range(len(df))]
        return df
    mask = df["review_id"].isna() | (df["review_id"].astype(str).str.strip() == "")
    if mask.any():
        fills = [f"{source_name}_{i:07d}" for i in range(mask.sum())]
        df.loc[mask, "review_id"] = fills
    dup_idx = df.groupby("review_id").cumcount()
    has_dup = dup_idx > 0
    if has_dup.any():
        df.loc[has_dup, "review_id"] = (
            df.loc[has_dup, "review_id"].astype(str) + "-" + dup_idx[has_dup].astype(str)
        )
    return df

def _pick_id_column(df: pd.DataFrame) -> Optional[str]:
    """Return the first matching id column name (case-insensitive), else None."""
    candidates = ["review_id", "reviewId", "sentence_id", "id"]
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

# ===========================
# Review Normalization
# ===========================
def normalize_reviews(df: pd.DataFrame, source_info: SourceConfig) -> pd.DataFrame:
    """Normalize reviews with stable IDs and minimal metadata (text + optional rating)."""
    initial_count = len(df)
    df = df.copy()
    id_col = _pick_id_column(df)
    if id_col:
        logging.info(f"Using ID column: {id_col}")
        df["review_id"] = df[id_col].astype(str)
    else:
        logging.info("No ID column found; using row index as review_id")
        df["review_id"] = df.index.astype(str)
    def col(s: Optional[str]) -> pd.Series:
        if s and s in df.columns:
            return df[s]
        return pd.Series([None] * len(df), index=df.index)
    normalized = pd.DataFrame({
        "review_id": df["review_id"],
        "review_text": df[source_info.text_column],
        "rating": col(source_info.rating_column),
    })
    normalized = normalized[normalized["review_text"].notna()]
    normalized["review_text"] = normalized["review_text"].map(clean_text)
    normalized = normalized[normalized["review_text"].str.len() >= source_info.min_text_length]

    # logging.info("Filtering non-English reviews...")
    # is_english_mask = pd.Series(
    #     [is_english(text) for text in tqdm(normalized["review_text"], desc="Detecting language")],
    #     index=normalized.index
    # )
    # normalized = normalized[is_english_mask]

    # Language filtering

    if getattr(source_info, "language", "en") == "en": #hatuki
        logging.info("Filtering non-English reviews...")
        is_lang_mask = pd.Series(
            [is_english(text) for text in tqdm(normalized["review_text"], desc="Detecting English")],
            index=normalized.index
        )
        normalized = normalized[is_lang_mask]
    else:
        logging.info(f"Skipping language filter for language={source_info.language}") #hatuki


    normalized = normalized.drop_duplicates(subset=["review_text"])
    normalized = ensure_unique_review_ids(normalized, source_info.name)
    lang_info = "N/A"
    if getattr(source_info, "language", "en") == "en":
        lang_info = str(int(is_lang_mask.sum()))

    logging.info(f"""
        Source: {source_info.name}
        Initial reviews: {initial_count}
        After language filtering: {lang_info}
        Final reviews: {len(normalized)}
        Removed: {initial_count - len(normalized)}
    """)
    return normalized.reset_index(drop=True), initial_count

# ===========================
# Data Validation & Selection
# ===========================
def validate_data(df: pd.DataFrame) -> bool:
    checks = {
        "Has reviews": len(df) > 0,
        "No missing text": df["review_text"].notna().all(),
        "Text length": (df["review_text"].str.len() >= 5).all(),
        "Valid ratings": (df["rating"].dropna().between(1, 5).all()) if "rating" in df.columns else True,
        "Unique IDs": df["review_id"].nunique() == len(df),
        "Text quality": df["review_text"].str.split().str.len().ge(2).all(),
    }
    for check_name, passed in checks.items():
        if not passed:
            logging.error(f"Data validation failed: {check_name}")
            return False
    return True

def get_user_selection(available_sources: Dict[str, SourceConfig]) -> Dict[str, SourceConfig]:
    print("\nAvailable datasets:")
    for idx, (name, _cfg) in enumerate(available_sources.items(), 1):
        print(f"{idx}. {name}")
    print("\nEnter the numbers of datasets you want to process (comma-separated)")
    print("Example: 1,2,3,4,5 or press Enter for all datasets")
    selection = input("Your selection: ").strip()
    if not selection:
        return available_sources
    try:
        indices = [int(i.strip()) for i in selection.split(",")]
        return dict(list(available_sources.items())[i-1] for i in indices)
    except (ValueError, IndexError):
        print("Invalid selection. Using all datasets.")
        return available_sources

def get_data_stats(df: pd.DataFrame, initial_count: int) -> Dict:
    word_lengths = df["review_text"].str.split().str.len()
    min_w = int(word_lengths.min()) if len(word_lengths) else 0
    max_w = int(word_lengths.max()) if len(word_lengths) else 0
    avg_w = float(word_lengths.mean()) if len(word_lengths) else 0.0
    rating_dist = None
    if "rating" in df.columns:
        vc = df["rating"].dropna().value_counts()
        rating_dist = {int(k): int(v) for k, v in vc.items()}
    return {
        "total_reviews": {
            "before_preprocessing": int(initial_count),
            "after_preprocessing": int(len(df)),
        },
        "unique_reviews": int(df["review_text"].nunique()),
        "avg_text_length": float(df["review_text"].str.len().mean() if len(df) else 0.0),
        "word_count_stats": {
            "min": min_w,
            "max": max_w,
            "avg": avg_w,
        },
        "rating_distribution": rating_dist,
    }

# ===========================
# Main Execution
# ===========================
def main():
    """Main pipeline with error handling and reporting (no date/app)"""
    root_dir = Path(__file__).resolve().parent.parent   # .../methods
    input_dir = root_dir / "data" / "raw"               # .../methods/data/raw
    output_dir = root_dir / "data" / "processed"        # .../methods/data/processed
    print(f"[preprocessing] input_dir = {input_dir}")
    print(f"[preprocessing] output_dir = {output_dir}")
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        'spotify': SourceConfig(
            path=input_dir / 'spotify_reviews.csv',
            text_column='content',
            rating_column='score'
        ),
        'google_play': SourceConfig(
            path=input_dir / 'google_play_reviews.csv',
            text_column='content',
            rating_column='score'
        ),
        'aware': SourceConfig(
            path=input_dir / 'AWARE_Comprehensive.csv',
            text_column='sentence',
            rating_column='rating'
        ),
        'uit': SourceConfig(
            path=input_dir / 'uit.csv',  # file raw 2 cột content, score
            text_column='comment',
            rating_column='n_star',
            name='uit',
            language='vi'  # ⚡ quan trọng: bỏ lọc tiếng Anh
        ),
         'uit_all': SourceConfig(
            path=input_dir / 'uit_all.csv',  # file raw 2 cột content, score
            text_column='comment',
            rating_column='n_star',
            name='uit_all',
            language='vi'  # ⚡ quan trọng: bỏ lọc tiếng Anh
    )
    }
    stats = {}
    selected_sources = get_user_selection(sources)
    for source_name, config in tqdm(selected_sources.items(), desc="Processing sources"):
        try:
            logging.info(f"Processing {source_name}...")
            df = load_reviews_csv(config.path, config.text_column)
            clean_df, initial_count = normalize_reviews(df, config)

            # Convert rating column to numeric
            if "rating" in clean_df.columns:
                clean_df["rating"] = pd.to_numeric(clean_df["rating"], errors="coerce")

            if validate_data(clean_df):
                source_output = output_dir / f"{source_name}_clean.csv"
                clean_df.to_csv(source_output, index=False)
                stats[source_name] = get_data_stats(clean_df, initial_count)
                with open(output_dir / f"{source_name}_stats.json", 'w') as f:
                    json.dump(stats[source_name], f, indent=2, default=str)
                logging.info(f"Successfully processed and saved {source_name}: {len(clean_df)} reviews")
            else:
                logging.error(f"Data validation failed for {source_name}")
        except Exception as e:
            logging.exception(f"Error processing {source_name}")
            continue
    if not stats:
        logging.error("No datasets were successfully processed")

if __name__ == "__main__":
    main()
