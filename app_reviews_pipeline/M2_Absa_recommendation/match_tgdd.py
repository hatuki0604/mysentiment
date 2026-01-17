import csv
from pathlib import Path

FILE1 = Path("/Users/hatrungkien/my-sentiment/app_reviews_pipeline/outputs/absa/processed_tgdd_reviews/processed_tgdd_reviews_absa_gpt-4o-mini_2429.csv")
FILE2 = Path("/Users/hatrungkien/my-sentiment/app_reviews_pipeline/M2_Absa_recommendation/data/tgdd_reviews.csv")
OUT   = Path("/Users/hatrungkien/my-sentiment/app_reviews_pipeline/outputs/absa/processed_tgdd_reviews/processed_tgdd_reviews_absa_gpt-4o-mini_2429_enriched.csv")

ADD_COLS = ["model", "rating", "used_days"]  # bỏ product_url

OUT_FIELDS = [
    "review_id",
    "sent_idx",
    "model",
    "sentence",
    "rating",
    "used_days",
    "aspects",
    "sentiments",
]

def enrich_by_order(file1: Path, file2: Path, out: Path):
    if not file1.exists():
        raise FileNotFoundError(f"Missing file1: {file1}")
    if not file2.exists():
        raise FileNotFoundError(f"Missing file2: {file2}")

    out.parent.mkdir(parents=True, exist_ok=True)

    with file1.open("r", encoding="utf-8-sig", newline="") as f1, \
         file2.open("r", encoding="utf-8-sig", newline="") as f2, \
         out.open("w", encoding="utf-8", newline="") as fout:

        r1 = csv.DictReader(f1)
        r2 = csv.DictReader(f2)

        if not r1.fieldnames or not r2.fieldnames:
            raise ValueError("One of the files is missing header/fieldnames")

        required_f1 = {"review_id", "sent_idx", "sentence", "aspects", "sentiments"}
        missing_f1 = required_f1 - set(r1.fieldnames)
        if missing_f1:
            raise ValueError(f"file1 missing columns: {sorted(missing_f1)}")

        missing_f2 = set(ADD_COLS) - set(r2.fieldnames)
        if missing_f2:
            raise ValueError(f"file2 missing columns: {sorted(missing_f2)}")

        w = csv.DictWriter(fout, fieldnames=OUT_FIELDS)
        w.writeheader()

        i = 0
        mismatch = 0

        while True:
            try:
                row1 = next(r1)
            except StopIteration:
                row1 = None
            try:
                row2 = next(r2)
            except StopIteration:
                row2 = None

            if row1 is None and row2 is None:
                break

            if row1 is None or row2 is None:
                raise ValueError(
                    f"Row count mismatch at line {i+2} (excluding header). "
                    f"file1_row_exists={row1 is not None}, file2_row_exists={row2 is not None}"
                )

            rid1 = (row1.get("review_id") or "").strip()
            rid2 = (row2.get("review_id") or "").strip()
            if rid1 and rid2 and rid1 != rid2:
                mismatch += 1

            for c in ADD_COLS:
                row1[c] = (row2.get(c) or "").strip()

            w.writerow({k: row1.get(k, "") for k in OUT_FIELDS})
            i += 1

        print(f"✅ Done: {out}")
        print(f"Rows written: {i}")
        if mismatch:
            print(f"⚠️ Warning: {mismatch} rows have different review_id between file1 and file2 (order may be off).")

if __name__ == "__main__":
    enrich_by_order(FILE1, FILE2, OUT)
