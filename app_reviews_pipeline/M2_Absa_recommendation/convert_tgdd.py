import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
input_path = BASE_DIR / "data" / "tgdd_reviews.csv"
output_path = BASE_DIR / "data" / "tgdd_reviews_converted.csv"

def convert_csv(input_path: Path, output_path: Path):
    if not input_path.exists():
        raise FileNotFoundError(f"Không thấy input file: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)  # tạo folder data/ nếu cần

    with input_path.open("r", encoding="utf-8-sig", newline="") as f_in, \
         output_path.open("w", encoding="utf-8", newline="") as f_out:

        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=["review_id", "review_text", "rating"])
        writer.writeheader()

        count = 0
        for row in reader:
            review_id = (row.get("review_id") or "").strip()
            review_text = (row.get("sentence") or "").strip()
            rating_raw = (row.get("sentiments") or "").strip()

            try:
                rating = int(float(rating_raw)) if rating_raw != "" else ""
            except ValueError:
                rating = ""

            writer.writerow({
                "review_id": review_id,
                "review_text": review_text,
                "rating": rating
            })
            count += 1

    print(f"✅ Wrote {count} rows to: {output_path}")

if __name__ == "__main__":
    convert_csv(input_path, output_path)
