import pandas as pd

INPUT_FILE = "/Users/hatrungkien/my-sentiment/data/raw/tgdd_reviews.csv"        # file gốc bạn crawl ra
OUTPUT_FILE = "processed_tgdd_reviews.csv"  # file sau preprocess


def preprocess_reviews():

    df = pd.read_csv(INPUT_FILE)

    print(f"📌 File gốc có {len(df)} dòng")

    # ======================
    # 1) Tạo review_text
    # ======================
    df["review_text"] = df["model"].astype(str) + " - " + df["sentence"].astype(str)

    # ======================
    # 2) Tạo rating
    # ======================
    df["rating"] = df["sentiments"]

    # ======================
    # 3) Tạo review_id tự tăng
    # ======================
    df = df.reset_index(drop=True)
    df["review_id"] = df.index  # số từ 0 →

    # ======================
    # 4) Chọn cột cần thiết
    # ======================
    df_final = df[["review_id", "review_text", "rating"]]

    # ======================
    # 5) Lưu file mới
    # ======================
    df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"🎉 DONE! Đã lưu file {OUTPUT_FILE}")
    print(df_final.head())


if __name__ == "__main__":
    preprocess_reviews()
