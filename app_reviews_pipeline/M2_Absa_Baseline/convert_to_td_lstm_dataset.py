import pandas as pd
import re

INPUT = "/Users/hatrungkien/my-sentiment/data/raw/UIT-ViSFD/Train.csv"          # file bạn đang có
OUTPUT = "train_td_lstm.csv"     # file chuẩn để train TD-LSTM


def parse_label_string(s):
    """
    Chuyển chuỗi: {BATTERY#Negative};{GENERAL#Positive};
    → list of (aspect, sentiment)
    """
    pairs = []
    items = re.findall(r"\{(.*?)\}", s)  # bắt tất cả bên trong {}
    for item in items:
        if "#" in item:
            asp, sent = item.split("#")
            pairs.append((asp.strip(), sent.strip()))
    return pairs


def convert():
    df = pd.read_csv(INPUT)

    rows = []

    for _, row in df.iterrows():
        comment = str(row["comment"])
        label_str = str(row["label"])

        pairs = parse_label_string(label_str)

        for aspect, sentiment in pairs:
            # Nếu sentiment trống (VD {OTHERS})
            if sentiment == "" or sentiment.lower() == "others":
                continue

            rows.append({
                "sentence": comment,
                "aspect": aspect,
                "label": sentiment
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT, index=False)
    print(f"✔ Saved TD-LSTM dataset → {OUTPUT}")
    print("Rows:", len(out_df))


if __name__ == "__main__":
    convert()
