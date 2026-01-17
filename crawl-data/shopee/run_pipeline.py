import csv
from get_product_name import parse_shopee_url, get_product_name
from get_reviews import get_reviews

INPUT_FILE = "products.csv"
OUTPUT_FILE = "shopee_reviews.csv"

def run():
    with open(INPUT_FILE, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        products = list(reader)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["url", "shop_id", "item_id", "product_name", "rating", "comment"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in products:
            url = row["url"]
            shop_id, item_id = parse_shopee_url(url)

            if not shop_id:
                print("URL lỗi:", url)
                continue

            product_name = get_product_name(shop_id, item_id)

            reviews = get_reviews(shop_id, item_id)
            for r in reviews:
                writer.writerow({
                    "url": url,
                    "shop_id": shop_id,
                    "item_id": item_id,
                    "product_name": product_name,
                    "rating": r["rating"],
                    "comment": r["comment"],
                })

            print(f"Done {product_name}")

if __name__ == "__main__":
    run()
