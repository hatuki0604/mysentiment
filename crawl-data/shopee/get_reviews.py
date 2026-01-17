import requests
import time

def get_reviews(shop_id, item_id, limit=50):
    offset = 0
    all_reviews = []

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": f"https://shopee.vn/product/{shop_id}/{item_id}"
    }

    while True:
        url = "https://shopee.vn/api/v2/item/get_ratings"
        params = {
            "filter": 0,
            "flag": 1,
            "itemid": item_id,
            "limit": limit,
            "offset": offset,
            "shopid": shop_id,
            "type": 0,
        }

        res = requests.get(url, params=params, headers=headers).json()
        ratings = res.get("data", {}).get("ratings", [])

        if not ratings:
            break

        for r in ratings:
            all_reviews.append({
                "comment": r.get("comment", ""),
                "rating": r.get("rating_star", 0),
            })

        offset += limit
        time.sleep(0.5)

    return all_reviews
