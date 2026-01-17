import re
import requests

def parse_shopee_url(url):
    """
    Lấy shop_id và item_id từ URL dạng:
    https://shopee.vn/abc-i.SHOPID.ITEMID
    """
    match = re.search(r"i\.(\d+)\.(\d+)", url)
    if match:
        return match.group(1), match.group(2)
    return None, None

def get_product_name(shop_id, item_id):
    url = "https://shopee.vn/api/v4/item/get"
    params = {"itemid": item_id, "shopid": shop_id}

    headers = {
        "User-Agent": "Mozilla/5.0",
    }

    res = requests.get(url, params=params, headers=headers).json()
    return res["data"]["name"] if "data" in res else None


if __name__ == "__main__":
    test_url = "https://shopee.vn/...-i.123456.987654"
    shop_id, item_id = parse_shopee_url(test_url)
    print(shop_id, item_id)
    print(get_product_name(shop_id, item_id))
