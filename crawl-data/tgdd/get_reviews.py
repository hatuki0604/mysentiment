import time
import uuid
import pandas as pd
from bs4 import BeautifulSoup
import os

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


# =======================
# Tạo driver
# =======================
def make_driver(headless=False):
    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1400,900")
    if headless:
        options.add_argument("--headless=new")

    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


# =======================
# Crawl review 1 sản phẩm
# =======================
def crawl_reviews(product_url):

    driver = make_driver()
    review_url = product_url.rstrip("/") + "/danh-gia"
    print("\n📦 Crawl:", review_url)

    driver.get(review_url)
    time.sleep(3)

    # Scroll để load JS
    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
    time.sleep(1.5)

    html = driver.page_source
    driver.quit()

    soup = BeautifulSoup(html, "lxml")

    # Lấy MODEL đúng
    model_tag = soup.select_one(".boxrate__top .box-product .content h3")
    if not model_tag:
        model_tag = soup.select_one(".breadcrumb-rating li a")
    model = model_tag.get_text(strip=True) if model_tag else "Unknown"

    reviews = []

    for li in soup.select("ul.comment-list li.par"):

        stars = len(li.select("div.cmt-top-star i.iconcmt-starbuy"))

        txt = li.select_one("div.cmt-content p.cmt-txt")
        sentence = txt.get_text(strip=True) if txt else None

        used_tag = li.select_one("span.cmtd")
        used_days = used_tag.get_text(strip=True) if used_tag else None

        if sentence:
            reviews.append({
                "review_id": li.get("id") or str(uuid.uuid4()),
                "shop": "thegioididong",
                "model": model,
                "sentence": sentence,
                "sentiments": stars,
                "used_days": used_days,
                "product_url": product_url
            })

    print(f"   ✔ Lấy được {len(reviews)} review.")
    return reviews


# =======================
# APPEND CSV SAFELY
# =======================
def append_to_csv(data, csv_path="product_reviews.csv"):
    df = pd.DataFrame(data)

    if not os.path.exists(csv_path):  # nếu chưa có file → tạo mới
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    else:  # nếu có file → append
        df.to_csv(csv_path, mode="a", index=False, header=False, encoding="utf-8-sig")


# =======================
# MAIN
# =======================
if __name__ == "__main__":

    # Danh sách sản phẩm
    df_products = pd.read_csv("products.csv")
    product_urls = df_products["product_url"].tolist()

    # File lưu sản phẩm đã crawl
    DONE_FILE = "done_products.txt"
    done = set()

    # Load danh sách sản phẩm đã crawl nếu file tồn tại
    if os.path.exists(DONE_FILE):
        with open(DONE_FILE, "r") as f:
            done = set(line.strip() for line in f.readlines())

    print(f"📌 Tổng sản phẩm: {len(product_urls)}")
    print(f"📌 Đã crawl trước đó: {len(done)} sản phẩm")
    print(f"📌 Còn lại: {len(product_urls) - len(done)} sản phẩm\n")

    for url in product_urls:

        if url in done:
            print(f"⏭ Bỏ qua (đã crawl): {url}")
            continue

        try:
            reviews = crawl_reviews(url)

            if len(reviews) > 0:
                append_to_csv(reviews, "tgdd_reviews.csv")

            # ghi vào done list
            with open(DONE_FILE, "a") as f:
                f.write(url + "\n")

        except Exception as e:
            print("⚠️ Lỗi:", e)

        time.sleep(1.5)

    print("\n🎉 HOÀN THÀNH! Review đã được append vào tgdd_reviews.csv")
