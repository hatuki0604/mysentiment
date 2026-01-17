import time
import os
import pandas as pd
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

keyword = "dong-ho-thong-minh"

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


def get_all_products():
    url = f"https://www.thegioididong.com/{keyword}"
    print("➡️ Đang mở trang danh sách sản phẩm...")
    driver = make_driver()
    driver.get(url)
    time.sleep(3)

    # ===== 1) LIÊN TỤC NHẤN "Xem thêm" =====
    while True:
        try:
            load_more_btn = driver.find_element(By.CSS_SELECTOR, "a[href='javascript:;'] strong.see-more-btn")

            driver.execute_script("arguments[0].scrollIntoView(true);", load_more_btn)
            time.sleep(1)
            print("➡️ Click nút Xem thêm...")
            load_more_btn.click()
            time.sleep(2.5)
        except:
            print("✔ Không còn nút 'Xem thêm' nữa.")
            break

    # ===== 2) Scroll thêm lần cuối =====
    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
    time.sleep(2)

    html = driver.page_source
    driver.quit()

    soup = BeautifulSoup(html, "lxml")

    items = soup.select(".item a.main-contain")

    results = []

    for item in items:
        name = item.get("data-name") or item.get_text(strip=True)
        href = item.get("href")

        if href and href.startswith(f"/{keyword}/"):
            full_url = "https://www.thegioididong.com" + href
            results.append({
                "shop": "thegioididong",
                "model": name,
                "product_url": full_url
            })

    print(f"📌 Tổng sản phẩm lấy được: {len(results)}")
    return results


if __name__ == "__main__":
    new_products = get_all_products()
    df_new = pd.DataFrame(new_products)

    csv_path = "products.csv"

    if os.path.exists(csv_path):
        # Đã có file → đọc file cũ, ghép thêm rồi bỏ trùng
        df_old = pd.read_csv(csv_path)

        df_all = pd.concat([df_old, df_new], ignore_index=True)

        # Bỏ trùng theo shop + product_url (tránh 1 sản phẩm xuất hiện nhiều lần)
        df_all.drop_duplicates(subset=["shop", "product_url"], inplace=True)

        df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"✅ Đã merge với data cũ. Tổng sản phẩm hiện có: {len(df_all)}")
    else:
        # Chưa có file → tạo mới
        df_new.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"✅ Tạo mới {csv_path} với {len(df_new)} sản phẩm.")

    print("\n👀 5 dòng đầu trong products.csv:")
    print(pd.read_csv(csv_path).head())
