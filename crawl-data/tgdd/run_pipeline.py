# run_pipeline.py
import subprocess
import sys

def run_script(script_name: str):
    """Chạy 1 file .py bằng đúng bản Python hiện tại."""
    try:
        subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Chạy {script_name} bị lỗi, code: {e.returncode}")

def main():
    while True:
        print("\n=== CHỌN PIPELINE ===")
        print("1. Chạy get_product_name.py")
        print("2. Chạy get_reviews.py")
        print("0. Thoát")
        choice = input("Nhập lựa chọn (0/1/2): ").strip()

        if choice == "1":
            run_script("get_product_name.py")
        elif choice == "2":
            run_script("get_reviews.py")
        elif choice == "0":
            print("Thoát.")
            break
        else:
            print("Lựa chọn không hợp lệ, vui lòng nhập 0, 1 hoặc 2.")

if __name__ == "__main__":
    main()
