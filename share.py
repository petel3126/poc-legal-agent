import subprocess
import re
import sys
import os
import time

def start_tunnel():
    print("=" * 65)
    print(" 🚀 ĐANG KHỞI TẠO ĐƯỜNG LINK CHIA SẺ RA INTERNET...")
    print("=" * 65)
    
    cloudflared_path = os.path.join(os.path.dirname(__file__), "cloudflared.exe")
    if not os.path.exists(cloudflared_path):
        print("[!] Không tìm thấy cloudflared.exe trong thư mục!")
        return

    # Start cloudflared process
    proc = subprocess.Popen(
        [cloudflared_path, "tunnel", "--url", "http://127.0.0.1:8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    found_url = False
    print("Đang kết nối đến Cloudflare Server (khoảng 3-5 giây)...")
    
    # Read stderr to find the trycloudflare.com link
    while True:
        line = proc.stderr.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            # Check for trycloudflare.com URL
            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
            if match and not found_url:
                found_url = True
                url = match.group(0)
                print("\n" + "=" * 65)
                print(" 🎉 ĐƯỜNG LINK PUBLIC CỦA BẠN ĐÃ SẴN SÀNG:")
                print(f" 👉 {url}")
                print("=" * 65)
                print("\n💡 Hãy copy link trên (bắt đầu bằng https://...) gửi cho người khác.")
                print("⚠️  LƯU Ý: KHÔNG TẮT cửa sổ này khi đang chia sẻ!\n")
    
    proc.wait()

if __name__ == "__main__":
    start_tunnel()
