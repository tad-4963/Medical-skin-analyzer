# File: check_models.py
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Lỗi: Không tìm thấy GOOGLE_API_KEY trong file .env")
else:
    try:
        genai.configure(api_key=api_key)
        print(f"🔑 Đang kiểm tra với Key: {api_key[:10]}...")
        
        print("\n📋 DANH SÁCH MODEL BẠN ĐƯỢC DÙNG:")
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"   ✅ {m.name}")
                available_models.append(m.name)
        
        if not available_models:
            print("\n⚠️ Không tìm thấy model nào! Có thể Key bị lỗi hoặc sai vùng (Region).")
    except Exception as e:
        print(f"\n❌ Lỗi kết nối: {e}")