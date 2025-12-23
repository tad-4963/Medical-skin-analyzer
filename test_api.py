import requests

# URL của API mới (Vừa có Vision, vừa có Chatbot)
API_URL = "http://127.0.0.1:8000/api/v1/diagnosis/analyze-full"

# Đường dẫn đến 1 file ảnh bất kỳ trong máy bạn để test
# Bạn nhớ sửa lại đường dẫn này cho đúng ảnh thật nhé!
# Thêm chữ r ở đầu
IMAGE_PATH = r"F:\CODE\DL\data\imgs_part_1\PAT_29_40_561.png"

def test_full_flow():
    print(f"🚀 Đang gửi ảnh {IMAGE_PATH} lên Server...")

    try:
        # 1. Chuẩn bị file ảnh
        files = {
            'file': ('image.jpg', open(IMAGE_PATH, 'rb'), 'image/jpeg')
        }

        # 2. Chuẩn bị thông tin bệnh nhân (Metadata)
        data = {
            'age': '25',
            'gender': 'Nam',
            'itch': 'Yes',    # Có ngứa
            'grew': 'No',     # Không lớn nhanh
            'bleed': 'No'     # Không chảy máu
        }

        # 3. Gửi Request
        response = requests.post(API_URL, files=files, data=data)

        # 4. Xem kết quả
        if response.status_code == 200:
            result = response.json()
            print("\n" + "="*50)
            print("✅ KẾT QUẢ PHÂN TÍCH:")
            print("="*50)
            
            # Phần Vision
            vision = result.get('vision_result', {})
            print(f"👁️  Bác sĩ AI (Vision) chẩn đoán: {vision.get('diagnosis')}")
            print(f"📊  Độ tin cậy: {vision.get('confidence')}")
            
            print("-" * 50)
            
            # Phần Chatbot
            print("💬 Bác sĩ AI (Chatbot) tư vấn:")
            print(result.get('advice')) # Hoặc 'consultation' tùy code bạn đặt key là gì
            print("="*50)
        else:
            print("❌ Lỗi Server:")
            print(response.text)

    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file ảnh! Hãy sửa lại biến IMAGE_PATH.")
    except Exception as e:
        print(f"❌ Lỗi kết nối: {e}")

if __name__ == "__main__":
    test_full_flow()