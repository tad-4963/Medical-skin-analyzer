import torch
import torch.nn.functional as F
from PIL import Image
import io
import os

from app.services.model import SkinNet
from app.services.transforms import get_transforms

MODEL_PATH = "data/models/efficientnet_b0.pth"
DEVICE = "cpu" 
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Danh sách nhãn 
CLASS_NAMES = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']

DISEASE_MAP = {
    'ACK': 'Dày sừng quang hóa (Actinic Keratosis)',
    'BCC': 'Ung thư biểu mô tế bào đáy (Basal Cell Carcinoma)',
    'MEL': 'U hắc tố (Melanoma) - NGUY HIỂM',
    'NEV': 'Nốt ruồi lành tính (Nevus)',
    'SCC': 'Ung thư biểu mô tế bào vảy (Squamous Cell Carcinoma)',
    'SEK': 'Dày sừng tiết bã (Seborrheic Keratosis)'
}

class VisionService:
    def __init__(self):
        self.device = DEVICE
        print(f"--> Đang load Vision Model từ: {MODEL_PATH}")
        
        self.model = SkinNet()
        
        # Load weights
        if os.path.exists(MODEL_PATH):
            try:
                state_dict = torch.load(MODEL_PATH, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                print("Vision Model đã sẵn sàng!")
            except Exception as e:
                print(f"Lỗi khi load weights: {e}")
                self.model = None
        else:
            print(f"Lỗi: Không tìm thấy file {MODEL_PATH}")
            self.model = None
            
        self.transform = get_transforms()

    def predict_image_bytes(self, image_bytes, patient_data):
        if self.model is None:
            return {"error": "Model chưa được load thành công"}

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)

            try:
                age = float(patient_data.get('age', 30))
            except:
                age = 30.0
                
            gender = 1.0 if str(patient_data.get('gender')).upper() in ['MALE', 'NAM', '1'] else 0.0
            itch = 1.0 if str(patient_data.get('itch')).upper() in ['YES', 'CÓ', 'TRUE'] else 0.0
            grew = 1.0 if str(patient_data.get('grew')).upper() in ['YES', 'CÓ', 'TRUE'] else 0.0
            bleed = 1.0 if str(patient_data.get('bleed')).upper() in ['YES', 'CÓ', 'TRUE'] else 0.0

            input_data = [age, gender, itch, grew, bleed]
            meta_tensor = torch.tensor([input_data], dtype=torch.float32).to(self.device)

            with torch.no_grad():
                outputs = self.model(img_tensor, meta_tensor)
                probs = F.softmax(outputs, dim=1)[0]

            top_idx = probs.argmax().item()
            top_label = CLASS_NAMES[top_idx]
            confidence = probs[top_idx].item()

            return {
                "diagnosis": DISEASE_MAP.get(top_label, top_label),
                "diagnosis_code": top_label,
                "confidence": round(confidence, 4)
            }
            
        except Exception as e:
            return {"error": str(e)}

vision_service = VisionService()