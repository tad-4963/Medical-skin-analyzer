from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.vision_service import vision_service
from app.services.rag_service import rag_service

router = APIRouter()

@router.post("/analyze-full")
async def analyze_full(
    file: UploadFile = File(...),
    age: str = Form(...),    
    gender: str = Form(...),
    itch: str = Form("No"),
    grew: str = Form("No"),
    bleed: str = Form("No")
):
    # Đọc ảnh
    content = await file.read()
    
    # Chạy Vision (Nhìn ảnh đoán bệnh)
    print("Đang chạy Vision Model...")
    patient_data = {"age": age, "gender": gender, "itch": itch, "grew": grew, "bleed": bleed}
    vision_result = vision_service.predict_image_bytes(content, patient_data)
    
    if "error" in vision_result:
        raise HTTPException(status_code=500, detail=vision_result["error"])
        
    # Chạy RAG (Hỏi bác sĩ AI)
    print(f"Đang tư vấn cho bệnh: {vision_result['diagnosis']}")
    user_info = f"Tuổi: {age}, Giới tính: {gender}, Ngứa: {itch}, Lớn nhanh: {grew}, Chảy máu: {bleed}"
    
    advice = rag_service.get_consultation(
        disease_name=vision_result['diagnosis'],
        confidence=vision_result['confidence'],
        user_info=user_info
    )
    
    # Trả về Full kết quả
    return {
        "status": "success",
        "vision_result": vision_result,
        "advice": advice
    }