from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.rag_service import rag_service

router = APIRouter()

# Định nghĩa dữ liệu đầu vào 
# Web Chatbot sẽ gửi cục JSON này lên
class ConsultationRequest(BaseModel):
    disease_name: str       
    confidence: float       
    user_info: dict         

# Định nghĩa API Endpoint
@router.post("/consult")
async def get_medical_advice(request: ConsultationRequest):
    try:
        # Gọi RAG Service
        advice = rag_service.get_consultation(
            disease_name=request.disease_name,
            confidence=request.confidence,
            user_info=request.user_info
        )
        
        return {
            "status": "success",
            "disease": request.disease_name,
            "advice": advice
        }
        
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        raise HTTPException(status_code=500, detail="Lỗi hệ thống tư vấn AI")