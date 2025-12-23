from fastapi import APIRouter
from app.api.v1.endpoints import chat, diagnosis  # <--- Thêm diagnosis vào đây

api_router = APIRouter()

# Router cho Chatbot (Đã có)
api_router.include_router(chat.router, prefix="/chat", tags=["Chatbot RAG"])

# Router cho Vision (Thêm mới đoạn này)
# URL sẽ là: /api/v1/diagnosis/analyze-full
api_router.include_router(diagnosis.router, prefix="/diagnosis", tags=["AI Vision"])