from fastapi import APIRouter
from app.api.v1.endpoints import chat, diagnosis  

api_router = APIRouter()

# Router cho Chatbot
api_router.include_router(chat.router, prefix="/chat", tags=["Chatbot RAG"])

# Router cho Vision 
# URL sẽ là: /api/v1/diagnosis/analyze-full
api_router.include_router(diagnosis.router, prefix="/diagnosis", tags=["AI Vision"])