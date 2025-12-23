from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.api import api_router
from app.core.config import settings  # Nếu chưa có file config thì có thể bỏ qua dòng này

app = FastAPI(title="Medical Chatbot API")

# 1. Cấu hình CORS (Để Web Chatbot gọi được API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép mọi nguồn (Web, Mobile...)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Kết nối Router
# Bao gồm cả API chẩn đoán ảnh và API Chatbot RAG
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "Medical Chatbot API is running!"}

if __name__ == "__main__":
    import uvicorn
    # Chạy server tại port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)