from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

# Import service
from app.services.rag_service import rag_service
from app.services.vision_service import vision_service

router = APIRouter()

@router.post("/query")
async def chat_query(
    question: Optional[str] = Form(None),       # Tùy chọn
    session_id: str = Form(...),                # Bắt buộc
    file: Optional[UploadFile] = File(None)     # Tùy chọn
):
    vision_context = ""
    has_image = False 

    if file:
        try:
            content = await file.read()
            print(f"🔍 Đang tìm thông tin bệnh nhân trong lịch sử chat của session: {session_id}...")
            
            # Lấy thông tin user từ lịch sử chat
            real_user_data = rag_service.extract_user_info(session_id)
            
            # Gọi Vision 
            vision_result = vision_service.predict_image_bytes(content, real_user_data)
            
            if "error" not in vision_result:
                diagnosis = vision_result.get('diagnosis', 'Không rõ')
                confidence = vision_result.get('confidence', 0)
                
                vision_context = f"""
                [THÔNG TIN TỪ ẢNH USER GỬI]:
                - Vision AI chẩn đoán: {diagnosis}
                - Độ tin cậy: {confidence*100:.2f}%
                - Thông tin bệnh nhân: {real_user_data}
                (Hãy dùng thông tin này để tư vấn cho câu hỏi của user)
                """
                has_image = True
            else:
                vision_context = "[Lỗi Vision]: Không phân tích được ảnh."
        except Exception as e:
            print(f"Lỗi đọc ảnh chat: {e}")
            vision_context = "[Lỗi]: File ảnh bị hỏng."
    else:
        # Nếu không có ảnh
        vision_context = "[Lưu ý]: User KHÔNG gửi ảnh. Nếu họ hỏi bệnh, hãy nhắc họ bấm nút kẹp ghim (📎) để gửi ảnh."

    
    # Nếu question rỗng hoặc None
    if not question or not question.strip():
        if has_image:
            real_question = "Hãy chẩn đoán và tư vấn dựa trên bức ảnh này."
        else:
            return {"answer": "Bạn cần nhập câu hỏi hoặc gửi ảnh để tôi tư vấn nhé."}
    else:
        real_question = question

    # --- 3. TẠO PROMPT & GỌI RAG ---
    full_prompt = f"""
    {vision_context}
    
    [Câu hỏi người dùng]: {real_question}
    
    Hãy trả lời ngắn gọn, thân thiện như bác sĩ da liễu.
    """

    try:
        # Gọi RAG Service
        answer = rag_service.get_answer(query=full_prompt, session_id=session_id)
        return {"answer": answer}
        
    except Exception as e:
        print(f"Lỗi RAG Chat: {e}")
        return {"answer": "Xin lỗi, bác sĩ AI đang bận. Vui lòng thử lại sau."}