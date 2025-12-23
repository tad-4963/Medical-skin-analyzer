import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from app.services.vector_store import vector_db_service

# Load biến môi trường nếu cần (thường FastAPI sẽ tự load khi chạy main, nhưng thêm cho chắc)
from dotenv import load_dotenv
load_dotenv()

class RAGService:
    def __init__(self):
        # Kiểm tra Key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Cảnh báo: Chưa tìm thấy OPENAI_API_KEY. Hãy kiểm tra file .env!")
        
        # Khởi tạo LLM (gpt-4o-mini)
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini", 
            temperature=0.4,  
            api_key=api_key
        )

    def get_consultation(self, disease_name: str, confidence: float, user_info: dict) -> str:
        """
        Hàm tư vấn chính:
        1. Tìm kiếm tài liệu y khoa liên quan đến 'disease_name'.
        2. Tổng hợp thông tin + metadata người dùng.
        3. Yêu cầu ChatGPT viết lời khuyên.
        """
        
        # Truy vấn Vector DB 
        print(f"Đang tìm tài liệu về: {disease_name}...")
        docs = vector_db_service.search_similar(disease_name, k=2)
        
        context_text = "\n\n".join([d.page_content for d in docs])
        
        if not context_text:
            context_text = "Không tìm thấy tài liệu y khoa cụ thể nào trong cơ sở dữ liệu."

        # Xây dựng Prompt 
        template = """
        Bạn là Dr. AI, một trợ lý y tế ảo chuyên về da liễu, thân thiện và chuyên nghiệp.
        
        DỮ LIỆU ĐẦU VÀO:
        - Bệnh được chẩn đoán qua ảnh: {disease}
        - Độ tin cậy của AI Vision: {confidence}
        - Thông tin bệnh nhân: {user_info}
        
        KIẾN THỨC Y KHOA THAM KHẢO (Đã được xác thực):
        {context}
        
        NHIỆM VỤ CỦA BẠN:
        Hãy viết câu trả lời tư vấn cho bệnh nhân. Cấu trúc câu trả lời:
        1. **Chào hỏi & Xác nhận**: Chào bệnh nhân (dựa trên thông tin họ tên/tuổi nếu có) và thông báo kết quả phân tích ảnh.
        2. **Giải thích**: Giải thích ngắn gọn bệnh {disease} là gì dựa trên kiến thức tham khảo.
        3. **Lời khuyên**: Đưa ra hướng xử lý tại nhà hoặc cảnh báo nếu nguy hiểm.
        4. **Cảnh báo quan trọng**: Nếu độ tin cậy < 0.65, hãy khuyên họ chụp lại ảnh rõ nét hơn.
        
        BẮT BUỘC:
        Cuối câu trả lời phải có câu: "Lưu ý: Đây chỉ là chẩn đoán sơ bộ. Vui lòng đến gặp bác sĩ chuyên khoa da liễu để được thăm khám chính xác nhất."
        """
        
        prompt_template = PromptTemplate(
            template=template,
            input_variables=["disease", "confidence", "user_info", "context"]
        )
        
        # Tạo prompt hoàn chỉnh
        final_prompt = prompt_template.format(
            disease=disease_name,
            confidence=confidence,
            user_info=str(user_info), 
            context=context_text
        )
        
        # Gửi cho ChatGPT
        print("Đang hỏi ý kiến chuyên gia AI...")
        response = self.llm.invoke(final_prompt)
        
        return response.content

rag_service = RAGService()