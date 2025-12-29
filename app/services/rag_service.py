import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from app.services.vector_store import vector_db_service

from dotenv import load_dotenv
load_dotenv()

class RAGService:
    def __init__(self):
        # Kiểm tra Key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Cảnh báo: Chưa tìm thấy OPENAI_API_KEY. Hãy kiểm tra file .env!")
        
        # Khởi tạo LLM (gpt-4o-mini hoặc gemini-2.0-flash)
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=api_key,
            temperature=0.5,
        )
        self.chat_sessions = {}

    def get_consultation(self, disease_name: str, confidence: float, user_info: dict) -> str:
        """
        Hàm tư vấn CHUYÊN SÂU (Dùng cho luồng /analyze-full)
        """
        print("\n" + "="*60)
        print(f"[DIAGNOSIS-LOG] Bắt đầu quy trình tư vấn cho bệnh: {disease_name}")
        search_query = f"Bệnh {disease_name}: Nguyên nhân, triệu chứng, phác đồ điều trị thuốc và lời khuyên chăm sóc tại nhà"
        # 1. Truy vấn Vector DB 
        print(f"[RAG-LOG] Đang truy vấn Vector DB với từ khóa: '{search_query}'...")
        docs = vector_db_service.search_similar(search_query, k=3) # Lấy 3 tài liệu sát nhất
        
        # Chi tiết phần RAG
        print(f"[RAG-LOG] Đã tìm thấy {len(docs)} tài liệu tham khảo:")
        
        for i, doc in enumerate(docs):
            # Lấy nguồn file 
            source = doc.metadata.get('source', 'Không rõ nguồn') 
            page = doc.metadata.get('page', 'N/A')
            
            content_preview = doc.page_content[:200].replace('\n', ' ')
            
            print(f"  [Tài liệu {i+1}]:")
            print(f"       - Nguồn gốc: {source} (Trang {page})")
            print(f"       - Nội dung trích xuất: \"{content_preview}...\"")
            print("   " + "-"*30)
        
        context_text = "\n\n".join([d.page_content for d in docs])
        
        if not context_text:
            print("[CẢNH BÁO] Không tìm thấy tài liệu nào khớp!")
            context_text = "Không tìm thấy tài liệu y khoa cụ thể nào trong cơ sở dữ liệu."

        # 2. Xây dựng Prompt 
        template = """
        Bạn là Dr. AI, một trợ lý y tế ảo chuyên về da liễu.
        
        DỮ LIỆU ĐẦU VÀO:
        - Bệnh chẩn đoán: {disease} (Độ tin cậy: {confidence})
        - Thông tin bệnh nhân: {user_info}
        
        KIẾN THỨC THAM KHẢO:
        {context}
        
        NHIỆM VỤ:
        Tư vấn cho bệnh nhân theo cấu trúc:
        1. Chào hỏi & Thông báo kết quả.
        2. Giải thích bệnh {disease} là gì.
        3. Lời khuyên điều trị/chăm sóc.
        4. Cảnh báo (nếu độ tin cậy thấp).
        LƯU Ý: Hạn chế sử dụng ký tự đặc biệt như '-', '*'.
        BẮT BUỘC: Kết thúc bằng câu: "Lưu ý: Đây chỉ là chẩn đoán sơ bộ. Vui lòng đến gặp bác sĩ chuyên khoa để thăm khám."
        """
        
        prompt_template = PromptTemplate(
            template=template,
            input_variables=["disease", "confidence", "user_info", "context"]
        )
        
        final_prompt = prompt_template.format(
            disease=disease_name,
            confidence=confidence,
            user_info=str(user_info), 
            context=context_text
        )
        
        # 3. Gửi cho LLM
        print("[LLM-LOG] Đang gửi Prompt tổng hợp cho ChatGPT...")
        response = self.llm.invoke(final_prompt)
        print("[DONE] Đã nhận câu trả lời tư vấn.")
        print("="*60 + "\n")
        
        return response.content

    def get_answer(self, query: str, session_id: str = "guest") -> str:
        """
        Hàm trả lời HỘI THOẠI (Chatbot) 
        """
        print(f"\n [CHAT-LOG] Session: {session_id} | User hỏi: {query}")

        if session_id not in self.chat_sessions:
            self.chat_sessions[session_id] = []
        
        current_history = self.chat_sessions[session_id]
        
        history_text = ""
        for msg in current_history[-10:]: 
            role = "Bệnh nhân" if msg['role'] == 'user' else "Bác sĩ AI"
            history_text += f"- {role}: {msg['content']}\n"
        
        if not history_text: history_text = "Chưa có hội thoại trước đó."

        social_keywords = ["xin chào", "chào", "hi", "hello", "alo", "bác sĩ ơi", "có ai không"]
        is_pure_greeting = len(query.split()) <= 3 and any(k in query.lower() for k in social_keywords)

        context_text = ""
        if not is_pure_greeting:
            refine_prompt = f"""
            Bạn là trợ lý tìm kiếm y tế. Dựa vào lịch sử chat và câu hỏi mới nhất, hãy tạo ra 1 câu truy vấn tìm kiếm (Search Query) để tra cứu trong tài liệu y khoa.
            
            [LỊCH SỬ CHAT]:
            {history_text}
            
            [CÂU HỎI MỚI]: {query}
            
            [YÊU CẦU]:
            1. Xác định tên bệnh (nếu đã được nhắc đến trong lịch sử hoặc câu hỏi).
            2. Tập trung vào các từ khóa: "Triệu chứng", "Điều trị", "Thuốc", "Nguyên nhân".
            3. Loại bỏ các từ cảm thán (ơi, à, nhé, cứu tôi...).
            4. Chỉ trả về câu truy vấn ngắn gọn.
            
            Ví dụ:
            - Input: "Cái nốt ruồi này có sao ko" -> Output: "Dấu hiệu ung thư hắc tố melanoma nốt ruồi ác tính"
            - Input: "Bôi thuốc gì cho hết" (đang nói về mụn) -> Output: "Thuốc bôi điều trị mụn trứng cá phác đồ"
            """
            try:
                # Gọi GPT để tạo từ khóa tìm kiếm
                refined_query_response = self.llm.invoke(refine_prompt)
                search_query = refined_query_response.content.strip()
                print(f"🔍[SEARCH-QUERY]: {search_query}") 
            except Exception:
                search_query = query
            # Chỉ search DB nếu không phải là câu chào xã giao ngắn gọn
            try:
                context_docs = vector_db_service.search_similar(search_query, k=3)
                if context_docs:
                    context_text = "\n\n".join([doc.page_content for doc in context_docs])
                else:
                    context_text = "Không tìm thấy tài liệu y khoa phù hợp."
            except Exception as e:
                print(f"Lỗi Vector DB: {e}")
                context_text = "Lỗi truy xuất dữ liệu."
        else:
            print("--> Phát hiện chào hỏi xã giao -> Bỏ qua RAG search.")
            context_text = "Đây là câu chào hỏi xã giao. Hãy chào lại thân thiện và hỏi người dùng cần giúp gì về da liễu."

        # 2. Prompt thông minh (Xử lý 2 tình huống)
        final_chat_prompt = f"""
        Bạn là Dr. AI - Trợ lý da liễu tại Phòng khám Thái Hà.
        
        [LỊCH SỬ HỘI THOẠI]:
        {history_text}

        [THÔNG TIN TRA CỨU TỪ DATABASE]:
        {context_text}
        ------------------------------------------
        
        [CÂU HỎI NGƯỜI DÙNG]: 
        {query}
        
        [HƯỚNG DẪN TRẢ LỜI]:
        Hãy phân tích ý định của người dùng:

        -->QUAN TRỌNG: HÃY XEM LỊCH SỬ HỘI THOẠI ĐỂ TRẢ LỜI BỆNH NHÂN: NẾU BỆNH NHÂN MUỐN TƯ VẤN THÌ HÃY YÊU CẦU HỌ GỬI THÔNG TIN CẦN THIẾT

        TRƯỜNG HỢP 1: XÃ GIAO / CHÀO HỎI
        - Nếu người dùng chỉ chào (vd: "Xin chào", "Hi", "Bác sĩ ơi")... hoặc hỏi vu vơ không liên quan đến bệnh.
        - HÀNH ĐỘNG: Bỏ qua hoàn toàn [THÔNG TIN TRA CỨU]. Hãy chào lại thân thiện, ngắn gọn và hỏi họ cần giúp gì về da liễu.
        
        TRƯỜNG HỢP 2: HỎI BỆNH / KIẾN THỨC
        - Nếu người dùng mô tả triệu chứng hoặc hỏi về bệnh.
        - HÀNH ĐỘNG: Sử dụng [THÔNG TIN TRA CỨU] để trả lời chính xác. Nếu thông tin tra cứu không đủ, hãy trả lời dựa trên kiến thức y khoa chung của bạn nhưng nhớ nhắc họ đi khám.
        
        TRƯỜNG HỢP 3: CẦN ĐIỀU TRỊ / TÌM NƠI KHÁM (QUAN TRỌNG)
           - Nếu người dùng hỏi: "Chữa ở đâu?", "Có thuốc gì không?", "Bệnh này nặng không?", hoặc hỏi về liệu trình điều trị cụ thể.
           - HÀNH ĐỘNG: 
             + Giải thích sơ lược về phương pháp điều trị (Laser, tiểu phẫu, thuốc bôi...).
             + CHỐT HẠ: Khuyên người dùng nên đến trực tiếp Phòng khám Thái Hà để bác sĩ soi da và lên phác đồ điều trị chuẩn xác nhất.
             + Câu mẫu: "Để điều trị dứt điểm, mời bạn ghé Phòng khám Thái Hà. Các bác sĩ chuyên khoa của chúng tôi sẽ thăm khám trực tiếp cho bạn nhé."
             + Nếu người dùng hỏi cách đặt lịch thì hướng dẫn họ liên hệ qua số hotline: 0365 116 117 hoặc ấn nút đặt lịch ngay phía trên góc phải trang web
        Lưu ý: Luôn trả lời bằng tiếng Việt, giọng văn bác sĩ ân cần.
                 Hạn chế sử dụng ký tự đặc biệt như '-', '*'.
        """

        # 3. Gọi LLM
        response = self.llm.invoke(final_chat_prompt)
        bot_reply = response.content
        
        # Lưu vào bộ nhớ
        self.chat_sessions[session_id].append({"role": "user", "content": query})
        self.chat_sessions[session_id].append({"role": "bot", "content": bot_reply})
        
        return bot_reply
    
    def extract_user_info(self, session_id: str) -> dict:
        """
        Đọc lịch sử chat để trích xuất thông tin: Tuổi, Giới tính, Triệu chứng.
        """
        # 1. Lấy lịch sử
        history = self.chat_sessions.get(session_id, [])
        if not history:
            # Nếu chưa chat gì thì trả về mặc định
            return {"age": 0, "gender": "Unknown", "itch": "Unknown", "bleed": "Unknown", "grew": "Unknown"}

        # 2. Gom lịch sử thành văn bản
        history_text = ""
        for msg in history:
            role = "Bệnh nhân" if msg['role'] == 'user' else "Bác sĩ"
            history_text += f"- {role}: {msg['content']}\n"

        # 3. Prompt yêu cầu AI trích xuất thông tin 
        extraction_prompt = f"""
        Dựa vào đoạn hội thoại dưới đây, hãy trích xuất thông tin y tế của bệnh nhân thành JSON.
        
        [HỘI THOẠI]:
        {history_text}
        
        [YÊU CẦU]:
        Hãy tìm các thông tin sau (nếu không có thì điền "Unknown"):
        - age: Tuổi (số nguyên, nếu không rõ điền 0).
        - gender: Giới tính (Male/Female/Unknown).
        - itch: Có ngứa không? (True/False/Unknown).
        - bleed: Có chảy máu không? (True/False/Unknown).
        - grew: Nốt có to lên không? (True/False/Unknown).
        
        Chỉ trả về chuỗi JSON duy nhất, không giải thích gì thêm.
        Ví dụ: {{"age": 25, "gender": "Male", "itch": "True", "bleed": "False", "grew": "Unknown"}}
        """

        try:
            # Gọi LLM để phân tích
            response = self.llm.invoke(extraction_prompt)
            content = response.content.strip()
            
            # Làm sạch chuỗi JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[0].strip()
                
            import json
            user_info = json.loads(content)
            print(f"[INFO-EXTRACT] Đã trích xuất thông tin từ lịch sử: {user_info}")
            return user_info
            
        except Exception as e:
            print(f" Lỗi trích xuất thông tin: {e}")
            return {"age": 0, "gender": "Unknown", "itch": "Unknown", "bleed": "Unknown", "grew": "Unknown"}

# Khởi tạo instance
rag_service = RAGService()