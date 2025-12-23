import os
from langchain_chroma import Chroma                          
from langchain_huggingface import HuggingFaceEmbeddings

DB_PATH = "data/vector_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class VectorDBService:
    def __init__(self):
        print("Đang load Vector Database...")
        
        self.embedding_func = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"Không tìm thấy DB tại {DB_PATH}. Hãy chạy script ingest_data.py trước!")
            
        self.db = Chroma(
            persist_directory=DB_PATH, 
            embedding_function=self.embedding_func
        )
        print("Đã kết nối thành công với Vector DB!")

    def search_similar(self, query: str, k=3):
        """
        Tìm kiếm k đoạn văn bản liên quan nhất với query.
        """
        docs = self.db.similarity_search(query, k=k)
        return docs

vector_db_service = VectorDBService()