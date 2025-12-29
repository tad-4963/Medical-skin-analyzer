import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


DATA_PATH = "../data/documents"      #file PDF/TXT
DB_PATH = "../data/vector_db"        
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 

def create_vector_db():
    if not os.path.exists(DATA_PATH):
        print(f"Lỗi: Không tìm thấy thư mục {DATA_PATH}")
        return

    print("Đang đọc tài liệu từ folder data/documents/...")
    
    pdf_loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    pdf_docs = pdf_loader.load()
    
    txt_loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    txt_docs = txt_loader.load()
    
    documents = pdf_docs + txt_docs
    
    if len(documents) == 0:
        print("Không tìm thấy tài liệu nào! Hãy bỏ file PDF/TXT vào data/documents/ trước.")
        return

    print(f"Đã load {len(documents)} văn bản.")

    # Chia nhỏ văn bản 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    print(f"Đã chia thành {len(chunks)} đoạn nhỏ.")

    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    print("Đang tạo Vector Database (ChromaDB)")
    
    # Tạo embedding function
    embedding_func = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Lưu vào ChromaDB
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_func, 
        persist_directory=DB_PATH
    )
    
    
    print(f"Hoàn tất! Dữ liệu đã được lưu tại {DB_PATH}")

if __name__ == "__main__":
    create_vector_db()