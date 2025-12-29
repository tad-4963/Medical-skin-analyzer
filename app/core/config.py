import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Tên dự án
    PROJECT_NAME: str = "Medical Skin Analyzer"
    
    # Load API Key 
    OPENAI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    # Cấu hình nạp file .env
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()