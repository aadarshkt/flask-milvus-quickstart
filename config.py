import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    # Milvus Configuration
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "text_embeddings")
    DIMENSION = int(os.getenv("VECTOR_DIMENSION", "768"))

    # Flask Configuration
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")

    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # Model Configuration
    MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
