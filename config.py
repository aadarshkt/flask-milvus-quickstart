import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    # Milvus configuration
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

    # Collection configuration
    COLLECTION_NAME = "text_embeddings"
    DIMENSION = (
        384  # Default dimension for sentence-transformers/all-MiniLM-L6-v2 model
    )

    # Flask configuration
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
    DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"
