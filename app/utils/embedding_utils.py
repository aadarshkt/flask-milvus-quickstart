from sentence_transformers import SentenceTransformer

# Load the model (this will download it on first run)
model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text):
    """
    Generate embedding for the given text using sentence-transformers
    """
    # Convert text to embedding
    embedding = model.encode(text)
    # Convert to list for Milvus compatibility
    return embedding.tolist()
