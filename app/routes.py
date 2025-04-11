from flask import Blueprint, request, jsonify
from app.services.embedding_service import EmbeddingService
from app.utils.embedding_utils import get_embedding

# Create blueprint
main = Blueprint("main", __name__)

# Initialize service
embedding_service = EmbeddingService()


@main.route("/api/embeddings", methods=["POST"])
def add_embedding():
    """
    Add a new text and its embedding to the collection
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Text is required"}), 400

    text = data["text"]
    embedding = get_embedding(text)

    try:
        embedding_service.add_embedding(text, embedding)
        return jsonify({"message": "Embedding added successfully"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main.route("/api/embeddings/search", methods=["POST"])
def search_embeddings():
    """
    Search for similar texts based on query text
    """
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Query text is required"}), 400

    try:
        query_embedding = get_embedding(data["query"])
        results = embedding_service.search_similar(query_embedding)
        return jsonify({"results": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main.route("/api/embeddings", methods=["GET"])
def list_embeddings():
    """
    List all stored texts
    """
    try:
        texts = embedding_service.list_texts()
        return jsonify({"texts": texts}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
