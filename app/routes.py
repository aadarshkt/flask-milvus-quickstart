from flask import Blueprint, request, jsonify
from app.services.embedding_service import EmbeddingService
from app.services.llm.llm_service import LLMService
from app.utils.embedding_utils import get_embedding
from app.utils.file_utils import process_file

# Create blueprint
main = Blueprint("main", __name__)

# Initialize service
embedding_service = EmbeddingService()
llm_service = LLMService()


@main.route("/api/embeddings", methods=["GET"])
def list_embeddings():
    """
    List all stored texts with pagination
    """
    try:
        # Get limit from query parameters, default to 1000
        limit = request.args.get("limit", default=1000, type=int)
        texts = embedding_service.list_texts(limit=limit)
        return jsonify({"texts": texts}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main.route("/api/embeddings", methods=["POST"])
def add_embedding():
    """
    Add a new document and its chunks to the collection
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read file content
        file_content = file.read().decode("utf-8")

        # Process file and get chunks
        document_id, chunks = process_file(file_content)

        # Add each chunk to the collection and collect vector IDs
        vector_ids = []
        for chunk, chunk_id in chunks:
            embedding = get_embedding(chunk)
            vector_id = embedding_service.add_embedding(
                chunk, embedding, chunk_id, document_id
            )
            vector_ids.append(vector_id[0])  # Get the first (and only) ID from the list

        return (
            jsonify(
                {
                    "message": "Document processed successfully",
                    "document_id": document_id,
                    "chunk_count": len(chunks),
                    "vector_ids": vector_ids,
                }
            ),
            201,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main.route("/api/embeddings/<int:vector_id>", methods=["GET"])
def get_embedding(vector_id):
    """
    Get embedding and its metadata by vector ID
    """
    try:
        result = embedding_service.get_embedding_by_id(vector_id)
        if not result:
            return jsonify({"error": "Embedding not found"}), 404

        return jsonify(result), 200
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


@main.route("/api/embeddings/llm-search", methods=["POST"])
def llm_search():
    """
    Search for similar texts and get LLM response
    """
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Query text is required"}), 400

    try:
        # Get query embedding
        query_embedding = get_embedding(data["query"])

        # Search for similar texts
        relevant_texts = embedding_service.search_similar(
            query_embedding, limit=5  # Get top 5 most relevant texts
        )

        # Get LLM response
        llm_response = llm_service.get_response(data["query"], relevant_texts)

        return (
            jsonify({"relevant_texts": relevant_texts, "llm_response": llm_response}),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main.route("/api/embeddings/search-with-documents", methods=["POST"])
def search_with_documents():
    """
    Search for similar texts within specific documents
    """
    data = request.get_json()
    if not data or "query" not in data or "document_ids" not in data:
        return jsonify({"error": "Query text and document_ids are required"}), 400

    try:
        query_embedding = get_embedding(data["query"])
        results = embedding_service.search_similar_with_document_filter(
            query_embedding, data["document_ids"], limit=data.get("limit", 5)
        )
        return jsonify({"results": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
