# Milvus Flask Application

A scalable Flask application that demonstrates text embedding storage and retrieval using Milvus, enhanced with LLM-powered responses using Google's Gemini.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set up environment variables:

```bash
# Create .env file
touch .env

# Add required environment variables
GEMINI_API_KEY=your_gemini_api_key_here
```

3. Start Milvus server (using Docker):

```bash
docker run -d --name milvus_standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:v2.3.6
```

4. Run the application:

```bash
python app.py
```

## Project Structure

- `app/`: Main application package
  - `__init__.py`: Application factory
  - `routes.py`: API routes
  - `models/`: Database models
  - `services/`: Business logic
    - `embedding_service.py`: Milvus vector operations
    - `llm/`: LLM-related services
      - `llm_service.py`: Gemini integration
  - `utils/`: Utility functions
    - `embedding_utils.py`: Text embedding generation
- `config.py`: Configuration settings
- `app.py`: Application entry point

## API Endpoints

### Vector Search Endpoints

- POST `/api/embeddings`: Store text and its embedding
- GET `/api/embeddings`: List all stored texts
- POST `/api/embeddings/search`: Search for similar texts

### LLM-Enhanced Search Endpoint

- POST `/api/embeddings/llm-search`: Search for similar texts and get AI-generated response
  - Request body:
    ```json
    {
      "query": "Your question here"
    }
    ```
  - Response:
    ```json
    {
      "relevant_texts": [
        {
          "text": "Similar text 1",
          "distance": 0.1
        }
      ],
      "llm_response": "AI-generated response based on context"
    }
    ```

## Features

1. **Vector Search**

   - Text embedding generation and storage
   - Similarity search using cosine distance
   - Efficient indexing with IVF_FLAT

2. **LLM Integration**
   - Context-aware responses using Google's Gemini
   - Relevant text retrieval from vector database
   - Natural language understanding and generation

## Dependencies

- Flask 3.0.2
- Pymilvus 2.3.6
- Sentence Transformers 2.5.1
- Google Generative AI 0.3.2
- Python-dotenv 1.0.1
