# Milvus Flask Application

A scalable Flask application that demonstrates text embedding storage and retrieval using Milvus.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start Milvus server (using Docker):

```bash
docker run -d --name milvus_standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:v2.3.6
```

3. Run the application:

```bash
python app.py
```

## Project Structure

- `app/`: Main application package
  - `__init__.py`: Application factory
  - `routes.py`: API routes
  - `models/`: Database models
  - `services/`: Business logic
  - `utils/`: Utility functions
- `config.py`: Configuration settings
- `app.py`: Application entry point

## API Endpoints

- POST `/api/embeddings`: Store text and its embedding
- GET `/api/embeddings/search`: Search for similar texts
- GET `/api/embeddings`: List all stored texts
