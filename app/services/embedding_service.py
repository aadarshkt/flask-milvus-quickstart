from pymilvus import Collection, utility
from config import Config


class EmbeddingService:
    def __init__(self):
        """
        Initialize the embedding service and create collection if it doesn't exist
        """
        self.collection_name = Config.COLLECTION_NAME
        self.dimension = Config.DIMENSION

        # Create collection if it doesn't exist
        if not utility.has_collection(self.collection_name):
            self._create_collection()

        # Load collection
        self.collection = Collection(self.collection_name)
        self.collection.load()

    def _create_collection(self):
        """
        Create a new collection with the required schema
        """
        from pymilvus import CollectionSchema, FieldSchema, DataType

        # Define fields
        id_field = FieldSchema(
            name="id", dtype=DataType.INT64, is_primary=True, auto_id=True
        )
        text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
        embedding_field = FieldSchema(
            name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension
        )
        chunk_id_field = FieldSchema(name="chunk_id", dtype=DataType.INT64)
        document_id_field = FieldSchema(
            name="document_id", dtype=DataType.VARCHAR, max_length=255
        )

        # Create schema
        schema = CollectionSchema(
            fields=[
                id_field,
                text_field,
                embedding_field,
                chunk_id_field,
                document_id_field,
            ],
            description="Collection for storing text embeddings with document and chunk information",
        )

        # Create collection
        collection = Collection(name=self.collection_name, schema=schema)

        # Create index
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }
        collection.create_index(field_name="embedding", index_params=index_params)

    def add_embedding(self, text, embedding, chunk_id, document_id):
        """
        Add a new text and its embedding to the collection with chunk and document information
        Returns the vector IDs of the inserted embeddings
        """
        entities = [[text], [embedding], [chunk_id], [document_id]]
        # Insert and get the primary keys (vector IDs)
        result = self.collection.insert(entities)
        return result.primary_keys  # Returns list of vector IDs

    def get_embedding_by_id(self, vector_id):
        """
        Retrieve embedding and its metadata by vector ID
        """
        result = self.collection.query(
            expr=f"id == {vector_id}",
            output_fields=["text", "embedding", "chunk_id", "document_id"],
        )
        return result[0] if result else None

    def search_similar(self, query_embedding, limit=5):
        """
        Search for similar texts based on query embedding
        """
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=["text", "chunk_id", "document_id"],
        )

        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append(
                    {
                        "text": hit.entity.get("text"),
                        "distance": hit.distance,
                        "chunk_id": hit.entity.get("chunk_id"),
                        "document_id": hit.entity.get("document_id"),
                    }
                )

        return formatted_results

    def search_similar_with_document_filter(
        self, query_embedding, document_ids, limit=5
    ):
        """
        Search for similar texts based on query embedding, filtered by document IDs
        """
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        # Create expression to filter by document IDs
        document_ids_str = ", ".join([f"'{doc_id}'" for doc_id in document_ids])
        expr = f"document_id in [{document_ids_str}]"

        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            expr=expr,  # Add the filter expression
            output_fields=["text", "chunk_id", "document_id"],
        )

        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append(
                    {
                        "text": hit.entity.get("text"),
                        "distance": hit.distance,
                        "chunk_id": hit.entity.get("chunk_id"),
                        "document_id": hit.entity.get("document_id"),
                    }
                )

        return formatted_results

    def list_texts(self, limit=1000):
        """
        List all stored texts with pagination
        """
        results = self.collection.query(expr="", output_fields=["text"], limit=limit)
        return [result.get("text") for result in results]
