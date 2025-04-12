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

        # Create schema
        schema = CollectionSchema(
            fields=[id_field, text_field, embedding_field],
            description="Collection for storing text embeddings",
        )

        # Create collection
        collection = Collection(name=self.collection_name, schema=schema)

        # Create index
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }
        collection.create_index(field_name="embedding", index_params=index_params)

    def add_embedding(self, text, embedding):
        """
        Add a new text and its embedding to the collection
        """
        entities = [[text], [embedding]]
        self.collection.insert(entities)

    def search_similar(self, query_embedding, limit=5):
        """
        Search for similar texts based on query embedding
        """
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=["text"],
        )

        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append(
                    {"text": hit.entity.get("text"), "distance": hit.distance}
                )

        return formatted_results

    def list_texts(self, limit=1000):
        """
        List all stored texts with pagination
        """
        results = self.collection.query(expr="", output_fields=["text"], limit=limit)
        return [result.get("text") for result in results]
