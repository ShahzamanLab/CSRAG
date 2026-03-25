from langchain_huggingface import HuggingFaceEmbeddings
class EmbeddingGenerator:
    """Class to create embeddings using HuggingFace."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
    
    def create_embeddings(self):
        embedding_model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device}
        )
        return embedding_model