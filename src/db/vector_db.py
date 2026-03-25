from langchain_community.vectorstores import FAISS
from langchain_classic.schema import Document  # Use langchain.schema.Document

class VectorStore:
    """Class to create a FAISS vectorstore from document chunks and embeddings."""
    def __init__(self, docs, embedding_model):
        self.docs = docs
        self.embedding_model = embedding_model
    
    def build(self):
        # Convert plain strings to Document objects if needed
        doc_objects = [
            d if isinstance(d, Document) else Document(page_content=d)
            for d in self.docs
        ]
        vectorstore = FAISS.from_documents(doc_objects, self.embedding_model)
        return vectorstore