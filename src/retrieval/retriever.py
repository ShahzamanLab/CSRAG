from langchain_community.vectorstores import FAISS

class Retriever:
    """Class to create a retriever from a FAISS vectorstore."""

    def __init__(self, vectorstore: FAISS, search_kwargs=None):
        """
        vectorstore: FAISS vectorstore object
        search_kwargs: dict, optional parameters for retrieval like k
        """
        self.vectorstore = vectorstore
        self.search_kwargs = search_kwargs or {"k": 5}  # default top 5 results

    def get_retriever(self):
        """Return a LangChain retriever from the FAISS vectorstore."""
        retriever = self.vectorstore.as_retriever(search_kwargs=self.search_kwargs)
        return retriever

# -------------------------
# Example usage
# -------------------------
# Suppose you already have:
# vs = VectorStore(docs, embedding_model).build()

# retriever_obj = Retriever(vs)
# retriever = retriever_obj.get_retriever()

# Now you can use retriever.get_relevant_documents(query)