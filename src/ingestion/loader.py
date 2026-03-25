from langchain_community.document_loaders import TextLoader
class DataLoader:
    """Class to load text documents from a file."""
    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = file_path
        self.encoding = encoding
    
    def load(self):
        loader = TextLoader(self.file_path, encoding=self.encoding)
        docs = loader.load()
        return docs