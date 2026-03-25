from langchain_text_splitters import RecursiveCharacterTextSplitter
from nltk.tokenize import regexp_tokenize

class DataSplitter:
    """Class to split documents into chunks."""
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split(self, docs):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_documents(docs)
        return chunks
    
class DataStripping:
    def __init__(self,dataset):
        self.dataset = dataset
    def data_splitting(self):
        texts = [str(doc) for doc in self.dataset]
        all_text = " ".join(texts)
        sentences = regexp_tokenize(all_text, pattern=r'[^.!?]+[.!?]?', gaps=False)
        sentences = [s.strip() for s in sentences]
        return sentences
    
