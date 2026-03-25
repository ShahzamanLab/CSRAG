import os
import re
import asyncio
from dotenv import load_dotenv

# LangChain & Specialized Imports
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader

# Custom modules (Assuming these handle the heavy lifting)
from src.ingestion.loader import DataLoader
from src.ingestion.chunker import DataStripping
from src.ingestion.embedder import EmbeddingGenerator
from src.retrieval.retriever import Retriever
from src.retrieval.evaluator_class import LoRAEvaluator
from src.db.vector_db import VectorStore


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") # Fixed typo

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data.txt")
ADAPTER_ZIP = os.path.join(BASE_DIR, "src", "retrieval", "rag_lora_adapter_zip.zip")
EXTRACT_PATH = os.path.join(BASE_DIR, "src", "retrieval", "rag_lora_adapter")

# -------------------------
# Robust Web Fetcher
# -------------------------
def fetch_web_context(query):
    urls = [
        "https://ibex.co/", "https://waveix.ibex.co/",
        "https://www.ibex.co/industries/retail-ecommerce/",
        "https://www.ibex.co/staff-augmentation/"
    ]
    
    all_text = ""
    # Extract meaningful keywords for fuzzy matching
    keywords = [w.lower() for w in query.split() if len(w) > 3]
    
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            for doc in docs:
                content = doc.page_content
                # If any significant keyword exists in the page
                if any(k in content.lower() for k in keywords):
                    all_text += content[:800] + "\n"
        except Exception as e:
            print(f"Error fetching {url}: {e}")

    return all_text[:4000]

# -------------------------
# Async Evaluation Logic
# -------------------------
async def evaluate_docs_async(evaluator, query, retrieved_docs):
    """Runs LoRA evaluations in parallel to save time."""
    tasks = []
    for doc in retrieved_docs:
        # Wrap the synchronous evaluator.evaluate in a thread to run in parallel
        tasks.append(asyncio.to_thread(evaluator.evaluate, query, doc.page_content))
    
    results = await asyncio.gather(*tasks)
    
    scored_docs = []
    for i, result in enumerate(results):
        # Improved Regex: Look for the last digit in case the model explains itself
        matches = re.findall(r"[0-2]", result)
        score = int(matches[0]) if matches else 0
        scored_docs.append((retrieved_docs[i].page_content, score))
    
    return scored_docs

# -------------------------
# Initialization
# -------------------------
def initialize_system():
    print("🚀 Initializing IBEX RAG System...")
    
    # 1. Load Data
    loader = DataLoader(DATA_PATH)
    raw_docs = loader.load()
    sentences = DataStripping(raw_docs).data_splitting()
    docs = [Document(page_content=s) for s in sentences]

    # 2. Vector DB
    embedding_model = EmbeddingGenerator(device="cpu").create_embeddings()
    vectorstore = VectorStore(docs, embedding_model).build()
    retriever = Retriever(vectorstore).get_retriever()

    # 3. Models
    evaluator = LoRAEvaluator("Qwen/Qwen2-0.5B", ADAPTER_ZIP, EXTRACT_PATH)
    chat = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1) # Using a stable Groq ID
    
    return retriever, evaluator, chat

# -------------------------
# Main Execution Entry
# -------------------------
async def main():
    retriever, evaluator, chat = initialize_system()

    while True:
        query = input("\n[User]: ")
        if query.lower() in ["exit", "quit"]: break

        # Step 1: Semantic Retrieval
        retrieved_docs = retriever.invoke(query)

        # Step 2: LoRA Re-ranking (Async)
        scored_docs = await evaluate_docs_async(evaluator, query, retrieved_docs)
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Step 3: Filtering logic
        filtered_docs = [doc for doc, score in scored_docs if score >= 1]

        # Step 4: Web Fallback
        if not filtered_docs:
            print("🔍 No local match. Searching IBEX web resources...")
            web_data = fetch_web_context(query)
            if web_data:
                filtered_docs = [web_data]
            else:
                print("❌ Information not found in company records.")
                continue

        # Step 5: Final Generation
        context_text = "\n\n".join(filtered_docs[:3])
        prompt = f"""You are the official IBEX AI Assistant. 
        Use the context below to answer accurately. 
        If unsure, state 'Not found in context'.

        Context: {context_text}
        Question: {query}
        Answer:"""

        response = chat.invoke(prompt)
        print(f"\n===== IBEX ASSISTANT =====\n{response.content}")

