import os
import re
import asyncio
from typing import List, Tuple
from dotenv import load_dotenv

# LangChain & Specialized Imports
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader

# Custom modules (your existing classes)
from src.ingestion.loader import DataLoader
from src.ingestion.chunker import DataStripping
from src.ingestion.embedder import EmbeddingGenerator
from src.retrieval.retriever import Retriever
from src.retrieval.evaluator_class import LoRAEvaluator
from src.db.vector_db import VectorStore

# Optional: Tavily for dynamic web search
try:
    from tavily import AsyncTavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    print("⚠️ Tavily not installed. Web search will fall back to static URLs.")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # optional

# Set a default user agent to avoid warning
os.environ["USER_AGENT"] = "IBEX_CRAG_Assistant/1.0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data.txt")
ADAPTER_ZIP = os.path.join(BASE_DIR, "src", "retrieval", "rag_lora_adapter_zip.zip")
EXTRACT_PATH = os.path.join(BASE_DIR, "src", "retrieval", "rag_lora_adapter")

# -------------------------
# CRAG constants (tuned for better relevance)
# -------------------------
RELEVANCE_THRESHOLD = 0      # Keep all documents for now; we'll rely on max_score logic
MAX_WEB_PAGES = 3            # Number of web pages to fetch in fallback
RETRIEVAL_K = 10             # Retrieve more documents for scoring

# -------------------------
# Web fallback – dynamic search first, then static URLs
# -------------------------
async def fetch_web_context_async(query: str) -> str:
    """
    Fetch relevant content from the web.
    - If Tavily is available and API key is set, perform a dynamic search.
    - Otherwise fall back to static IBEX URLs with keyword filtering.
    Returns combined text truncated to 4000 chars.
    """
    # ---- Option 1: Dynamic search with Tavily ----
    if TAVILY_AVAILABLE and TAVILY_API_KEY:
        try:
            tavily = AsyncTavilyClient(api_key=TAVILY_API_KEY)
            response = await tavily.search(
                query,
                search_depth="advanced",
                max_results=MAX_WEB_PAGES
            )
            snippets = [result["content"] for result in response["results"]]
            combined = "\n\n".join(snippets)
            return combined[:4000]
        except Exception as e:
            print(f"⚠️ Tavily search error: {e}. Falling back to static URLs.")

    # ---- Option 2: Static URLs (fallback) ----
    urls = [
        "https://ibex.co/",
        "https://waveix.ibex.co/",
        "https://www.ibex.co/industries/retail-ecommerce/",
        "https://www.ibex.co/staff-augmentation/",
        "https://www.ibex.co/contact/",        # Added contact page
        "https://www.ibex.co/about/"            # Added about page
    ]

    # Extract meaningful keywords (longer than 3 letters)
    keywords = {w.lower() for w in query.split() if len(w) > 3}
    if not keywords:
        keywords = {"ibex"}

    all_text = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs = await asyncio.to_thread(loader.load)
            for doc in docs:
                content = doc.page_content.lower()
                # More permissive: include if at least one keyword matches
                if any(kw in content for kw in keywords):
                    # Keep first 4000 chars per page (was 2000)
                    all_text.append(doc.page_content[:4000])
        except Exception as e:
            print(f"⚠️ Error fetching {url}: {e}")

    # Combine and cap total length
    combined = "\n\n".join(all_text[:MAX_WEB_PAGES])
    return combined[:4000]

# -------------------------
# Safe evaluator (sequential, thread‑safe)
# -------------------------
async def evaluate_docs_sequentially(evaluator, query: str, retrieved_docs: List[Document]) -> List[Tuple[Document, int]]:
    """
    Runs the LoRA evaluator one by one to avoid GPU thread conflicts.
    Returns list of (document, score) sorted by score descending.
    """
    scored = []
    for doc in retrieved_docs:
        result = await asyncio.to_thread(evaluator.evaluate, query, doc.page_content)
        # Extract score (0,1,2) with regex
        match = re.search(r"\b([0-2])\b", result)
        score = int(match.group(1)) if match else 0
        scored.append((doc, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

# -------------------------
# Initialization
# -------------------------
def initialize_system():
    print("🚀 Initializing IBEX CRAG System...")

    # 1. Load and process your local knowledge base
    loader = DataLoader(DATA_PATH)
    raw_docs = loader.load()
    sentences = DataStripping(raw_docs).data_splitting()
    docs = [Document(page_content=s) for s in sentences]

    # 2. Build vector store and retriever
    embedding_model = EmbeddingGenerator(device="cpu").create_embeddings()
    vectorstore = VectorStore(docs, embedding_model).build()
    
    # Get the retriever from your custom class
    retriever_obj = Retriever(vectorstore).get_retriever()
    
    # Try to set the number of documents to retrieve
    if hasattr(retriever_obj, 'search_kwargs'):
        retriever_obj.search_kwargs = {"k": RETRIEVAL_K}
    elif hasattr(retriever_obj, 'k'):
        retriever_obj.k = RETRIEVAL_K
    else:
        # If the retriever is a LangChain retriever, we can use as_retriever with k
        print("⚠️ Could not set retrieval k directly; trying vectorstore.as_retriever.")
        retriever_obj = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    # 3. Load the evaluator (LoRA model)
    evaluator = LoRAEvaluator("Qwen/Qwen2-0.5B", ADAPTER_ZIP, EXTRACT_PATH)

    # 4. LLM for final generation
    chat = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

    return retriever_obj, evaluator, chat

# -------------------------
# Main CRAG Loop
# -------------------------
async def main():
    retriever, evaluator, chat = initialize_system()

    while True:
        query = input("\n[User]: ")
        if query.lower() in ["exit", "quit"]:
            break

        # ---- Step 1: Semantic retrieval ----
        retrieved_docs = retriever.invoke(query)
        print(f"📚 Retrieved {len(retrieved_docs)} documents from local DB.")

        # ---- Step 2: Relevance scoring (LoRA) ----
        scored_docs = await evaluate_docs_sequentially(evaluator, query, retrieved_docs)

        # ---- Log top scores for debugging ----
        for i, (doc, score) in enumerate(scored_docs[:3]):
            print(f"   Score {score}: {doc.page_content[:80]}...")

        # ---- Step 3: Determine best score ----
        max_score = max([score for _, score in scored_docs]) if scored_docs else 0
        print(f"🏆 Maximum relevance score: {max_score}")

        # ---- Step 4: Decide whether to use web fallback ----
        # Trigger web fallback if the best local document scores 0 (or if there are very few)
        use_web_fallback = (max_score == 0) or (len(scored_docs) < 2)

        if use_web_fallback:
            print("🔍 Low local relevance – performing web fallback...")
            web_context = await fetch_web_context_async(query)
            if web_context:
                context_text = web_context
                print(f"🌐 Retrieved web context length: {len(context_text)} chars")
            else:
                print("❌ No web context found. Using local documents anyway.")
                # Use the top 3 local documents (even if they scored 0)
                context_text = "\n\n".join([doc.page_content for doc in scored_docs[:3]])
        else:
            # Use top 3 relevant documents (sorted by score)
            context_text = "\n\n".join([doc.page_content for doc in scored_docs[:3]])

        # ---- Step 5: Generate final answer with improved prompt ----
        prompt = f"""You are the official IBEX AI Assistant.

**Context (from IBEX internal or public sources):**
{context_text}

**Instructions:**
- Answer concisely and helpfully using ONLY the context above.
- Do NOT invent facts.
- If the context does not contain the answer, politely say:
  "I couldn't find that specific information in the available materials. For further assistance, please contact IBEX support or visit ibex.co."
- Keep the tone professional and friendly.

**Question:** {query}
**Answer:"""

        try:
            response = await chat.ainvoke(prompt)
            print(f"\n===== IBEX ASSISTANT =====\n{response.content}")
        except Exception as e:
            print(f"❌ Generation error: {e}")

if __name__ == "__main__":
    asyncio.run(main())