import os
import re
import asyncio
from typing import List, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain & your custom modules
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader

from src.ingestion.loader import DataLoader
from src.ingestion.chunker import DataStripping
from src.ingestion.embedder import EmbeddingGenerator
from src.retrieval.retriever import Retriever
from src.retrieval.evaluator_class import LoRAEvaluator
from src.db.vector_db import VectorStore

# Optional: Tavily
try:
    from tavily import AsyncTavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    print("⚠️ Tavily not installed. Web search will fall back to static URLs.")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Set user agent to avoid warning
os.environ["USER_AGENT"] = "IBEX_CRAG_Assistant/1.0"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data.txt")
ADAPTER_ZIP = os.path.join(BASE_DIR, "src", "retrieval", "rag_lora_adapter_zip.zip")
EXTRACT_PATH = os.path.join(BASE_DIR, "src", "retrieval", "rag_lora_adapter")

# CRAG constants
RELEVANCE_THRESHOLD = 0      # not used directly; we use max_score logic
MAX_WEB_PAGES = 3
RETRIEVAL_K = 10

# Global variables (initialised at startup)
retriever = None
evaluator = None
chat = None

# -------------------------
# CRAG Helper Functions (exactly as in your working script)
# -------------------------
async def fetch_web_context_async(query: str) -> str:
    """Fetch web content – dynamic Tavily search if available, else static URLs."""
    # Option 1: Dynamic search with Tavily
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

    # Option 2: Static URLs (fallback)
    urls = [
        "https://ibex.co/",
        "https://waveix.ibex.co/",
        "https://www.ibex.co/industries/retail-ecommerce/",
        "https://www.ibex.co/staff-augmentation/",
        "https://www.ibex.co/contact/",
        "https://www.ibex.co/about/"
    ]
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
                if any(kw in content for kw in keywords):
                    all_text.append(doc.page_content[:4000])
        except Exception as e:
            print(f"⚠️ Error fetching {url}: {e}")

    combined = "\n\n".join(all_text[:MAX_WEB_PAGES])
    return combined[:4000]

async def evaluate_docs_sequentially(evaluator, query: str, retrieved_docs: List[Document]) -> List[Tuple[Document, int]]:
    scored = []
    for doc in retrieved_docs:
        result = await asyncio.to_thread(evaluator.evaluate, query, doc.page_content)
        match = re.search(r"\b([0-2])\b", result)
        score = int(match.group(1)) if match else 0
        scored.append((doc, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

def initialize_system():
    print("🚀 Initializing IBEX CRAG System...")

    loader = DataLoader(DATA_PATH)
    raw_docs = loader.load()
    sentences = DataStripping(raw_docs).data_splitting()
    docs = [Document(page_content=s) for s in sentences]

    embedding_model = EmbeddingGenerator(device="cpu").create_embeddings()
    vectorstore = VectorStore(docs, embedding_model).build()

    # Build retriever with k=RETRIEVAL_K
    retriever_obj = Retriever(vectorstore).get_retriever()
    if hasattr(retriever_obj, 'search_kwargs'):
        retriever_obj.search_kwargs = {"k": RETRIEVAL_K}
    elif hasattr(retriever_obj, 'k'):
        retriever_obj.k = RETRIEVAL_K
    else:
        retriever_obj = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    evaluator = LoRAEvaluator("Qwen/Qwen2-0.5B", ADAPTER_ZIP, EXTRACT_PATH)
    chat = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

    return retriever_obj, evaluator, chat

# -------------------------
# FastAPI Setup
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, evaluator, chat
    retriever, evaluator, chat = initialize_system()
    print("✅ IBEX CRAG System ready.")
    yield

app = FastAPI(lifespan=lifespan)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class Question(BaseModel):
    question: str

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(q: Question):
    query = q.question
    print(f"\n📨 Received query: {query}")

    # Step 1: Retrieve
    retrieved_docs = retriever.invoke(query)
    print(f"📚 Retrieved {len(retrieved_docs)} documents from local DB.")

    # Step 2: Score
    scored_docs = await evaluate_docs_sequentially(evaluator, query, retrieved_docs)

    # Log top scores
    for i, (doc, score) in enumerate(scored_docs[:3]):
        print(f"   Score {score}: {doc.page_content[:80]}...")

    # Step 3: Determine max score
    max_score = max([score for _, score in scored_docs]) if scored_docs else 0
    print(f"🏆 Maximum relevance score: {max_score}")

    # Step 4: Fallback decision
    use_web_fallback = (max_score == 0) or (len(scored_docs) < 2)

    if use_web_fallback:
        print("🔍 Low local relevance – performing web fallback...")
        web_context = await fetch_web_context_async(query)
        if web_context:
            context_text = web_context
            print(f"🌐 Retrieved web context length: {len(context_text)} chars")
        else:
            print("❌ No web context found. Using local documents anyway.")
            # FIX: unpack tuple to get Document
            context_text = "\n\n".join([doc.page_content for doc, _ in scored_docs[:3]])
    else:
        # Use top 3 relevant documents
        # FIX: unpack tuple to get Document
        context_text = "\n\n".join([doc.page_content for doc, _ in scored_docs[:3]])

    # Step 5: Generate answer
    prompt = f"""You are the official IBEX AI Assistant.

**Context (from IBEX internal or public sources):**
{context_text}

**Instructions:**
- Answer concisely and helpfully using ONLY the context above.
- Do NOT invent facts but you can rewrite the answer little bit so that its look little bit profesioanl
donot sound robotic proepr information convey to the our client...
- If the context does not contain the answer, politely say:
  "I couldn't find that specific information in the available materials. For further assistance, please contact IBEX support or visit ibex.co."
- Keep the tone professional and friendly.

**Question:** {query}
**Answer:"""

    try:
        response = await chat.ainvoke(prompt)
        answer = response.content
        print(f"💬 Generated answer: {answer[:200]}...")
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return JSONResponse(content={"answer": "An error occurred while generating the answer."}, status_code=500)