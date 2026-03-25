import os
import re
from dotenv import load_dotenv

# Custom modules
from src.ingestion.loader import DataLoader
from src.ingestion.chunker import DataStripping
from src.ingestion.embedder import EmbeddingGenerator
from src.retrieval.retriever import Retriever
from src.retrieval.evaluator_class import LoRAEvaluator
from src.db.vector_db import VectorStore

from langchain_classic.schema import Document
from langchain_groq import ChatGroq

# -------------------------
# Load ENV
# -------------------------
load_dotenv()
GROQ_TOKEN = os.getenv("GROOQ_API_KEY")

# -------------------------
# 1️⃣ Load + preprocess data
# -------------------------
file_path = r"D:\GITHUB_PROJECTS\CSRAG\data.txt"

loader = DataLoader(file_path)
raw_docs = loader.load()

splitter = DataStripping(raw_docs)
sentences = splitter.data_splitting()

docs = [Document(page_content=s) for s in sentences]

# -------------------------
# 2️⃣ Embeddings + Vector DB
# -------------------------
embedding_generator = EmbeddingGenerator(device="cpu")
embedding_model = embedding_generator.create_embeddings()

vectorstore = VectorStore(docs, embedding_model).build()

# -------------------------
# 3️⃣ Retriever
# -------------------------
retriever_obj = Retriever(vectorstore)
retriever = retriever_obj.get_retriever()

# -------------------------
# 4️⃣ LoRA Evaluator Init
# -------------------------
BASE_MODEL = "Qwen/Qwen2-0.5B"
ADAPTER_ZIP = r"D:\GITHUB_PROJECTS\CSRAG\src\retrieval\rag_lora_adapter_zip.zip"
EXTRACT_PATH = r"D:\GITHUB_PROJECTS\CSRAG\src\retrieval\rag_lora_adapter"

evaluator = LoRAEvaluator(BASE_MODEL, ADAPTER_ZIP, EXTRACT_PATH)

# -------------------------
# 5️⃣ ChatGroq LLM
# -------------------------
chat = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2
)

# -------------------------
# 6️⃣ MAIN LOOP
# -------------------------
while True:
    query = input("\nEnter your query (or 'exit'): ")

    if query.lower() == "exit":
        break

    # -------------------------
    # Retrieve documents
    # -------------------------
    retrieved_docs = retriever._get_relevant_documents(query, run_manager=None)

    # -------------------------
    # Evaluate each document
    # -------------------------
    scored_docs = []

    for doc in retrieved_docs:
        result = evaluator.evaluate(query, doc.page_content)

        # Extract score (0/1/2)
        match = re.search(r"\b[0-2]\b", result)
        score = int(match.group()) if match else 0

        scored_docs.append((doc.page_content, score))

    # -------------------------
    # Sort by relevance
    # -------------------------
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # -------------------------
    # Debug (optional)
    # -------------------------
    print("\n--- Retrieved + Scored Docs ---")
    for i, (doc, score) in enumerate(scored_docs):
        print(f"{i+1}. Score: {score}")
        print(doc[:120])
        print("-" * 40)

    # -------------------------
    # Filter best docs
    # -------------------------
    filtered_docs = [doc for doc, score in scored_docs if score == 2]

    if not filtered_docs:
        filtered_docs = [doc for doc, score in scored_docs if score == 1]

    if not filtered_docs:
        print("⚠️ No relevant context found.")
        continue

    # Take top 3
    context_text = "\n\n".join(filtered_docs[:3])

    # -------------------------
    # Final Prompt
    # -------------------------
    prompt = f"""
You are a RAG-based assistant for IBEX Company.

Answer ONLY from the given context.
If answer is not in context, say "Not found in context".

Context:
{context_text}

Question:
{query}

Answer:
"""

    # -------------------------
    # LLM Call
    # -------------------------
    response = chat.invoke(input=prompt)

    print("\n===== FINAL ANSWER =====")
    print(response.content)