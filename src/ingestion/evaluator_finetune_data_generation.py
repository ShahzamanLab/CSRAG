import random
import json
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------
# Step 1: Load text file
# -----------------------------
file_path = r"D:\GITHUB_PROJECTS\CSRAG\data.txt"

loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()

# -----------------------------
# Step 2: Split text
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

split_docs = text_splitter.split_documents(docs)

print("Total chunks:", len(split_docs))

# -----------------------------
# Step 3: Queries
# -----------------------------
base_queries = [
    "What is BPO 2.0?",
    "Which countries have delivery centers?",
    "What services does ibex Digital provide?",
    "How does ibex CX handle feedback?",
    "What are ibex's key service pillars?"
]

# expand queries
queries = []
for q in base_queries:
    queries.append(q)
    queries.append(f"Explain {q.lower()}")
    queries.append(f"Give details about {q.lower()}")

# -----------------------------
# Step 4: Dataset generation
# -----------------------------
dataset = []

target_size = 1500

while len(dataset) < target_size:

    query = random.choice(queries)

    # correct document
    correct_doc = random.choice(split_docs)

    dataset.append({
        "query": query,
        "document": correct_doc.page_content,
        "label": 2
    })

    # wrong document
    wrong_doc = random.choice(split_docs)

    dataset.append({
        "query": query,
        "document": wrong_doc.page_content,
        "label": 0
    })

    # ambiguous document
    amb_doc = random.choice(split_docs)

    short_doc = amb_doc.page_content[:200]

    dataset.append({
        "query": query,
        "document": short_doc,
        "label": 1
    })

# shuffle dataset
random.shuffle(dataset)

dataset = dataset[:target_size]

# -----------------------------
# Step 5: Save JSONL
# -----------------------------
output_file = r"D:\GITHUB_PROJECTS\CSRAG\ibex_retrieval_dataset.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for row in dataset:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Dataset created successfully")
print("Total samples:", len(dataset))
print("Saved at:", output_file)