import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import zipfile
import os

# -------------------------
# Paths
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ADAPTER_ZIP = r"D:\GITHUB_PROJECTS\CSRAG\src\retrieval\rag_lora_adapter_zip.zip"  # your zip file
EXTRACT_PATH = r"D:\GITHUB_PROJECTS\CSRAG\src\retrieval\rag_lora_adapter"  # folder to extract
BASE_MODEL = "Qwen/Qwen2-0.5B"

# -------------------------
# Unzip adapter if not already extracted
# -------------------------
if not os.path.exists(EXTRACT_PATH):
    with zipfile.ZipFile(ADAPTER_ZIP, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)
        print(f"Adapter extracted to {EXTRACT_PATH}")

# -------------------------
# Load tokenizer and base model
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, 
    device_map="auto" if DEVICE=="cuda" else None,
    torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32
)

# -------------------------
# Load LoRA adapter on top
# -------------------------
model = PeftModel.from_pretrained(base_model, EXTRACT_PATH)
model.to(DEVICE)
model.eval()

# -------------------------
# Inference function
# -------------------------
def evaluate(query, document):
    prompt = f"""
Query: {query}

Document: {document}

Classify relevance:
0 = wrong
1 = ambiguous
2 = correct

Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    output = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()

# -------------------------
# Example usage
# -------------------------
query = "What is BPO 2.0?"
document = "BPO 2.0 is ibex's forward-looking framework for customer experience..."
result = evaluate(query, document)
print(f"Evaluator output: {result}")