# rag_lora_finetune_colab.py
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
DATA_FILE = r"D:\GITHUB_PROJECTS\CSRAG\ibex_retrieval_dataset.jsonl"  
MODEL_NAME = "Qwen/Qwen2-0.5B"       # small model for GPU
OUTPUT_DIR = "/content/rag_lora_adapter"
MAX_LENGTH = 256
BATCH_SIZE = 4   # T4 GPU can handle 4, increase if memory allows
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD DATASET
# -----------------------------
print("Loading dataset...")
dataset = load_dataset("json", data_files=DATA_FILE)["train"]

# -----------------------------
# FORMAT PROMPTS
# -----------------------------
def format_prompt(example):
    return {
        "text": f"""
Query: {example['query']}

Document: {example['document']}

Classify relevance:
0 = wrong
1 = ambiguous
2 = correct

Answer: {example['label']}
"""
    }

dataset = dataset.map(format_prompt)

# -----------------------------
# LOAD TOKENIZER AND MODEL (4-bit)
# -----------------------------
print("Loading tokenizer and model in 4-bit...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# -----------------------------
# ADD LoRA ADAPTER
# -----------------------------
print("Adding LoRA adapters...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.to(DEVICE)

# -----------------------------
# TOKENIZE DATASET
# -----------------------------
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

dataset = dataset.map(tokenize)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# TRAINING LOOP
# -----------------------------
print("Starting training...")
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

model.train()
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch in tqdm(train_loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["input_ids"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1} loss: {loss.item():.4f}")

# -----------------------------
# SAVE LoRA MODEL
# -----------------------------
print(f"Saving LoRA adapter to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# -----------------------------
# INFERENCE EXAMPLE
# -----------------------------
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
    output = model.generate(**inputs, max_new_tokens=5)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()

# Test example
test_query = "What is BPO 2.0?"
test_doc = "BPO 2.0 is ibex's forward-looking framework for customer experience..."
result = evaluate(test_query, test_doc)
print(f"Evaluator output: {result}")

# -----------------------------
# OPTIONAL: ZIP FOR DOWNLOAD
# -----------------------------
import shutil
shutil.make_archive("/content/rag_lora_adapter_zip", 'zip', OUTPUT_DIR)
print("LoRA adapter zipped and ready for download.")