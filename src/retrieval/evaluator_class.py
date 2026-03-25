import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import zipfile
import os

class LoRAEvaluator:
    """
    LoRA-based evaluator for query-document relevance classification.
    """

    def __init__(self, base_model_name, adapter_zip, extract_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model_name = base_model_name
        self.adapter_zip = adapter_zip
        self.extract_path = extract_path

        # Extract adapter if needed
        self._extract_adapter()

        # Load tokenizer and base model
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )

        # Load LoRA adapter on top
        self.model = PeftModel.from_pretrained(self.base_model, self.extract_path)
        self.model.to(self.device)
        self.model.eval()

    def _extract_adapter(self):
        """Extract the LoRA adapter zip if the folder does not exist."""
        if not os.path.exists(self.extract_path):
            with zipfile.ZipFile(self.adapter_zip, 'r') as zip_ref:
                zip_ref.extractall(self.extract_path)
                print(f"Adapter extracted to {self.extract_path}")

    def evaluate(self, query, document, max_new_tokens=50):
        """Evaluate a single query-document pair."""
        prompt = f"""
Query: {query}

Document: {document}

Classify relevance:
0 = wrong
1 = ambiguous
2 = correct

Answer:
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    BASE_MODEL = "Qwen/Qwen2-0.5B"
    ADAPTER_ZIP = r"D:\GITHUB_PROJECTS\CSRAG\src\retrieval\rag_lora_adapter_zip.zip"
    EXTRACT_PATH = r"D:\GITHUB_PROJECTS\CSRAG\src\retrieval\rag_lora_adapter"

    evaluator = LoRAEvaluator(BASE_MODEL, ADAPTER_ZIP, EXTRACT_PATH)

    query = "What is BPO 2.0?"
    document = "BPO 2.0 is ibex's forward-looking framework for customer experience..."
    result = evaluator.evaluate(query, document)
    print(f"Evaluator output: {result}")