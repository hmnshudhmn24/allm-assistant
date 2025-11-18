from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class AllMAssistant:
    """Lightweight wrapper for causal LMs used for inference in this project."""
    def __init__(self, model_name_or_path: str, device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            # ensure pad token exists
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.7, top_p: float = 0.95):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=top_p)
        return self.tokenizer.decode(gen[0], skip_special_tokens=True)
