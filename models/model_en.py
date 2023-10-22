from typing import Any

import torch
from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer


class Llama():
    def __init__(self) -> None:
        self.tokenizer = LlamaTokenizer.from_pretrained("Llama-2-7b-chat-hf", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("Llama-2-7b-chat-hf", load_in_4bit=True, trust_remote_code=True)
        self.model = self.model.eval()
    
    def __call__(self, prompt: str):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(input_ids, max_new_tokens=100)
        output_str = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        return output_str
    
if __name__ == '__main__':
    model = Llama()
    print(model('Hi there. How\'s everything going?'))