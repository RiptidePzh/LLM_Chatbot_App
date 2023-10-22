from typing import Any

import torch
from bigdl.llm.transformers import AutoModel
from transformers import AutoTokenizer


class ChatGLM():
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("chatglm2-6b-int4", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("chatglm2-6b-int4", trust_remote_code=True)
        self.model = self.model.eval()
    
    def __call__(self, prompt: str):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(input_ids, max_new_tokens=100)
        output_str = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        return output_str
    
if __name__ == '__main__':
    model = ChatGLM()
    print(model('请你扮演一个疯言疯语的傻子回答问题\n\n问：咋没去上课呢宝？\n\n答：'))