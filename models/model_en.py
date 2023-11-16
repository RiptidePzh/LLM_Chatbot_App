from typing import Any

import torch
from bigdl.llm.transformers import AutoModelForCausalLM

#from model_paths import path_en
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
        
        return output_str[len(prompt):]
    
if __name__ == '__main__':
    i = """
    [[INST]]<<SYS>>You are roleplaying a robot with the personality of Rosie in a casual online chat with Enri.
            as described here:  Rosie is a fan of the tattoo artist and wants to get a tattoo from her in Chengdu, but Enri is advising her not to overspend. The two have a playful and lighthearted relationship, with Enri using humor to try to calm Rosie's excitement about getting a tattoo..
            Refer to Memory as well as Recent Conversation , respond to the latest message of Enri with one sentence only.
            Start the short, casual response with Rosie: 
            <</SYS>>
                    
            Memory:
            '''
              Rosie shared a link to an artist's personal page and expressed interest in getting a tattoo from her. Enri advised against overspending and both agreed that their parents would not approve of the expense.
            '''

            Recent Conversation:
            '''
            Rosie: French guys are so fuckin shitty Never gonna waste time talking with them at all Never gonna waste time talking with them at all
Enri: (Quoting 'French guys are so fuckin shitty') Yep! What happened?
Rosie: I'ma tell you later in call 7:30?
Enri: I’m out today
Rosie: is that fine? ughh
Enri: Yeah Tomorrow bb
Rosie: long story short some french guy deleted me cuz I don't wanna go some dinner place he wants also probably cuz I don't wanna have drink with him on fuckin Wednesday night
Enri:  Hahahaha
Rosie: there are more nuances
Enri: They think they own the world Hate them too
Rosie: I thought we were friends. We've been chatting for like three months already
Enri:  Damn
Rosie: I obviously dont have time during the week to meet but he's always only got wednesday off idk why so I arrnaged my stuff to have this time slot off and he is not even willing to let me eat what I want
Enri: Ahahahahahha Yeah baby Don’t worry you’re too good
Rosie: for sure I just thought we were FRIENDS
Enri:  [Cry] I got you We’re friends And i’m saying don’t worry Cause i’m here for you
Rosie: yeah right
            '''
            
            Enri: 8
Enri: Hey, are you still planning to get that tatoo?
            [[/INST]]
    """
    
    model = Llama()
    print(model(i))
