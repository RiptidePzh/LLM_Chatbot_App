# Friend Replica

**Clone A ChatBot With Your Memory**  
**Local LLM Model For All-round Privacy**

Supports:
* Semantic Search
* Personality Generation
* Relationship Analysis
* Memory Archive

## Introduction

**Memory Archive** 🧠  
   A new way to archive your treasured memories, present fresh as new retrospects of past thoughts and feelings, as well as vivid flashbacks of shared moments with your loved ones.  

**Friend Replica** 😶‍🌫️  
   Chat with a digital clone of yourself, your friend, or your past significant other, generated from your shared real world memory archive.

## Features

1. **Semantic Search**: Recollect memory in chat even with vague queries (Blazingly Fast! Enabled by torch & sentence_piece model)
2. **Personality & Relationship Observer**: A private observer for your relationship with friends
3. **Memory Archive**: A new way to archive your treasured memories with friends
4. **Chat with Recollection**: Your replica bot accessing memory from the archive through chains of search


## Usage
### 1. Web App
> We've developed a user friendly interface for our application. We are working on deploying a test demo onto an open url!


<img width="1243" alt="image" src="https://github.com/RiptidePzh/LLM_Chatbot_App/assets/85790664/ac882af4-4453-47d3-a0f3-4791b67c149e">
<img width="1248" alt="image" src="https://github.com/RiptidePzh/LLM_Chatbot_App/assets/85790664/3ef309ad-db9b-46dc-833e-02e70c4bf26f">

--[todo] Add full model & function support.

--[todo] Deploying on Web-Server Url: https://fducommallm.streamlit.app/


### 2. Terminal for Replica Chat
To run a complete chat session (including personality generalization, memory search, etc.), run following command in terminal:
```bash
sh run_chat.sh
```
Parameters
| field | description | default |
|---|---|---|
| my_name | Your name in chat history. | Rosie |
| friend_name | Friend's name in chat history. | 王 |
| language | To use chinese llm or english llm. | chinese |
| device | Device to run llm on. | cpu |
| debug | Whether to print debugging information (intermediate results of CoT, etc.) | Rosie |
| embedding_model_path | Path to local embedding model. Automatically downloads when set to None. | None |
| chinese_model_path | Path to local chinese llm. Automatically downloads when set to None. | None |
| english_model_path | Path to local chinese llm. Automatically downloads when set to None. | None |


### 3. Python Script
You may read through the following or refer to `examples.py` directly
#### Initialize Chat with one friend
* For Chat with one friend, **first initialize the chat ** by passing the `chat_config` and spliting long chat history into blocks.  
  Remember to specify your device for better inference speed.
```
chat_config = ChatConfig(
    my_name="Rosie",
    friend_name="王",
    language="chinese",
)
chat_with_friend = Chat(device='cuda', chat_config=chat_config)
chat_blocks = split_chat_data(chat_with_friend.chat_data)
```
#### Semantic Search
* Semantic Search with one friend
```
# Semantic Memory Search among chat history with this friend
queries = ["sad"]
print("Searching for:", queries)
contexts = chat_with_friend.semantic_search(queries)
for context in contexts:
    print('\n'.join(format_chat_history(context, chat_with_friend.chat_config, for_read=True, time=True)))
    print()
```
* Semantic Search in the Memory Archive
You may construct the whole Memory Search database (with all friends' chat history), which allows you to do memory search freely with multiple friends.
```
c = Chat(device='mps')
c.vectorize()

queries = ["good restaurants"]
friends = ["Eddie", "Andrew"]

contexts = {friend_name: c.semantic_search(queries, friend_name=friend_name) for friend_name in friends}
for (friend_name, context) in contexts.items():
    print(f"friend_name:{friend_name}")
    print(context)
    print()
```
#### Memory Recollection
* Load MemoryRecollectionModel
```
# Load Memory Recollection Model
from models.model_cn import ChatGLM
model = ChatGLM()
m = LanguageModelwithRecollection(model, chat_with_friend, debug=True)
```
* Personality and Relationship Analysis 
```
m.generalize_personality(chat_blocks[1])
```
* Summarize Memory
```
m = LanguageModelwithRecollection(model, chat_with_friend, debug=True)
print('\n'.join(format_chat_history(chat_blocks[1], chat_with_friend.chat_config, for_read=True)))
print()
print(m.summarize_memory(chat_blocks[1]))
```
* Memory Archive Generation (this may take a long time)
```
m.memory_archive(chat_blocks)
```
#### Friend Replica Chat Session
```
### Freind Replica Chat Session
model = ChatGLM()

chat_config = ChatConfig(
    my_name="Rosie",
    friend_name="王",
    language="chinese",
)
chat_with_friend = Chat(device='cpu', chat_config=chat_config)
m = LanguageModelwithRecollection(model, chat_with_friend, debug=True)

q = ''
current_chat = []
while q:
    q = input("")
    a = m(q, '\n'.join(current_chat))
    current_chat.append(chat_config.friend_name + ': ' + q)
    current_chat.append(chat_config.my_name + ': ' + a)
```

### Prerequisites

`BigDL` library for Intel chip devices, tested for the Chinese language model `chatglm2-6b`  
`GPT4ALL` for other devices, tested for the English language model `llama-2-chat-7b`

