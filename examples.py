# %%
from langchain.llms import GPT4All
from models.model_cn import ChatGLM

from friend_replica.format_chat import ChatConfig, split_chat_data, format_chat_history
from friend_replica.recollection import LanguageModelwithRecollection
from friend_replica.semantic_search import *


# %%
### Main
# Initialize Chat with one friend
chat_config = ChatConfig(
    my_name="Rosie",
    friend_name="王",
    language="chinese",
)
chat_with_friend = Chat(device='cuda', chat_config=chat_config)
chat_blocks = split_chat_data(chat_with_friend.chat_data)
print([len(c) for c in chat_blocks])


# %%
# Semantic Memory Search among chat history with this friend
queries = ["sad"]
print("Searching for:", queries)
contexts = chat_with_friend.semantic_search(queries)
for context in contexts:
    print('\n'.join(format_chat_history(context, chat_with_friend.chat_config, for_read=True, time=True)))
    print()

'''
Example Output (English):

Searching for: ['sad']
2023-08-31T22:40, Rosie: I ruined my day with that pack of junk food tho [Sob]
2023-08-31T22:40, Andrew: (Sent a sticker)
2023-08-31T22:41, Rosie: Woke up at 8, did core exercise, studied, did hip exercise, studied then finally chips wtf
2023-08-31T22:41, Andrew: Wtf 🤷‍♂️🤷‍♂️🤷‍♂️
2023-08-31T22:41, Andrew: You were in a such good combo
2023-08-31T22:41, Andrew: And
2023-08-31T22:41, Andrew: Ruined it …
2023-08-31T22:41, Andrew: Hope it was good chips 
2023-08-31T22:42, Andrew: Not the shitty Lays 🫠
2023-08-31T22:42, Andrew: And 
2023-08-31T22:42, Andrew: Not a fucked up flavor 🫠
2023-08-31T22:42, Andrew:  If you dare telling me
2023-08-31T22:42, Andrew: It was Lays with Seaweed flavor…
2023-08-31T22:43, Andrew: (Sent a sticker)
2023-08-31T22:56, Rosie: no it’s not even real chips
2023-08-31T22:57, Rosie: (Sent an image)
2023-08-31T23:00, Andrew: (Sent a sticker)
2023-08-31T23:00, Andrew: Nooooooo

'''

# %%
# Load Memory Recollection Model
model = GPT4All(model="llama-2-7b-chat.ggmlv3.q4_0.bin", allow_download=False)
# model = ChatGLM()
m = LanguageModelwithRecollection(model, chat_with_friend, debug=True)

# %%
# Memory Archive Generation
# m.memory_archive(chat_blocks)

# For one Chat Block
print('\n'.join(format_chat_history(chat_blocks[1], chat_with_friend.chat_config, for_read=True, time=True)))
print()
print(m.summarize_memory(chat_blocks[1]))

'''
Example Output (English):
2023-08-11T05:09, Eddie: (Sent the link of an online video titled 'China is really out of this world ❤️ 🥵')
2023-08-11T11:45, Rosie: I’ve never heard about the places in this video hahahah but let’s go Dunhuang maybe (Sent an image) (Sent an image) You could ride camel and see these cave arts  (Sent an image) I’m bored. How’s the place you’re traveling at? Send me some pics 
2023-08-11T15:17, Eddie: Let’s absolutely go to Dunhuang When would be the best period?  (Sent an image) (Sent an image) (Sent a video) (Sent an image)
2023-08-11T15:32, Rosie: Peaceful village 
2023-08-12T08:13, Eddie: Not very hahaha They were shooting fireworks every night lol
2023-08-12T10:43, Rosie: wow quite romantic and good for couple travelling hahahah, have fun bb
2023-08-13T00:23, Eddie: Love u thx ❤️ (Sent the link of an online video titled '你都去过哪里呢😍 #旅行 #爱中国')
2023-08-13T00:27, Rosie: hahahha I could see you can’t wait to travel here Be sure to not do it during the October National Day holiday tho  It would be freaking crowded everywhere 
2023-08-13T03:21, Eddie: I can’t wait you’re right haha I’m gonna chill in october i guess Visit beijing likely
 
Rosie and Eddie are discussing travel destinations, with Eddie expressing interest in visiting Dunhuang, while Rosie recommends avoiding the October National Day holiday due to crowds. 
They also share images and videos of their respective locations, with Eddie looking forward to traveling in China and Rosie mentioning that she can't wait to see Eddie's adventures.
'''

# %%
# Personality and Relationship Analysis from chat history
print(m.generalize_personality(chat_blocks[1]))

'''
Example Output (English):
Rosie is a fun-loving and adventurous person who enjoys traveling and exploring new places. 
Eddie and Rosie have a friendly and casual relationship, with Eddie seeking advice from Rosie on their shared interest in traveling in China.
'''

# %%
# Chatbot Friend Replica
print(m(friend_input="what do you think"))


# %%
### Semantic Search
# You may construct the whole Memory Search database (with all friends' chat history)
c = Chat(device='mps')
c.vectorize()

# This allows you to do memory search freely with multiple friends
queries = ["good restaurants"]
friends = ["Eddie", "Andrew"]

contexts = {friend_name: c.semantic_search(queries, friend_name=friend_name) for friend_name in friends}
for (friend_name, context) in contexts.items():
    print(f"friend_name:{friend_name}")
    print(context)
    print()

# %%
### Freind Replica Chat Session
# 
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
    