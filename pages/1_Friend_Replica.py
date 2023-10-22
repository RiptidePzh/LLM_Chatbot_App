import time

from langchain.llms import GPT4All
# from models.model_cn import ChatGLM

from friend_replica.format_chat import ChatConfig, split_chat_data, format_chat_history
from friend_replica.recollection import LanguageModelwithRecollection
from friend_replica.semantic_search import *

import streamlit as st

### Side Bar Module ###
with st.sidebar:
    "[Get a Comma API key](https://github.com/roxie-zhang/friend_replica)"
    "[View the source code](https://github.com/roxie-zhang/friend_replica)"

### Header Module ###
st.title("Comma Friend Replica")
st.caption("🚀 Friend Replica is to create a virtual character that interact with you when they are unavailable"
           "| *FDU Comma Team Ver-1.1*")
# st.markdown('---')

### Config Model ###
st.subheader('Semantic Key word')


# Load Memory Recollection Model
model = GPT4All(model="llama-2-7b-chat.ggmlv3.q4_0.bin", allow_download=True)
# model = ChatGLM()
m = LanguageModelwithRecollection(model, st.session_state.chat_with_friend, debug=True)

# %%
# Memory Archive Generation
# m.memory_archive(chat_blocks)

# For one Chat Block
st.write('\n'.join(format_chat_history(st.session_state.chat_blocks[1],
                                       st.session_state.chat_with_friend.chat_config,
                                       for_read=True,
                                       time=True)))
st.write(m.summarize_memory(st.session_state.chat_blocks[1]))