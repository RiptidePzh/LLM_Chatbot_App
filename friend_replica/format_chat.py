from typing import List, Dict, Literal
from datetime import datetime, timedelta

def concatenate_chat(chat_data: List[Dict]) -> List[Dict]:
    '''
    Concatenate neighboring messages from the same user,
    to match few-shot learning prompt format for language models.
    '''
    conc_chat = []
    prev_msg = chat_data[0]
    for msg in chat_data[1:]:
        if prev_msg['mesDes'] != msg['mesDes']:
            conc_chat.append(prev_msg)
            prev_msg = msg.copy()
        else:
            prev_msg['msgContent'] = prev_msg['msgContent'] + " " + msg['msgContent']
            prev_msg['msgCreateTime'] = msg['msgCreateTime']  # Use the latest msg timestamp for concatenated messages (subject to change)
    if prev_msg not in conc_chat:
        conc_chat.append(prev_msg)
    return conc_chat

def split_chat_data(
        chat_data: List[Dict], 
        max_msg_interval: timedelta=timedelta(days=1),
        max_block_interval: timedelta=timedelta(days=3),
        min_message_count: int=5,
        ) -> List:
    '''
    Split chat history into blocks of messages based on time intervals.
    '''
    blocks = []
    current_block = []
    last_timestamp = None
    start_timestamp = None

    for message in chat_data:
        if last_timestamp is None:
            current_block.append(message)
            start_timestamp = message['msgCreateTime']
            last_timestamp = message['msgCreateTime']
        else:
            msg_interval = message['msgCreateTime'] - last_timestamp
            block_interval = message['msgCreateTime'] - start_timestamp
            if (len(current_block) >= min_message_count 
                and (msg_interval > max_msg_interval.total_seconds() 
                     or block_interval > max_block_interval.total_seconds())):
                blocks.append(current_block)
                current_block = [message]
                start_timestamp = message['msgCreateTime']
                last_timestamp = message['msgCreateTime']
            else:
                current_block.append(message)
                last_timestamp = message['msgCreateTime']
        
    if current_block:
        blocks.append(current_block)
    
    blocks = [concatenate_chat(block) for block in blocks]

    return blocks


class ChatConfig:
    def __init__(self,
                 my_name: str="assistant",
                 friend_name: str="user",
                 language: Literal["english", "chinese"]="english",
                 ):
        self.my_name = my_name
        self.friend_name = friend_name
        self.language = language

def format_chat_msg(
        msg: Dict,
        chat_config: ChatConfig,
        for_read: bool,
        time: bool
        ) -> Dict:
    '''
    Format each chat msg for prompt.
    1. Put my_name and friend_name from chat_config into chat history.
    2. If not for_read, format msg according to community template, for few-shot learning chat generation tasks.
       If for_read, format msg into readable text, for summarization tasks.
    3. Under for_read, if time is True, include timestamp in the formatted msg.
    ''' 
    role = chat_config.my_name if msg['mesDes'] == 0 else chat_config.friend_name
    content = msg['msgContent']
    if for_read:
        if time: 
            str_time = datetime.fromtimestamp(msg['msgCreateTime']).strftime('%Y-%m-%dT%H:%M')
            formatted_msg = f'{str_time}, {role}: {content}'
        else:
            formatted_msg = f'{role}: {content}'
    else:
        formatted_msg ={
            "role": role,
            "content": content,
            } 
    return formatted_msg


def format_chat_history(
        chat_data: List[Dict],
        chat_config: ChatConfig,
        for_read: bool=False,
        time: bool=False,
        ) -> List[Dict]:
    '''
    Format chat history for prompt or printout.
    '''
    chat_history = [format_chat_msg(msg, chat_config, for_read, time) for msg in chat_data]
    return chat_history

def st_format_chat_msg(
        msg: Dict,
        chat_config: ChatConfig,
        for_read: bool,
        time: bool
        ) -> Dict:
    '''
    Format each chat msg for prompt.
    1. Put my_name and friend_name from chat_config into chat history.
    2. If not for_read, format msg according to community template, for few-shot learning chat generation tasks.
       If for_read, format msg into readable text, for summarization tasks.
    3. Under for_read, if time is True, include timestamp in the formatted msg.
    '''
    role = chat_config.my_name if msg['mesDes'] == 0 else chat_config.friend_name
    content = msg['msgContent']
    if for_read:
        if time:
            str_time = datetime.fromtimestamp(msg['msgCreateTime']).strftime('%Y-%m-%dT%H:%M')
            formatted_msg = {"role":role, "time":str_time, "content": content}
        else:
            formatted_msg = {"role":role, "content": content}
    else:
        formatted_msg = {
            "role": role,
            "content": content,
            }
    return formatted_msg

def st_format_chat_history(
        chat_data: List[Dict],
        chat_config: ChatConfig,
        for_read: bool=False,
        time: bool=False,
        ) -> List[Dict]:
    '''
    Format chat history for prompt or printout.
    '''
    chat_history = [format_chat_msg(msg, chat_config, for_read, time) for msg in chat_data]
    return chat_history