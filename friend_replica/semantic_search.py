import datetime
import json
import os
from typing import Dict, List

import torch
from friend_replica.format_chat import ChatConfig, split_chat_data
from model_paths import path_semantic_search
from sentence_transformers import SentenceTransformer


def compute_time_difference(timestamp1, timestamp2):
    '''
    Calculates time difference of two timestamps.
    '''
    # Convert timestamps to datetime objects
    dt1 = datetime.datetime.fromtimestamp(timestamp1)
    dt2 = datetime.datetime.fromtimestamp(timestamp2)
    
    # Calculate the time difference
    time_difference = dt2 - dt1

    # Extract days, hours, minutes, and seconds from the time difference
    days = time_difference.days
    seconds = time_difference.seconds
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if days < 0:
        hours = 23 - hours
        days = - days - 1
    return [days, hours, minutes, seconds]


class Chat():
    '''
    Python class that handles vectorizing previous chat history and semantic searches.
    '''
    def __init__(
        self,
        chat_history_path: str = './chat_history',
        embedding_model_path: str = '../models',
        device: str = 'cuda',
        chat_config: ChatConfig = None,
    ) -> None:
        '''
        Args:
            chat_history_path: Path to all chat histories
            embedding_model_path: Path to pre-downloaded model weights 
            device: Where to place embedding model 
        '''
        self.root = chat_history_path
        self.model = SentenceTransformer(
            path_semantic_search,
            cache_folder=embedding_model_path,
            device=device,
        )
        # full model path: '../models/sentence-transformers_stsb-xlm-r-multilingual'
        self.device = device
        self.chat_config = chat_config
        if self.chat_config:
            self.friend_path = os.path.join(chat_history_path, f'chat_{chat_config.friend_name}')
            self.chat_data = json.load(open(os.path.join(self.friend_path, f'chat_{chat_config.friend_name}.json'), 'r'))
            self.chat_blocks = split_chat_data(self.chat_data)
            self.vectorize_one_friend()

    def vectorize_one_friend(self):
        '''
        Create vector memory data for just one friend. Stores embeddings where .json file is located.
        Only msgContent will be vectorized.
        '''
        msg_ls = []
        for msg in self.chat_data:
            msg_ls.append(msg['msgContent']) 
        
        vector = self.model.encode(msg_ls, convert_to_tensor=True) 
        torch.save(vector, os.path.join(self.friend_path, f'chat_{self.chat_config.friend_name}' + '.pth'))
        print(f'chat_{self.chat_config.friend_name} vectorized')

    def vectorize(self):
        '''
        Create vector memory database. Stores embeddings where .json file is located.
        Only msgContent will be vectorized.
        '''
        friends = os.listdir(self.root)
        for friend in friends:
            path = os.path.join(self.root, friend)
            js_path = os.path.join(path, friend + '.json')
            chat = json.load(open(js_path, 'r'))
            
            msg_ls = []
            for msg in chat:
                msg_ls.append(msg['msgContent']) 
            
            vector = self.model.encode(msg_ls, convert_to_tensor=True) 
            torch.save(vector, os.path.join(path, friend + '.pth'))
            print(f'{friend} vectorized')
            
    def __getitem__(self, args):
        '''
        Retrieve according messages.
        '''
        friend_name, idx = args
        if self.chat_config:
            return self.chat_data[idx]
        else:
            js_path = os.path.join(self.root, f'chat_{friend_name}', f'chat_{friend_name}.json')
            chat = json.load(open(js_path, 'r'))
            return chat[idx]
            
    def semantic_search(
        self,
        queries: List[str],
        friend_name: str = None,
        threshold: float = .5,
        k: int = 5, 
        time_difference: tuple[int, int] = (0, 2),
        num_context: int = 20, 
        debug: bool = False
    ) -> List[List[Dict]]:
        '''
        Performs semantic search on certain chat history.
        For every retrieved result, return messages created within a time difference of 2 hours of 
        that result (no more than 20 total messages).
        
        Args:
            quries: list of sentences or words to look up
            friend_name: name of the friend
            threshold: ignore if score of a retrieved message is below
            k: number of messages to retrieve (if score above threshold)
            time_difference: tuple that indicates threshold for selecting message contexts, formatted as (days, hours)
            num_context: total number of messages within a context
            
        Retures a list containing context windows, each context window is a list of messages.
        '''
        days_threshold, hours_threshold = time_difference
        if self.chat_config:
            len_chat = len(self.chat_data)
            friend_path = self.friend_path
            if friend_name and friend_name != self.chat_config.friend_name:
                print("Friend name for semantic search not consistent with the Chat history.")
                return
            else:
                friend_name = self.chat_config.friend_name
        else:
            friend_path = os.path.join(self.root, f'chat_{friend_name}')
            len_chat = len(json.load(open(os.path.join(friend_path, f'chat_{friend_name}' + '.json'), 'r')))
        
        vector_path = os.path.join(friend_path, f'chat_{friend_name}' + '.pth')
        
        try:
            v = torch.load(vector_path, map_location=torch.device(self.device))
        except:
            print(f'chat_{friend_name}.json not vectorized. Please run vectorize() first. ')
            return
        
        v_search = self.model.encode(queries, convert_to_tensor=True)
        score = v @ v_search.squeeze() / (torch.norm(v, p=2, dim=-1) * torch.norm(v_search, p=2))
        values, indices = torch.topk(score, k=k)
        indices = indices[torch.where(values >= threshold)]
        
        ls = indices.tolist()
        ls.sort()
        
        if debug:
            print(ls)
        
        #Check if there are contexts overlapping under current configuration
        #Merge if so
        context_ranges = []
        i = 0
        while i < len(ls):
            a = max(ls[i] - num_context // 2, 0)
            while i + 1 < len(ls) and ls[i + 1] - ls[i] <= num_context:
                i += 1
            b = min(ls[i] + num_context // 2, v.shape[0])
            context_ranges.append((a, b))
            i += 1
        if debug:
            print(context_ranges)   
                
        out_msg = []
        ptr2 = 0
        for ptr1 in range(len(ls)):
            idx = ls[ptr1]
            msg = self[friend_name, idx]
            timestamp1 = msg['msgCreateTime']
            
            flag = True
            while idx > context_ranges[ptr2][1]:
                ptr2 += 1
                flag = False
            if flag and ptr1 != 0:
                continue
            a, b = context_ranges[ptr2]
            if debug:
                print('context', a, b)
            
            context = []
            for idx2 in range(a, min(b + 1, len_chat)):
                msg2 = self[friend_name, idx2]
                timestamp2 = msg2['msgCreateTime']
                
                if debug:
                    print(compute_time_difference(timestamp1, timestamp2))
                days, hours, _, _ = compute_time_difference(timestamp1, timestamp2)
                
                #Ignore if the two messages are not created within the same day
                #or time difference is greater than 2 hours
                if days > days_threshold or hours >= hours_threshold:
                    continue
                
                context.append(self[friend_name, idx2])
                if debug:
                    print(idx2)
                
            out_msg.append(context)
        
        return out_msg if out_msg else "Related memory not found."
        
    def semantic_search_with_org_msg(
        self,
        queries: List[str],
        friend_name: str = None,
        threshold: float = .5,
        k: int = 5, 
        time_difference: tuple[int, int] = (0, 2),
        num_context: int = 20, 
        debug: bool = False
    ) -> List[List[Dict]]:
        '''
        Performs semantic search on certain chat history.
        For every retrieved result, return messages created within a time difference of 2 hours of 
        that result (no more than 20 total messages).
        
        Args:
            quries: list of sentences or words to look up
            friend_name: name of the friend
            threshold: ignore if score of a retrieved message is below
            k: number of messages to retrieve (if score above threshold)
            time_difference: tuple that indicates threshold for selecting message contexts, formatted as (days, hours)
            num_context: total number of messages within a context
            
        Retures a list containing context windows, each context window is a list of messages.
        '''
        days_threshold, hours_threshold = time_difference
        if self.chat_config:
            len_chat = len(self.chat_data)
            friend_path = self.friend_path
            if friend_name and friend_name != self.chat_config.friend_name:
                print("Friend name for semantic search not consistent with the Chat history.")
                return
            else:
                friend_name = self.chat_config.friend_name
        else:
            friend_path = os.path.join(self.root, f'chat_{friend_name}')
            len_chat = len(json.load(open(os.path.join(friend_path, f'chat_{friend_name}' + '.json'), 'r')))
        
        vector_path = os.path.join(friend_path, f'chat_{friend_name}' + '.pth')
        
        try:
            v = torch.load(vector_path, map_location=torch.device(self.device))
        except:
            print(f'chat_{friend_name}.json not vectorized. Please run vectorize() first. ')
            return
        
        v_search = self.model.encode(queries, convert_to_tensor=True)
        score = v @ v_search.squeeze() / (torch.norm(v, p=2, dim=-1) * torch.norm(v_search, p=2))
        values, indices = torch.topk(score, k=k)
        indices = indices[torch.where(values >= threshold)]
        
        ls = indices.tolist()
        ls.sort()
        
        if debug:
            print(ls)
        
        #Check if there are contexts overlapping under current configuration
        #Merge if so
        context_ranges = []
        i = 0
        while i < len(ls):
            a = max(ls[i] - num_context // 2, 0)
            while i + 1 < len(ls) and ls[i + 1] - ls[i] <= num_context:
                i += 1
            b = min(ls[i] + num_context // 2, v.shape[0])
            context_ranges.append((a, b))
            i += 1
        if debug:
            print(context_ranges)   
        
        org_msg = []
        p = 0
        for r in context_ranges:
            l = []
            while ls[p] >= r[0] and ls[p] <= r[1]:
                l.append(self[friend_name, ls[p]])
                p += 1
                if p >= len(ls):
                    break 
            org_msg.append(l)
                
        out_msg = []
        ptr2 = 0
        for ptr1 in range(len(ls)):
            idx = ls[ptr1]
            msg = self[friend_name, idx]
            timestamp1 = msg['msgCreateTime']
            
            flag = True
            while idx > context_ranges[ptr2][1]:
                ptr2 += 1
                flag = False
            if flag and ptr1 != 0:
                continue
            a, b = context_ranges[ptr2]
            if debug:
                print('context', a, b)
                        
            context = []
            for idx2 in range(a, min(b + 1, len_chat)):
                msg2 = self[friend_name, idx2]
                timestamp2 = msg2['msgCreateTime']
                
                if debug:
                    print(compute_time_difference(timestamp1, timestamp2))
                days, hours, _, _ = compute_time_difference(timestamp1, timestamp2)
                
                #Ignore if the two messages are not created within the same day
                #or time difference is greater than 2 hours
                if days > days_threshold or hours >= hours_threshold:
                    continue
                
                context.append(self[friend_name, idx2])
                if debug:
                    print(idx2)
                
            out_msg.append(context)
        
        return out_msg, org_msg if out_msg else "Related memory not found."