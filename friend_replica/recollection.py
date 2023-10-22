import json
import os

from langchain.prompts import PromptTemplate

from friend_replica.format_chat import format_chat_history, split_chat_data
from friend_replica.semantic_search import Chat


class LanguageModelwithRecollection():
    '''
    Wrap GPT4ALL models and Chat memory up.
    '''
    def __init__(self, 
                 model, 
                 chat: Chat, 
                 debug: bool=False, 
                 num_context: int=15,
                 num_search: int=3,
                 threshold: float=.5
                 ) -> None:
        self.model = model
        self.chat = chat if chat.chat_config else print("Please first pass chat_config to initialize Chat with one friend.")
        self.debug = debug
        self.num_context = num_context
        self.num_search = num_search
        self.threshold = threshold
                
    def generate_thoughts(self, friend_input):
        if self.chat.chat_config.language == "english":
            template = """[[INST]]<<SYS>>Please summarize the following sentence into less than three words.
            Reply with words only (less than three): 
            <</SYS>>
            
            {friend_input}

            [[/INST]] """
        
        else:
            template = """请用不超过三个中文短语概括句子内容，请只用这些中文短语作为回答：
            
            [Round 1]
            问：昨天那场音乐会真的爆炸好听，我哭死
            答：昨天 音乐会
            
            [Round 2]
            问：还记得我上周跟你提到的那本机器学习教材吗？
            答：上周 机器学习 教材
            
            [Round 3]
            问：{friend_input}
            答："""
            
        prompt = PromptTemplate(
            template=template, 
            input_variables=[
                'friend_input'
            ],
        )
        
        prompt_text = prompt.format(friend_input=friend_input)
        key_word = self.model(prompt_text) if self.chat.chat_config.language == "english" else key_word[len(prompt_text):]
        if self.debug:
            print(key_word)

        thoughts = self.chat.semantic_search(
            key_word, 
            friend_name=self.chat.chat_config.friend_name, 
            debug=False, 
            num_context=self.num_context, 
            k=self.num_search,
            threshold=self.threshold
        )
        
        return thoughts, key_word
    
    def summarize_recollection(
        self, 
        friend_input,
    ):
        '''
        Summarize recollected memories from thoughts.
        '''
        thoughts, key_word = self.generate_thoughts(friend_input)
        recollections = ['\n'.join(format_chat_history(thought, self.chat.chat_config, for_read=True, time=True)) for thought in thoughts]
        
        if self.debug:
            print(recollections)
            print(len(recollections))
        
        if self.chat.chat_config.language == "english":
            template = """[[INST]]<<SYS>>Please tell me when the following conversation took place, and
            summarize its main idea into only one sentence with regard to {key_words}: 
            <</SYS>>
            
            {recollections}

            One-sentence summary:
            [[/INST]] """

        else:
            template = """请告诉我下列对话的发生时间，并用一句话简短地概括它的整体内容，其中关键词为 {key_words}：
            
            [Round 1]
            对话：
            2023-08-16T11:33:44 from friend: 中午去哪吃？
            2023-08-16T11:35:14 from me: 西域美食吃吗
            2023-08-16T11:33:44 from friend: 西域美食
            2023-08-16T11:33:44 from friend: 好油啊
            2023-08-16T11:33:44 from friend: 想吃点好的
            2023-08-16T11:35:14 from me: 那要不去万达那边？
            2023-08-16T11:33:44 from friend: 行的行的
            
            总结：
            以上对话发生在2023年8月16日中午，我和我的朋友在商量中饭去哪里吃，经过商量后决定去万达。
            
            [Round 2]
            对话：
            {recollections}
            
            总结："""
        
        prompt = PromptTemplate(
            template=template, 
            input_variables=[
                'key_words',
                'recollections',
            ],
        )
        
        out = []
        for recollection in recollections:
            prompt_text = prompt.format(key_words=key_word, 
                                        recollections=recollection,
                                        )
            if self.chat.chat_config.language == "english":
                out0 = self.model(prompt_text).strip()
            else:
                out0 = self.model(prompt_text)[len(prompt_text):].strip()
            if self.debug:
                print(out0)
            out.append(out0)
            
        return out
    
    def generalize_personality(
        self, 
        chat_block,
    ):
        '''
        Generate personality for the chat and store the personality in json file for future usage.
        '''
        if self.chat.chat_config.language == "english":
            prompt_template = """[[INST]]<<SYS>>Be as concise and in-depth as possible. Follow the instructions and give response in one to two sentences.
            <</SYS>>
            Summarize in one to two sentences the personality of {my_name} and the relationship between {friend_name} and {my_name}, from the chat history given.
            '''
            {chat_history}
            '''
            Short summary:
            [[/INST]] """
            
        else:
            prompt_template = """
            从过往聊天记录中，总结{my_name}的性格特点，以及{my_name}和{friend_name}之间的人际关系。
            
            过往聊天：
            '''
            {chat_history}
            '''

            """
        
        prompt = PromptTemplate(
            template=prompt_template, 
            input_variables=[
                'my_name', 
                'friend_name', 
                'chat_history', 
            ],
        )
        
        prompt_text = prompt.format(
            my_name=self.chat.chat_config.my_name,
            friend_name=self.chat.chat_config.friend_name,
            chat_history='\n'.join(format_chat_history(chat_block, chat_config=self.chat.chat_config, for_read=True)),
        )
        
        if self.chat.chat_config.language == "english":
            out = self.model(prompt_text)
        else:
            out = self.model(prompt_text)[len(prompt_text):]

        # Store Personality Data
        time_interval = (chat_block[0]['msgCreateTime'], chat_block[-1]['msgCreateTime'])
        personality_entry = {
            'time_interval': time_interval,
            'personality': out,
        }
        output_js = os.path.join(self.chat.friend_path, f'personality_{self.chat.chat_config.friend_name}.json')
        if os.path.exists(output_js):
            with open(output_js, 'r', encoding='utf-8') as json_file:
                personality_data = json.load(json_file)
        else:
            personality_data = []
        personality_data.append(personality_entry)
        personality_data.sort(key=lambda x: x['time_interval'][0])
        with open(output_js, 'w', encoding='utf-8') as json_file:
            json.dump(personality_data, json_file, indent=4)

        return out

    def summarize_memory(
            self,
            chat_block,
    ):
        if self.chat.chat_config.language == "english":
            template = """[[INST]]<<SYS>>Be concise. Give response in one to two sentences.
            <</SYS>>
            Please summarize the main content of the following conversation.
            {chat_block}
            Short summary in one to two sentences:
            [[/INST]] """
            
        else:
            template = """请用一句话简短地概括下列聊天记录的整体思想.
            
            [Round 1]
            对话：
            friend: 中午去哪吃？
            me: 西域美食吃吗
            friend: 西域美食
            friend: 好油啊
            friend: 想吃点好的
            me: 那要不去万达那边？
            friend: 行的行的
            
            总结：
            以上对话发生在2023年8月16日中午，我和我的朋友在商量中饭去哪里吃，经过商量后决定去万达。
            
            [Round 2]
            对话：
            {chat_block}
            
            总结："""
        
        prompt = PromptTemplate(
            template=template, 
            input_variables=[
                "chat_block",
            ],
        )

        prompt_text = prompt.format(chat_block='\n'.join(format_chat_history(chat_block, chat_config=self.chat.chat_config, for_read=True)))

        return self.model(prompt_text) if self.chat.chat_config.language == "english" else self.model(prompt_text)[len(prompt_text):]

    def memory_archive(self):
        '''
        Generate memory archive for the chat.
        '''
        chat_blocks = split_chat_data(self.chat.chat_data)
        memory_archive = []
        for block in chat_blocks:
            memory = self.summarize_memory(block)
            topics = self.generate_thoughts(memory)
            time_interval = (block[0]['msgCreateTime'], block[-1]['msgCreateTime'])
            memory_entry = {
                time_interval: time_interval,
                memory: memory,
                topics: topics,
            }
            memory_archive.append(memory_entry)

        json_data = json.dumps(memory_archive, indent=4)
        output_js = os.path.join(self.chat.friend_path, f'memory_{self.chat.chat_config.friend_name}.json')
        os.makedirs(os.path.dirname(output_js), exist_ok=True)
        with open(output_js, 'w', encoding='utf-8') as json_file:
            json_file.write(json_data)
        
        return 

    def chat_with_recollection(
        self, 
        friend_input,
        current_chat: str = None,
    ):
        chat_blocks = split_chat_data(self.chat.chat_data)
        personality_data = os.path.join(self.chat.friend_path, f'personality_{self.chat.chat_config.friend_name}.json')
        if os.path.exists(personality_data):
            with open(personality_data,'r', encoding='utf-8') as json_file:
                personality_data = json.load(json_file)

            personality = personality_data[-1]['personality']
        else:
            personality = self.generalize_personality(chat_blocks[-1])

        recollections = self.generalize_recollection(friend_input)
        recollections = '\n'.join(recollections)
        
        if self.debug:
            print(recollections)
        
        if self.chat.chat_config.language == "english":
            prompt_template = """[[INST]]<<SYS>>You are roleplaying a robot with the personality of {my_name} in a casual online chat with {friend_name}.
            as described here: {personality}.
            Refer to Memory as well as Recent Conversation , respond to the latest message of {friend_name}.
            Start the short, casual response with {my_name}: 
            <</SYS>>
            
            Memory:
            '''
            {recollections}
            '''

            Recent Conversation:
            '''
            {recent_chat}
            '''

            {current_chat}
            {friend_name}: {friend_input}
            [[/INST]] """
            
        else:
            prompt_template = """接下来请你扮演一个在一场随性的网络聊天中拥有{my_name}性格特征的角色。
            首先从过往聊天记录中，学习总结{my_name}的性格特点，并掌握{my_name}和{friend_name}之间的人际关系。
            之后，运用近期聊天内容以及记忆中的信息，回复{friend_name}发送的消息。
            请用简短、随意的方式用{my_name}的身份进行回复：
            
            过往聊天：
            '''
            {chat_history}
            '''
            
            记忆：
            '''
            {recollections}
            '''

            近期聊天：
            '''
            {recent_chat}
            '''
 

            {current_chat}
            {friend_name}: {friend_input}
            
            """
        
        prompt = PromptTemplate(
            template=prompt_template, 
            input_variables=[
                'my_name', 
                'friend_name', 
                'chat_history', 
                'recent_chat', 
                'recollections',
                'friend_input',
                'current_chat'
            ],
        )
        
        prompt_text = prompt.format(
            my_name=self.chat.chat_config.my_name,
            friend_name=self.chat.chat_config.friend_name,
            personality=personality,
            recent_chat='\n'.join(format_chat_history(chat_blocks[-1], self.chat.chat_config)),
            recollections=recollections,
            friend_input=friend_input,
            current_chat=current_chat
        )
        
        if self.chat.chat_config.language == "english":
            out = self.model(prompt_text, stop='\n')
        else:
            out = self.model(prompt_text)[len(prompt_text):].split('\n')[0]
        
        return out
    
    def __call__(
        self, 
        friend_input,
        current_chat
    ):
        return self.chat_with_recollection(friend_input, current_chat)
    
    