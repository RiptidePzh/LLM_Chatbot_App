from typing import List, Dict, Literal
import re


def clean_chat(
        chat_data: List[Dict], 
        language: Literal['english', 'chinese'],
        ) -> List[Dict]:
    '''
    Clean chat history data for WeChat.
    Identify non-text <xml> messages by pattern search and replace them with meaningful text.
    '''
    if language == 'english':
        for msg in chat_data:
            if "<msgsource>" in msg['msgContent'] or '/refermsg' in msg['msgContent']:
                reply = re.search(r'<title>(.*?)</title>', msg['msgContent'], re.DOTALL).group(1)
                # Image quote
                if "imgdatahash" in msg['msgContent'] or "cdnmidimgurl" in msg['msgContent']:
                    msg['msgContent'] = f"(Quoting an image) {reply}" 
                # Text quote
                else:
                    likely_quotes = re.findall(r'<content>(.*?)</content>', msg['msgContent'], re.DOTALL)
                    quote_text = likely_quotes[1] if likely_quotes[0] == '' and len(likely_quotes) > 1 else likely_quotes[0]
                    if "title&gt;" in quote_text:    #special case: quoted msg is also a quote
                        quote_text = re.findall(r'title&gt;(.*?)&lt;/title', msg['msgContent'], re.DOTALL)[0]
                    msg['msgContent'] = f"(Quoting '{quote_text}') {reply}" 

            elif any(emoji in msg['msgContent'] for emoji in ["<msg><emoji", '<emojiinfo>', '<pattedUser>']):
                msg['msgContent'] = "(Sent a sticker)"

            elif "<img" in msg['msgContent']:
                msg['msgContent'] = "(Sent an image)"       

            elif "<videomsg" in msg['msgContent']:
                msg['msgContent'] = "(Sent a video)"

            elif "<media>" in msg['msgContent']:
                media_title = re.search(r'<desc><!\[CDATA\[(.*?)\]\]></desc>', msg['msgContent'], re.DOTALL).group(1)
                msg['msgContent'] = f"(Sent the link of an online video titled '{media_title}')"
    
            elif "<fileext>" in msg['msgContent']:
                file_title = re.search(r'<title>(.*?)</title>', msg['msgContent'], re.DOTALL).group(1)
                msg['msgContent'] = f"(Sent a file titled '{file_title}')"

            elif "<msg>" in msg['msgContent'] and "share" in msg['msgContent']:
                link_title = re.search(r'<title>(.*?)</title>', msg['msgContent'], re.DOTALL).group(1)
                msg['msgContent'] = f"(Shared a link of '{link_title}')"

            elif "<location" in msg['msgContent']:
                label = re.search(r'label="(.*?)"', msg['msgContent'], re.DOTALL).group(1)
                poiname = re.search(r'poiname="(.*?)"', msg['msgContent'], re.DOTALL).group(1)
                msg['msgContent'] = f"(Shared the map location {label} of {poiname})"

            if any(unwanted_pattern in msg['msgContent'] for unwanted_pattern in ['This content cannot be displayed', 'You recalled a message']):
                chat_data.remove(msg) 

    elif language == 'chinese':
        for msg in chat_data:
            if "<msgsource>" in msg['msgContent'] or '/refermsg' in msg['msgContent']:
                reply = re.search(r'<title>(.*?)</title>', msg['msgContent'], re.DOTALL).group(1)
                # Image quote
                if "imgdatahash" in msg['msgContent'] or "cdnmidimgurl" in msg['msgContent']:
                    msg['msgContent'] = f"(引用一张图片) {reply}" 
                # Text quote
                else:
                    likely_quotes = re.findall(r'<content>(.*?)</content>', msg['msgContent'], re.DOTALL)
                    quote_text = likely_quotes[1] if likely_quotes[0] == '' and len(likely_quotes) > 1 else likely_quotes[0]
                    if "title&gt;" in quote_text:    #special case: quoted msg is also a quote
                        quote_text = re.findall(r'title&gt;(.*?)&lt;/title', msg['msgContent'], re.DOTALL)[0]
                    msg['msgContent'] = f"(引用 '{quote_text}') {reply}" 

            elif any(emoji in msg['msgContent'] for emoji in ["<msg><emoji", '<emojiinfo>', '<pattedUser>']):
                msg['msgContent'] = "(发送了一个表情)"

            elif "<img" in msg['msgContent']:
                msg['msgContent'] = "(发送了一张图片)"       

            elif "<videomsg" in msg['msgContent']:
                msg['msgContent'] = "(发送了一个视频)"

            elif "<media>" in msg['msgContent']:
                media_title = re.search(r'<desc><!\[CDATA\[(.*?)\]\]></desc>', msg['msgContent'], re.DOTALL).group(1)
                msg['msgContent'] = f"(分享了一个网络视频，题为'{media_title}')"

            elif "<fileext>" in msg['msgContent']:
                file_title = re.search(r'<title>(.*?)</title>', msg['msgContent'], re.DOTALL).group(1)
                msg['msgContent'] = f"(发送了一个文件，题为'{file_title}')"

            elif "<msg>" in msg['msgContent'] and "share" in msg['msgContent']:
                link_title = re.search(r'<title>(.*?)</title>', msg['msgContent'], re.DOTALL).group(1)
                msg['msgContent'] = f"(分享了一个链接，题为'{link_title}')"

            elif "<location" in msg['msgContent']:
                label = re.search(r'label="(.*?)"', msg['msgContent'], re.DOTALL).group(1)
                poiname = re.search(r'poiname="(.*?)"', msg['msgContent'], re.DOTALL).group(1)
                msg['msgContent'] = f"(分享了一个地图定位，位置：{label}，地名：{poiname})"

            if any(unwanted_pattern in msg['msgContent'] for unwanted_pattern in ['This content cannot be displayed', 'You recalled a message', "你撤回了一条消息"]):
                chat_data.remove(msg) 
                               
    return chat_data



