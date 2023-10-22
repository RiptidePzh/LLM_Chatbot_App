####################################################################################################
'''
If you want to extract chat history from your own WeChat account,
please first follow the instructions in README.md to get the encryption key,
then change the corresponding variables in this file, and run this file.
'''
####################################################################################################


import os
from pysqlcipher3 import dbapi2 as sqlite
import json

from data_extraction.wechat.clean_chat import clean_chat


# [CHANGE HERE] WeChat database location and output path on MAC
apple_username = 'your_username'
wc_userid = 'your_userid'
path = f'/Users/{apple_username}/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/{wc_userid}'
chat_path = path+"/Message"
contact_path = path+"/Contact"

output_path = './chat_history'

# [CHANGE HERE] Encryption KEY for db (see README for how to get the KEY)
source = """
replace with your own source key
"""   

raw_key = ''.join(i.partition(':')[2].replace('0x', '').replace(' ', '') for i in source.split('\n'))


def decrypt_db(conn, raw_key):
    '''
    Decrypt the db with the raw key.
    '''
    sql_statements = [
        f'PRAGMA key = "x\'{raw_key}\'"',
        'PRAGMA cipher_page_size = 1024',
        'PRAGMA kdf_iter = 64000',
        'PRAGMA cipher_hmac_algorithm = HMAC_SHA1',
        'PRAGMA cipher_kdf_algorithm = PBKDF2_HMAC_SHA1',
        'PRAGMA cipher_default_plaintext_header_size = 0'
    ]
    for statement in sql_statements:
        conn.execute(statement)

def extract_username_from_db(db_contact, friend_name):
    '''
    Extract the username of a friend from the contact db.
    '''
    conn = sqlite.connect(db_contact[0])
    decrypt_db(conn, raw_key)

    cursor = conn.cursor()
    columns = ['m_nsUsrName', 'm_nsRemark', 'nickname']
    cursor.execute(f"SELECT {', '.join(columns)} FROM WCContact WHERE m_nsRemark LIKE '%{friend_name}%' OR nickname LIKE '%{friend_name}%'")
    rows = cursor.fetchall()
    if rows == []:
        print(f"Couldn't find friend <{friend_name}>")
        return
    else:
        for i in range(len(rows)):
            wxid, remark, nickname = rows[i]
            print(f"Friend#{i} with remark <{remark}>, nickname <{nickname}> found")
        if len(rows) > 1:
            which = int(input("Enter the # of friend you're seaching for: "))
            wxid, remark, nickname = rows[which]

        import hashlib 
        friend_index = hashlib.md5(wxid.encode()).hexdigest()

        return friend_index

def extract_chat_from_db(db, friend_index, chat_data):
    '''
    Extract the chat data of a friend from the chat db.
    '''
    conn = sqlite.connect(db)
    decrypt_db(conn, raw_key)

    cursor = conn.cursor()
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%Chat_{friend_index}%' AND name NOT LIKE '%dels'")
    chat_tables = [row[0] for row in cursor.fetchall()]
    columns = ['msgCreateTime', 'msgContent', 'mesDes'] 
    for chat in chat_tables:
        try:
            cursor.execute(f"SELECT {', '.join(columns)} FROM {chat}")
            rows = cursor.fetchall()
            chat_data[chat] = [{columns[i]: row[i] for i in range(len(columns))} for row in rows]
            print(f"Chat data extracted from ...{db[-20:]}")
        except sqlite.OperationalError as e:
            print(f"Error occurred for table'{chat}': {e}")

    cursor.close()
    conn.close()

def export_to_json(chat_data, friend_name, output_path): 
    chat_data = chat_data[list(chat_data.keys())[0]]  # Remove the redundant key of chat table name
    language = input("Select chat language: (English or Chinese)")
    language = language.lower()
    chat_data = clean_chat(chat_data, language)
    json_data = json.dumps(chat_data, indent=4)
    output_js = output_path + f'/chat_{friend_name}/chat_{friend_name}.json'
    os.makedirs(os.path.dirname(output_js), exist_ok=True)
    with open(output_js, 'w', encoding='utf-8') as json_file:
        json_file.write(json_data)
    print(f"Chat data of friend <{friend_name}> stored in {output_js}")


# [RUN] Could run this .py after changing the paths and key
if __name__ == "__main__":
  
    db_chats = [os.path.join(chat_path, f) for f in os.listdir(chat_path) if f.endswith('.db')]
    db_contact = [os.path.join(contact_path, f) for f in os.listdir(contact_path) if f.endswith('.db')]

    friend_name = input("Enter your friend's name (WeChat remark or nickname):")
    friend_index = extract_username_from_db(db_contact, friend_name)

    chat_data = {}
    for db in db_chats:
        extract_chat_from_db(db, friend_index, chat_data)

    export_to_json(chat_data, friend_name, output_path) 