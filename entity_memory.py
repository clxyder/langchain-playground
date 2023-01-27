'''
from: https://langchain.readthedocs.io/en/latest/modules/memory/examples/entity_summary_memory.html

Conversation Prompt
- Deven & Sam are working on a hackathon project
- They are trying to add more complex memory structures to Langchain
- They are adding in a key-value store for entities mentioned so far in the conversation.
- What do you know about Deven & Sam?
- Sam is the founder of a company called Daimon.
- What do you know about Sam?
'''

import os

from langchain import OpenAI, ConversationChain
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

import utils
import memory

ENTITY_BUFFER_FILE = "entity_buffer.json"
ENTITY_STORE_FILE = "entity_store.json"

if __name__ == "__main__":
    # initialize API keys
    utils.intialize_api_keys()

    # initialize large language model
    llm = OpenAI(temperature=0)

    # we can also pass previous buffer and store to continue from a save point
    if os.path.exists(ENTITY_BUFFER_FILE) and os.path.exists(ENTITY_STORE_FILE):
        buffer = utils.load_json(ENTITY_BUFFER_FILE)
        store = utils.load_json(ENTITY_STORE_FILE)
        memory = memory.ConversationEntityMemory(llm=llm, buffer=buffer, store=store)
    else:
        memory = memory.ConversationEntityMemory(llm=llm)

    # intialize conversation chain
    conversation = ConversationChain(
        llm=llm, 
        # verbose=True,
        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        memory=memory
    )

    try:
        while True:
            user_input = input("USER: ")
            print(f"AI: {conversation.predict(input = user_input)}")
    except KeyboardInterrupt:
        utils.save_json(ENTITY_BUFFER_FILE, conversation.memory.buffer)
        utils.save_json(ENTITY_STORE_FILE, conversation.memory.store)
