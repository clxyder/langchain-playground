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

from pprint import pprint

from langchain import OpenAI, ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

import utils

if __name__ == "__main__":
    utils.intialize_api_keys()

    llm = OpenAI(temperature=0)
    conversation = ConversationChain(
        llm=llm, 
        # verbose=True,
        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        memory=ConversationEntityMemory(llm=llm)
    )

    try:
        while True:
            user_input = input("USER: ")
            print(f"AI: {conversation.predict(input = user_input)}")
    except KeyboardInterrupt:
        pprint(conversation.memory.buffer)
        pprint(conversation.memory.store)
