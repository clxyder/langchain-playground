'''
From: https://langchain.readthedocs.io/en/latest/modules/agents/examples/custom_agent.html
'''

from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, SerpAPIWrapper, LLMChain

import utils

if __name__ == "__main__":

    utils.intialize_api_keys()

    # create Google API wrapper
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name = "Search",
            func=search.run,
            description="useful for when you need to answer questions about current events"
        )
    ]

    # create prompt
    prefix = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:"""
    suffix = """Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Args"

    Question: {input}
    {agent_scratchpad}"""

    # create prompt template
    prompt = ZeroShotAgent.create_prompt(
        tools, 
        prefix=prefix, 
        suffix=suffix, 
        input_variables=["input", "agent_scratchpad"]
    )
    print(prompt.template)

    # create language model
    llm = OpenAI(temperature=0)

    # create LLM chain with prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # create list of available tools
    tool_names = [tool.name for tool in tools]

    # create ZeroShotAgent 
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)

    # create agent executor
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    # execute agent with query
    agent_executor.run("How many people live in canada?")
