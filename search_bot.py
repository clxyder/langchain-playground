'''
From: https://langchain.readthedocs.io/en/latest/modules/agents/getting_started.html

Need SerpAPI API key for Google Searches: https://serpapi.com/pricing
'''

import os
from configparser import ConfigParser

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

from constants import (
    CONFIG_DEFAULT_KEY,
    CONFIG_OPENAI_API_KEY,
    CONFIG_SERPAPI_API_KEY,
)

if __name__ == "__main__":
    # Initialize API Key
    config = ConfigParser()
    config.read("config.ini")
    os.environ[CONFIG_OPENAI_API_KEY] = config[CONFIG_DEFAULT_KEY][CONFIG_OPENAI_API_KEY]
    os.environ[CONFIG_SERPAPI_API_KEY] = config[CONFIG_DEFAULT_KEY][CONFIG_SERPAPI_API_KEY]

    # First, let's load the language model we're going to use to control the agent.
    llm = OpenAI(temperature=0)

    # Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    # Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

    # Now let's test it out!
    agent.run("Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?")