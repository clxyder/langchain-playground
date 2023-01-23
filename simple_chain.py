import os
from configparser import ConfigParser

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

from constants import (
    CONFIG_DEFAULT_KEY,
    CONFIG_OPENAI_API_KEY,
)

if __name__ == "__main__":
    # Initialize API Key
    config = ConfigParser()
    config.read("config.ini")
    os.environ[CONFIG_OPENAI_API_KEY] = config[CONFIG_DEFAULT_KEY][CONFIG_OPENAI_API_KEY]

    # Initialize OpenAI LLM model and prompt template
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    
    # Initializa LLM Chain
    chain = LLMChain(llm=llm, prompt=prompt, verbose = True)

    # Execute LLM chain
    print(chain.run("colorful socks"))
