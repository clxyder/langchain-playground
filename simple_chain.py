
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

from utils import intialize_api_keys

if __name__ == "__main__":
    # Initialize API Key
    intialize_api_keys()

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
