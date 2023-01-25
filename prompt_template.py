'''
from: https://langchain.readthedocs.io/en/latest/modules/prompts/getting_started.html
'''

from langchain import PromptTemplate

if __name__ == "__main__":
    # An example prompt with no input variables
    no_input_prompt = PromptTemplate(input_variables=[], template="Tell me a joke.")
    print(no_input_prompt.format())
    # -> "Tell me a joke."

    # An example prompt with one input variable
    one_input_prompt = PromptTemplate(input_variables=["adjective"], template="Tell me a {adjective} joke.")
    print(one_input_prompt.format(adjective="funny"))
    # -> "Tell me a funny joke."

    # An example prompt with multiple input variables
    multiple_input_prompt = PromptTemplate(
        input_variables=["adjective", "content"], 
        template="Tell me a {adjective} joke about {content}."
    )
    print(multiple_input_prompt.format(adjective="funny", content="chickens"))
    # -> "Tell me a funny joke about chickens."
