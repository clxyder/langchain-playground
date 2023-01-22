from configparser import ConfigParser
from typing import List

import requests
from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document

from constants import (
    CONFIG_DEFAULT_KEY,
    CONFIG_OPENAI_API_KEY,
)


def get_wiki_data(title: str, first_paragraph_only: bool = False) -> Document:
    url = f"https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&explaintext=1&titles={title}"
    if first_paragraph_only:
        url += "&exintro=1"
    data = requests.get(url).json()
    return Document(
        page_content=list(data["query"]["pages"].values())[0]["extract"],
        metadata={"source": f"https://en.wikipedia.org/wiki/{title}"},
    )

class WikiChain:
    def __init__(self, config, sources: List[Document]) -> None:
        self.chain = load_qa_with_sources_chain(OpenAI(temperature=0, openai_api_key = config[CONFIG_DEFAULT_KEY][CONFIG_OPENAI_API_KEY]))
        self.sources = sources
    
    def print_answer(self, question: str) -> None:
        print(
            self.chain(
                {
                    "input_documents": self.sources,
                    "question": question,
                },
                return_only_outputs=True,
            )["output_text"]
        )

if __name__ == "__main__":
    # Initialize API Key
    config = ConfigParser()
    config.read("config.ini")


    wiki_topics = ["Unix", "Microsoft_Windows", "Linux", "Seinfeld"]
    sources = [get_wiki_data(x, first_paragraph_only=True) for x in wiki_topics]
    wiki_chain = WikiChain(config, sources)

    while True:
        question = input("What question would you like to ask the WikiChain bot?\n> ")
        wiki_chain.print_answer(question)
