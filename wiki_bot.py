'''
From: https://dagster.io/blog/chatgpt-langchain
'''
from configparser import ConfigParser
from typing import List

import requests
from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores import VectorStore
from langchain.text_splitter import CharacterTextSplitter

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
    def __init__(self, openai_api_key: str, search_index: VectorStore) -> None:
        self.chain = load_qa_with_sources_chain(OpenAI(temperature=0, openai_api_key = openai_api_key))
        self.search_index = search_index
    
    def print_answer(self, question: str) -> None:
        print(
            self.chain(
                {
                    "input_documents": self.search_index.similarity_search(question, k=4),
                    "question": question,
                },
                return_only_outputs=True,
            )["output_text"]
        )

if __name__ == "__main__":
    # Initialize API Key
    config = ConfigParser()
    config.read("config.ini")
    openai_api_key = config[CONFIG_DEFAULT_KEY][CONFIG_OPENAI_API_KEY]

    # Generate sources
    wiki_topics = [
        "Unix",
        "Microsoft_Windows",
        "Linux",
        "Seinfeld",
        "Matchbox_Twenty",
        "Roman_Empire",
        "London",
        "Python_(programming_language)",
        "Monty_Python"
    ]
    sources = [get_wiki_data(x, first_paragraph_only=False) for x in wiki_topics]

    # chunk the source information to handle large documents
    source_chunks = []
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
    for source in sources:
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=source.metadata))

    # Improving efficiency using a vector space search engine
    try:
        search_index: VectorStore = FAISS.from_documents(source_chunks, OpenAIEmbeddings(openai_api_key = openai_api_key))
    except Exception as exc:
        print(exc)

    # Initilize Wikipedia chat bot
    wiki_chain = WikiChain(openai_api_key, search_index)

    while True:
        question = input("What question would you like to ask the WikiChain bot?\n> ")
        try:
            wiki_chain.print_answer(question)
        except Exception as exc:
            print(exc)

'''
Questions
- Who were the writers of Seinfeld?
- What are the main differences between Linux and Windows?
- What are the differences between Keynesian and classical economics?
- Which members of Matchbox 20 play guitar?
'''