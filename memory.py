
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from pydantic import BaseModel

from langchain.chains.base import Memory
from langchain.chains.conversation.prompt import (
    ENTITY_EXTRACTION_PROMPT,
    ENTITY_SUMMARIZATION_PROMPT,
)
from langchain.chains.conversation.memory import _get_prompt_input_key
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate


class ConversationEntityMemory(Memory, BaseModel):
    """Entity extractor & summarizer to memory."""

    buffer: List[str] = []
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    """Prefix to use for AI generated responses."""
    llm: BaseLLM
    entity_extraction_prompt: BasePromptTemplate = ENTITY_EXTRACTION_PROMPT
    entity_summarization_prompt: BasePromptTemplate = ENTITY_SUMMARIZATION_PROMPT
    memory_keys: List[str] = ["entities", "history"]  #: :meta private:
    output_key: Optional[str] = None
    input_key: Optional[str] = None
    store: Dict[str, Optional[str]] = {}
    entity_cache: List[str] = []
    k: int = 3

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return ["entities", "history"]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        # create LLM chain
        chain = LLMChain(llm=self.llm, prompt=self.entity_extraction_prompt)

        # determine prompt_input_key
        if self.input_key is None:
            prompt_input_key = _get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key

        # query LLM for entities from the last k entries of the buffer
        output = chain.predict(
            history = "\n".join(self.buffer[-self.k :]),
            input = inputs[prompt_input_key],
        )

        # extract entities from LLM output
        if output.strip() == "NONE":
            entities = []
        else:
            entities = [w.strip() for w in output.split(",")]

        # update entity cache dictionary
        entity_summaries = {}
        for entity in entities:
            entity_summaries[entity] = self.store.get(entity, "")
        self.entity_cache = entities

        # create output dictionary
        return {
            "history": "\n".join(self.buffer[-self.k :]),
            "entities": entity_summaries,
        }

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        # determine prompt_input_key
        if self.input_key is None:
            prompt_input_key = _get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key

        # determine output_key
        if self.output_key is None:
            if len(outputs) != 1:
                raise ValueError(f"One output key expected, got {outputs.keys()}")
            output_key = list(outputs.keys())[0]
        else:
            output_key = self.output_key

        # query LLM summary for each entity
        # TODO only update relevant entities?
        for entity in self.entity_cache:
            # TODO do we need to re-create the LLMChain for each entity?
            chain = LLMChain(llm=self.llm, prompt=self.entity_summarization_prompt)

            # get key value store for entity
            existing_summary = self.store.get(entity, "")

            # query LLM to create a summary for the entity with it's existing
            # summary, written input and previous k buffer entries
            output = chain.predict(
                summary=existing_summary,
                history="\n".join(self.buffer[-self.k :]),
                input=inputs[prompt_input_key],
                entity=entity,
            )

            # update store with entity summary
            self.store[entity] = output.strip()

        # update the buffer with the human and AI entries
        # TODO add timestamp prefix
        human = f"{self.human_prefix}: " + inputs[prompt_input_key]
        ai = f"{self.ai_prefix}: " + outputs[output_key]
        new_lines = "\n".join([human, ai])
        self.buffer.append(new_lines)

        # TODO write buffer to json file
        # TODO write store to json file

    def clear(self) -> None:
        """Clear memory contents."""
        self.buffer = []
        self.store = {}