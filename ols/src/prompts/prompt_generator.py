"""Prompt generator based on model / context."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

from .prompts import (
    QUERY_SYSTEM_INSTRUCTION,
    USE_CONTEXT_INSTRUCTION,
    USE_HISTORY_INSTRUCTION,
)


class GeneratePrompt:
    """Generate prompt dynamically."""

    def __init__(
        self,
        query: str,
        rag_context: list[str] = [],
        history: list[str] = [],
        system_instruction: str = QUERY_SYSTEM_INSTRUCTION,
    ):
        """Initialize prompt generator."""
        self._query = query
        self._history = history
        self._rag_context = rag_context
        self._sys_instruction = system_instruction

    def _generate_prompt_gpt(self):
        """Generate prompt for GPT."""
        prompt_message = []
        sys_intruction = self._sys_instruction
        llm_input_values = {"query": self._query}

        if len(self._rag_context) > 0:
            rag_context = ""
            for c in self._rag_context:
                rag_context = rag_context + "\nDocument:\n" + c + "\n"
            llm_input_values["context"] = rag_context

            sys_intruction = sys_intruction + USE_CONTEXT_INSTRUCTION

        if len(self._history) > 0:
            chat_history = []
            for h in self._history:
                if h.startswith("human: "):
                    chat_history.append(HumanMessage(content=h.lstrip("human: ")))
                else:
                    chat_history.append(AIMessage(content=h.lstrip("ai: ")))
            llm_input_values["chat_history"] = chat_history

            sys_intruction = sys_intruction + USE_HISTORY_INSTRUCTION

        if "context" in llm_input_values:
            sys_intruction = sys_intruction + "{context}"
        prompt_message.append(SystemMessagePromptTemplate.from_template(sys_intruction))

        if "chat_history" in llm_input_values:
            prompt_message.append(MessagesPlaceholder("chat_history"))

        # prompt_message.append(HumanMessage(content="{query}"))
        prompt_message.append(HumanMessagePromptTemplate.from_template("{query}"))
        return ChatPromptTemplate.from_messages(prompt_message), llm_input_values

    def _generate_prompt_granite(self):
        """Generate prompt for Granite."""
        prompt_message = "<|system|>" + self._sys_instruction
        llm_input_values = {"query": self._query}

        if len(self._rag_context) > 0:
            rag_context = ""
            for c in self._rag_context:
                rag_context = rag_context + "\n[Document]\n" + c + "\n[End]"
            llm_input_values["context"] = rag_context

            prompt_message = prompt_message + USE_CONTEXT_INSTRUCTION

        if len(self._history) > 0:
            prompt_message = prompt_message + USE_HISTORY_INSTRUCTION

            chat_history = ""
            for h in self._history:
                if h.startswith("human: "):
                    chat_history = chat_history + "\n<|user|>\n" + h.lstrip("human: ")
                else:
                    chat_history = chat_history + "\n<|assistant|>\n" + h.lstrip("ai: ")
            llm_input_values["chat_history"] = chat_history

        if "context" in llm_input_values:
            prompt_message = prompt_message + "{context}\n"
        if "chat_history" in llm_input_values:
            prompt_message = prompt_message + "{chat_history}\n"

        prompt_message = prompt_message + "<|user|>\n{query}\n<|assistant|>\n"
        return PromptTemplate.from_template(prompt_message), llm_input_values

    def generate_prompt(self, model):
        """Generate prompt."""
        model = model.lower()
        if "granite" in model:
            return self._generate_prompt_granite()
        else:
            return self._generate_prompt_gpt()
