import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.openai_api_key,
        )

    def invoke(
        self, messages: List[Dict[str, str]], temperature: Optional[float] = None
    ) -> str:
        try:
            langchain_messages = self._convert_messages(messages)

            if temperature is not None:
                llm = self.llm.with_config(temperature=temperature)
            else:
                llm = self.llm

            response = llm.invoke(langchain_messages)
            return response.content

        except Exception as e:
            logger.error(f"Error invoking LLM: {str(e)}")
            raise

    def _convert_messages(self, messages: List[Dict[str, str]]) -> List:
        langchain_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            else:
                langchain_messages.append(HumanMessage(content=content))

        return langchain_messages

    def get_embedding_model(self):
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model, api_key=settings.openai_api_key
        )


llm_service = LLMService()
