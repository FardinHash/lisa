import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIError, RateLimitError
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APIError)),
        reraise=True,
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

        except (RateLimitError, APIError) as e:
            logger.warning(f"Retryable error invoking LLM: {str(e)}")
            raise
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
