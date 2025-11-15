import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import APIError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings
from app.services.cache import cache_service
from app.services.monitoring import monitoring_service

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    @abstractmethod
    def invoke(self, messages: List[dict], temperature: Optional[float] = None) -> str:
        pass

    @abstractmethod
    def get_embedding_model(self):
        pass

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        pass


class OpenAIProvider(BaseLLMProvider):
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
    def invoke(self, messages: List[dict], temperature: Optional[float] = None) -> str:
        start_time = time.time()
        temp = temperature if temperature is not None else settings.llm_temperature

        cached_result = cache_service.get_llm_result(messages, temp)
        if cached_result:
            duration = time.time() - start_time
            monitoring_service.record_llm_call(settings.llm_model, 0, duration)
            return cached_result

        try:
            langchain_messages = self._convert_messages(messages)

            if temperature is not None:
                llm = self.llm.with_config(temperature=temperature)
            else:
                llm = self.llm

            response = llm.invoke(langchain_messages)
            result = response.content

            duration = time.time() - start_time
            tokens = self.estimate_tokens(result)
            monitoring_service.record_llm_call(settings.llm_model, tokens, duration)

            cache_service.set_llm_result(messages, temp, result)

            return result

        except (RateLimitError, APIError) as e:
            logger.warning(f"Retryable error invoking LLM: {str(e)}")
            monitoring_service.record_error("llm_api_error", str(e))
            raise
        except Exception as e:
            logger.error(f"Error invoking LLM: {str(e)}")
            monitoring_service.record_error("llm_error", str(e))
            raise

    def _convert_messages(self, messages: List[dict]) -> List:
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
        return OpenAIEmbeddings(
            model=settings.embedding_model, api_key=settings.openai_api_key
        )

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4


class LLMService:
    def __init__(self, provider: Optional[BaseLLMProvider] = None):
        self.provider = provider or OpenAIProvider()

    def invoke(self, messages: List[dict], temperature: Optional[float] = None) -> str:
        return self.provider.invoke(messages, temperature)

    def get_embedding_model(self):
        return self.provider.get_embedding_model()


llm_service = LLMService()
