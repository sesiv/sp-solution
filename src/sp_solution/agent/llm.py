from __future__ import annotations

from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from ..config import Settings
from ..models import Observation


class LLMDescriber:
    def __init__(self, settings: Settings) -> None:
        self._client: Optional[ChatOpenAI] = None
        if settings.openrouter_api_key:
            self._client = ChatOpenAI(
                model=settings.openrouter_model,
                api_key=settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
            )

    async def describe(self, observation: Observation) -> str:
        if not self._client:
            raise RuntimeError("OpenRouter client is not configured. Set OPENROUTER_API_KEY.")
        prompt = (
            "You are an assistant summarizing a browser observation for an agent. "
            "Be concise, mention page title/url, key interactive elements, and any overlays."
        )
        message = HumanMessage(content=f"{prompt}\n\nObservation: {observation.model_dump()}")
        try:
            response = await self._client.ainvoke([message])
        except Exception as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
        return response.content.strip()
