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
            return self._fallback_description(observation)
        prompt = (
            "You are an assistant summarizing a browser observation for an agent. "
            "Be concise, mention page title/url, key interactive elements, and any overlays."
        )
        message = HumanMessage(content=f"{prompt}\n\nObservation: {observation.model_dump()}")
        try:
            response = await self._client.ainvoke([message])
        except Exception:
            return self._fallback_description(observation)
        return response.content.strip()

    @staticmethod
    def _fallback_description(observation: Observation) -> str:
        page = observation.page
        top_elements = [elem.name or elem.role or elem.eid for elem in observation.interactive[:8]]
        overlays = ", ".join(observation.overlays[:3]) if observation.overlays else "none"
        return (
            f"Page '{page.title}' at {page.url}. "
            f"Interactive elements: {len(observation.interactive)}. "
            f"Examples: {', '.join(top_elements) if top_elements else 'none'}. "
            f"Overlays: {overlays}."
        )
