from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..config import Settings
from ..models import Action, Observation


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


@dataclass
class ActionContext:
    current_goal: str | None
    user_message: str | None
    observation: Observation | None
    steps_taken: int
    max_steps: int
    recent_steps: list[str]


class LLMActionPlanner:
    def __init__(self, settings: Settings) -> None:
        self._client: Optional[ChatOpenAI] = None
        if settings.openrouter_api_key:
            self._client = ChatOpenAI(
                model=settings.openrouter_model,
                api_key=settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=0.2,
            )
        self._logger = logging.getLogger(__name__)

    def available(self) -> bool:
        return self._client is not None

    async def next_action(self, context: ActionContext) -> Action:
        if not self._client:
            raise RuntimeError("OpenRouter client is not configured. Set OPENROUTER_API_KEY.")
        prompt = self._build_prompt(context)
        self._logger.info("LLM prompt:\n%s", prompt)
        messages = [
            SystemMessage(content=_ACTION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        try:
            response = await self._client.ainvoke(messages)
        except Exception as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
        raw = response.content.strip()
        self._logger.info("LLM response:\n%s", raw)
        action = _parse_action_response(raw)
        self._logger.info("LLM action: %s", action.model_dump())
        return action

    def _build_prompt(self, context: ActionContext) -> str:
        observation = context.observation.model_dump() if context.observation else None
        return json.dumps(
            {
                "current_goal": context.current_goal,
                "user_message": context.user_message,
                "steps_taken": context.steps_taken,
                "max_steps": context.max_steps,
                "recent_steps": context.recent_steps,
                "observation": observation,
            },
            ensure_ascii=True,
        )


_ACTION_SYSTEM_PROMPT = (
    "You are a browser automation agent. Select the next single action to take. "
    "Use only the provided observation and goal. Do not invent selectors or URLs. "
    "Output only JSON with one action. Allowed actions:\n"
    "- observe\n"
    "- click (requires eid)\n"
    "- type (requires eid and text; only use editable roles like textbox/searchbox/combobox/textarea)\n"
    "- scroll (direction: up/down, amount integer)\n"
    "- wait (ms integer)\n"
    "- screenshot\n"
    "- need_user (reason: ask human to do login/captcha/2FA/payment)\n"
    "- stop (final_response)\n"
    "If a human login/captcha/2FA/payment is required, choose need_user.\n"
    "Never type into buttons/links or disabled elements.\n"
    "JSON format examples:\n"
    "{\"action\":\"click\",\"eid\":\"abc\"}\n"
    "{\"action\":\"type\",\"eid\":\"abc\",\"text\":\"hello\"}\n"
    "{\"action\":\"scroll\",\"direction\":\"down\",\"amount\":600}\n"
    "{\"action\":\"wait\",\"ms\":1000}\n"
    "{\"action\":\"need_user\",\"reason\":\"Please log in.\"}\n"
    "{\"action\":\"stop\",\"final_response\":\"Done.\"}\n"
)


def _parse_action_response(raw: str) -> Action:
    payload = _extract_json(raw)
    if not isinstance(payload, dict):
        return Action(kind="stop", final_response="Failed to parse model response.")
    kind = str(payload.get("action") or payload.get("kind") or "").strip().lower()
    if not kind:
        return Action(kind="stop", final_response="Model response missing action.")
    if kind == "observe":
        return Action(kind="observe")
    if kind == "click":
        eid = _as_str(payload.get("eid"))
        if not eid:
            return Action(kind="stop", final_response="Model response missing eid for click.")
        return Action(kind="click", eid=eid)
    if kind == "type":
        eid = _as_str(payload.get("eid"))
        text = _as_str(payload.get("text") or payload.get("value") or payload.get("input"))
        if not eid or text is None:
            return Action(kind="stop", final_response="Model response missing eid/text for type.")
        return Action(kind="type", eid=eid, value=text)
    if kind == "scroll":
        direction = _as_str(payload.get("direction") or "down") or "down"
        amount = _as_int(payload.get("amount"), 600)
        return Action(kind="scroll", args={"direction": direction, "amount": amount})
    if kind == "wait":
        timeout_ms = _as_int(payload.get("ms"), 1000)
        return Action(kind="wait", args={"timeout_ms": timeout_ms})
    if kind == "screenshot":
        return Action(kind="screenshot")
    if kind == "need_user":
        reason = _as_str(payload.get("reason") or payload.get("message")) or "Need user assistance."
        return Action(kind="need_user", reason=reason)
    if kind == "stop":
        final_response = _as_str(payload.get("final_response") or payload.get("response") or payload.get("text"))
        return Action(kind="stop", final_response=final_response or "Stopping.")
    return Action(kind="stop", final_response=f"Unknown action '{kind}'.")


def _extract_json(raw: str) -> dict | None:
    text = raw.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _as_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
