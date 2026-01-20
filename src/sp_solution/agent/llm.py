from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, model_validator

from ..config import Settings
from ..models import Action, Observation


_ESCAPED_UNICODE_RE = re.compile(r"(\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})")
_WHITESPACE_RE = re.compile(r"\s+")
_ZERO_WIDTH_CODEPOINTS = {
    0x200B,  # ZERO WIDTH SPACE
    0x200C,  # ZERO WIDTH NON-JOINER
    0x200D,  # ZERO WIDTH JOINER
    0x2060,  # WORD JOINER
    0xFEFF,  # BOM
    0x00AD,  # SOFT HYPHEN
}
_ZERO_WIDTH_TRANSLATION = {codepoint: None for codepoint in _ZERO_WIDTH_CODEPOINTS}
_EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"  # flags
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # geometric shapes extended
    "\U0001F800-\U0001F8FF"  # supplemental arrows
    "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
    "\U0001FA70-\U0001FAFF"  # symbols & pictographs extended-A
    "\U00002700-\U000027BF"  # dingbats
    "\U00002600-\U000026FF"  # misc symbols
    "\U00002300-\U000023FF"  # misc technical
    "\uFE0E\uFE0F"  # variation selectors
    "]"
)


def _decode_unicode_escapes(text: str) -> str:
    if not _ESCAPED_UNICODE_RE.search(text):
        return text
    length = len(text)
    i = 0
    out: list[str] = []
    while i < length:
        if text[i] == "\\" and i + 1 < length and text[i + 1] in {"u", "U"}:
            if text[i + 1] == "u" and i + 6 <= length:
                hex_part = text[i + 2 : i + 6]
                if re.fullmatch(r"[0-9a-fA-F]{4}", hex_part):
                    codepoint = int(hex_part, 16)
                    if 0xD800 <= codepoint <= 0xDBFF and i + 12 <= length:
                        if text[i + 6 : i + 8] == "\\u":
                            low_part = text[i + 8 : i + 12]
                            if re.fullmatch(r"[0-9a-fA-F]{4}", low_part):
                                low = int(low_part, 16)
                                if 0xDC00 <= low <= 0xDFFF:
                                    combined = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00)
                                    out.append(chr(combined))
                                    i += 12
                                    continue
                    out.append(chr(codepoint))
                    i += 6
                    continue
            if text[i + 1] == "U" and i + 10 <= length:
                hex_part = text[i + 2 : i + 10]
                if re.fullmatch(r"[0-9a-fA-F]{8}", hex_part):
                    codepoint = int(hex_part, 16)
                    out.append(chr(codepoint))
                    i += 10
                    continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if _ESCAPED_UNICODE_RE.search(text):
        text = _decode_unicode_escapes(text)
    text = text.translate(_ZERO_WIDTH_TRANSLATION)
    text = _EMOJI_RE.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text or None


def _is_punct_or_space(text: str) -> bool:
    return not any(char.isalnum() for char in text)


def _clean_text_blocks(values: list[Any]) -> list[str]:
    cleaned: list[str] = []
    last_value: str | None = None
    for value in values:
        text = _clean_text(value)
        if not text:
            continue
        if _is_punct_or_space(text):
            continue
        if last_value == text:
            continue
        cleaned.append(text)
        last_value = text
    return cleaned


def _clean_observation_for_llm(observation: Observation) -> dict[str, Any]:
    payload = observation.model_dump()
    page = payload.get("page") or {}
    if isinstance(page, dict):
        page["title"] = _clean_text(page.get("title"))
        page["url"] = _clean_text(page.get("url"))
        payload["page"] = page

    interactive = payload.get("interactive") or []
    if isinstance(interactive, list):
        cleaned_interactive: list[dict[str, Any]] = []
        for element in interactive:
            if not isinstance(element, dict):
                continue
            entry = dict(element)
            for key in ("name", "value", "placeholder", "role"):
                entry[key] = _clean_text(entry.get(key))
            cleaned_interactive.append(entry)
        payload["interactive"] = cleaned_interactive

    payload["text_blocks"] = _clean_text_blocks(payload.get("text_blocks") or [])
    payload["overlays"] = _clean_text_blocks(payload.get("overlays") or [])

    overlay_actions = payload.get("overlay_actions") or []
    if isinstance(overlay_actions, list):
        cleaned_actions: list[dict[str, Any]] = []
        for action in overlay_actions:
            if not isinstance(action, dict):
                continue
            text = _clean_text(action.get("text"))
            if not text:
                continue
            entry = dict(action)
            entry["text"] = text
            cleaned_actions.append(entry)
        payload["overlay_actions"] = cleaned_actions

    return payload


def _screenshot_to_image_url(screenshot: dict[str, str] | None) -> str | None:
    if not screenshot:
        return None
    url = screenshot.get("url")
    if isinstance(url, str) and url:
        return url
    data = screenshot.get("data")
    if isinstance(data, str) and data:
        mime_type = screenshot.get("mime_type") or "image/png"
        return f"data:{mime_type};base64,{data}"
    return None


def _openrouter_extra_body(settings: Settings) -> Dict[str, Any] | None:
    provider = settings.openrouter_provider
    if not provider:
        return None
    providers = [item.strip() for item in provider.split(",") if item.strip()]
    if not providers:
        return None
    return {"provider": {"order": providers}}


def _build_chat_client(settings: Settings) -> ChatOpenAI | None:
    if not settings.openrouter_api_key:
        return None
    extra_body = _openrouter_extra_body(settings)
    return ChatOpenAI(
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        extra_body=extra_body,
    )


def _messages_to_history(messages: list[BaseMessage]) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for msg in messages[-20:]:
        content = _clean_text(msg.content) if isinstance(msg.content, str) else str(msg.content)
        if not content:
            continue
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            role = "system"
        history.append({"role": role, "content": content})
    return history


class PlannerAction(BaseModel):
    action: Literal[
        "observe",
        "click",
        "type",
        "scroll",
        "wait",
        "screenshot",
        "need_user",
        "stop",
    ]
    eid: str | None = None
    text: str | None = None
    direction: Literal["up", "down"] | None = None
    amount: int | None = Field(default=None, ge=0)
    ms: int | None = Field(default=None, ge=0)
    reason: str | None = None
    final_response: str | None = None


class FactItem(BaseModel):
    fact: str

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"fact": value}
        return value


class PlanOutput(BaseModel):
    plan: list[str] = Field(min_length=1, max_length=6)
    facts: list[FactItem] | None = None


class ProgressState(BaseModel):
    current_index: int = 0
    done: list[str] = Field(default_factory=list)
    blocked: list[str] = Field(default_factory=list)
    note: str | None = None


class WorkingStateUpdate(BaseModel):
    progress: ProgressState | None = None
    facts: list[FactItem] | None = None
    needs_replan: bool | None = None


@dataclass
class ActionContext:
    user_message: str | None
    current_goal: str | None
    plan: list[str]
    progress: dict[str, Any] | None
    facts: list[dict[str, str]]
    observation: Observation | None
    steps_taken: int
    max_steps: int
    recent_steps: list[str]
    messages: list[BaseMessage]
    screenshot: dict[str, str] | None


@dataclass
class PlanContext:
    current_goal: str
    facts: list[dict[str, str]]
    messages: list[BaseMessage]


@dataclass
class StateUpdateContext:
    current_goal: str | None
    plan: list[str]
    progress: dict[str, Any] | None
    facts: list[dict[str, str]]
    last_action: dict[str, Any] | None
    last_tool_result: dict[str, Any] | None
    observation: Observation | None
    steps_taken: int


class LLMActionPlanner:
    def __init__(self, settings: Settings) -> None:
        self._client: ChatOpenAI | None = _build_chat_client(settings)
        self._structured = self._client.with_structured_output(PlannerAction) if self._client else None
        self._logger = logging.getLogger(__name__)

    def available(self) -> bool:
        return self._client is not None

    async def next_action(self, context: ActionContext) -> Action:
        if not self._structured:
            raise RuntimeError("OpenRouter client is not configured. Set OPENROUTER_API_KEY.")
        prompt = self._build_prompt(context)
        self._logger.info("LLM prompt:\n%s", prompt)
        image_url = _screenshot_to_image_url(context.screenshot)
        if image_url:
            human_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        else:
            human_content = prompt
        messages = [
            SystemMessage(content=_ACTION_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]
        try:
            response = await self._structured.ainvoke(messages)
        except Exception as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
        if isinstance(response, dict):
            parsed = PlannerAction.model_validate(response)
        else:
            parsed = response
        self._logger.info("LLM structured response: %s", parsed)
        action = _action_from_plan(parsed)
        self._logger.info("LLM action: %s", action.model_dump())
        return action

    def _build_prompt(self, context: ActionContext) -> str:
        observation = _clean_observation_for_llm(context.observation) if context.observation else None
        screenshot_meta = None
        if context.screenshot:
            screenshot_meta = {
                "available": True,
                "mime_type": context.screenshot.get("mime_type"),
                "has_url": "url" in context.screenshot,
            }
        return json.dumps(
            {
                "user_message": context.user_message,
                "current_goal": context.current_goal,
                "plan": context.plan,
                "progress": context.progress,
                "facts": context.facts,
                "steps_taken": context.steps_taken,
                "max_steps": context.max_steps,
                "recent_steps": context.recent_steps,
                "observation": observation,
                "chat_history": _messages_to_history(context.messages),
                "screenshot": screenshot_meta,
            },
            ensure_ascii=False,
        )


class LLMPlanBuilder:
    def __init__(self, settings: Settings) -> None:
        self._client: ChatOpenAI | None = _build_chat_client(settings)
        self._structured = self._client.with_structured_output(PlanOutput) if self._client else None
        self._logger = logging.getLogger(__name__)

    def available(self) -> bool:
        return self._client is not None

    async def build_plan(self, context: PlanContext) -> PlanOutput:
        if not self._structured:
            raise RuntimeError("OpenRouter client is not configured. Set OPENROUTER_API_KEY.")
        prompt = json.dumps(
            {
                "current_goal": context.current_goal,
                "facts": context.facts,
                "chat_history": _messages_to_history(context.messages),
            },
            ensure_ascii=False,
        )
        messages = [
            SystemMessage(content=_PLAN_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        try:
            response = await self._structured.ainvoke(messages)
        except Exception as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
        if isinstance(response, dict):
            parsed = PlanOutput.model_validate(response)
        else:
            parsed = response
        self._logger.info("LLM plan response: %s", parsed)
        return parsed


class LLMStateUpdater:
    def __init__(self, settings: Settings) -> None:
        self._client: ChatOpenAI | None = _build_chat_client(settings)
        self._structured = self._client.with_structured_output(WorkingStateUpdate) if self._client else None
        self._logger = logging.getLogger(__name__)

    def available(self) -> bool:
        return self._client is not None

    async def update_state(self, context: StateUpdateContext) -> WorkingStateUpdate:
        if not self._structured:
            raise RuntimeError("OpenRouter client is not configured. Set OPENROUTER_API_KEY.")
        observation = _clean_observation_for_llm(context.observation) if context.observation else None
        payload = {
            "current_goal": context.current_goal,
            "plan": context.plan,
            "progress": context.progress,
            "facts": context.facts,
            "last_action": context.last_action,
            "last_tool_result": context.last_tool_result,
            "observation": observation,
            "steps_taken": context.steps_taken,
        }
        messages = [
            SystemMessage(content=_STATE_UPDATE_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]
        try:
            response = await self._structured.ainvoke(messages)
        except Exception as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
        if isinstance(response, dict):
            parsed = WorkingStateUpdate.model_validate(response)
        else:
            parsed = response
        self._logger.info("LLM state update: %s", parsed)
        return parsed


_ACTION_SYSTEM_PROMPT = (
    "You are a browser automation agent. Select the next single action to take. "
    "Use only the provided observation (including any screenshot-derived cues), current_goal, plan, and progress. "
    "Always ground decisions in what is visible, and pick only the minimal relevant selectors. "
    "Do not invent selectors or URLs. "
    "Allowed actions: observe, click, type, scroll, wait, screenshot, need_user, stop. "
    "If a human login/captcha/2FA/payment is required, choose need_user. "
    "Never type into buttons/links or disabled elements. "
    "When typing, provide the text in the 'text' field. "
    "When scrolling, provide 'direction' (up/down) and 'amount'. "
    "When waiting, provide 'ms'. "
    "When stopping, provide 'final_response'."
)


_PLAN_SYSTEM_PROMPT = (
    "You are a planning assistant for a browser automation agent. "
    "Produce a short plan of 3-6 steps to achieve the current goal. "
    "The final step must be answering the user with results. "
    "Do not include raw DOM, selectors, or element ids. "
    "Keep steps concise and action-oriented."
)


_STATE_UPDATE_PROMPT = (
    "You update the agent working state after one action. "
    "Follow the framework: "
    "1) keep a plan of 3-6 steps, "
    "2) execute the plan step by step, "
    "3) while executing you may advance the current step, analyze the gathered info, store facts, or revise the plan, "
    "4) the final plan step must always be answering the user. "
    "Use the current goal, plan, progress, facts, last action/result, and observation. "
    "Update progress (current_index, done, blocked, note) conservatively. "
    "Store facts as dense summaries of information observed on pages (concise, factual, and page-derived). "
    "Facts must be a list of dicts with key 'fact' and a short value. "
    "DO NOT STORE INTERNAL REASONING, ACTION LOGS, OR PROCESS NOTES IN FACTS. "
    "Never include raw page text, DOM, or element ids in facts/progress. "
    "Set needs_replan true if the current plan is no longer valid."
)


def _action_from_plan(plan: PlannerAction) -> Action:
    kind = plan.action
    if kind == "observe":
        return Action(kind="observe")
    if kind == "click":
        if not plan.eid:
            return Action(kind="stop", final_response="Model response missing eid for click.")
        return Action(kind="click", eid=plan.eid)
    if kind == "type":
        if not plan.eid or plan.text is None:
            return Action(kind="stop", final_response="Model response missing eid/text for type.")
        return Action(kind="type", eid=plan.eid, value=plan.text)
    if kind == "scroll":
        direction = plan.direction or "down"
        amount = plan.amount if plan.amount is not None else 600
        return Action(kind="scroll", args={"direction": direction, "amount": amount})
    if kind == "wait":
        timeout_ms = plan.ms if plan.ms is not None else 1000
        return Action(kind="wait", args={"timeout_ms": timeout_ms})
    if kind == "screenshot":
        return Action(kind="screenshot")
    if kind == "need_user":
        reason = plan.reason or "Need user assistance."
        return Action(kind="need_user", reason=reason)
    if kind == "stop":
        final_response = plan.final_response or "Stopping."
        return Action(kind="stop", final_response=final_response)
    return Action(kind="stop", final_response=f"Unknown action '{kind}'.")
