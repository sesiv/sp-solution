from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

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


def _openrouter_extra_body(settings: Settings) -> Dict[str, Any] | None:
    provider = settings.openrouter_provider
    if not provider:
        return None
    providers = [item.strip() for item in provider.split(",") if item.strip()]
    if not providers:
        return None
    return {"provider": {"order": providers}}


@dataclass
class ActionContext:
    user_message: str | None
    observation: Observation | None
    steps_taken: int
    max_steps: int
    recent_steps: list[str]
    chat_history: list[dict[str, str]]


class LLMActionPlanner:
    def __init__(self, settings: Settings) -> None:
        self._client: Optional[ChatOpenAI] = None
        if settings.openrouter_api_key:
            extra_body = _openrouter_extra_body(settings)
            self._client = ChatOpenAI(
                model=settings.openrouter_model,
                api_key=settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=0.2,
                extra_body=extra_body,
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
        observation = (
            _clean_observation_for_llm(context.observation) if context.observation else None
        )
        return json.dumps(
            {
                "user_message": context.user_message,
                "steps_taken": context.steps_taken,
                "max_steps": context.max_steps,
                "recent_steps": context.recent_steps,
                "observation": observation,
                "chat_history": context.chat_history,
            },
            ensure_ascii=False,
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
