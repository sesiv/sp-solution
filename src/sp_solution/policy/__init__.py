from __future__ import annotations

import re

from ..models import Action, Observation, PolicyDecision


SAFE_ACTIONS = {"observe", "describe", "launch", "scroll", "wait", "screenshot", "stop", "need_user"}
CLICK_ROLES = {"button", "link", "menuitem", "tab", "option", "checkbox", "radio", "switch", "submit"}
DESTRUCTIVE_KEYWORDS = [
    # English
    "pay",
    "payment",
    "checkout",
    "buy",
    "purchase",
    "order",
    "confirm",
    "submit",
    "send",
    "delete",
    "remove",
    "unsubscribe",
    "subscribe",
    "sign up",
    "register",
    "book",
    "reserve",
    "donate",
    "transfer",
    "upgrade",
    "renew",
    "cancel",
    # Russian stems
    "\u043e\u043f\u043b\u0430\u0442",
    "\u043f\u043b\u0430\u0442\u0435\u0436",
    "\u043a\u0443\u043f\u0438\u0442\u044c",
    "\u0437\u0430\u043a\u0430\u0437",
    "\u043e\u0444\u043e\u0440\u043c",
    "\u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0434",
    "\u0443\u0434\u0430\u043b",
    "\u043e\u0442\u043f\u0440\u0430\u0432",
    "\u043f\u043e\u0434\u043f\u0438\u0441",
    "\u043f\u0435\u0440\u0435\u0432\u043e\u0434",
    "\u043f\u0435\u0440\u0435\u0432\u0435\u0441\u0442",
    "\u0431\u0440\u043e\u043d\u0438",
    "\u0441\u043f\u0438\u0441\u0430\u043d",
]


class PolicyGate:
    def assess(self, action: Action, observation: Observation | None = None) -> PolicyDecision:
        if action.kind in SAFE_ACTIONS:
            return PolicyDecision(allow=True, requires_confirmation=False)
        if action.kind == "click":
            if self._is_destructive_click(action, observation):
                return PolicyDecision(
                    allow=False,
                    requires_confirmation=True,
                    reason="Click looks potentially destructive. Confirmation required.",
                )
            return PolicyDecision(allow=True, requires_confirmation=False)
        if action.kind == "type":
            return PolicyDecision(allow=True, requires_confirmation=False)
        return PolicyDecision(
            allow=False,
            requires_confirmation=True,
            reason=f"Action '{action.kind}' requires confirmation",
        )

    def _is_destructive_click(self, action: Action, observation: Observation | None) -> bool:
        element_text = ""
        role = None
        if observation and action.eid:
            for element in observation.interactive:
                if element.eid == action.eid:
                    role = element.role
                    element_text = " ".join(
                        part for part in [element.name, element.value, element.placeholder] if part
                    )
                    break
        if role and role.lower() not in CLICK_ROLES:
            return False
        combined = self._normalize_text(element_text)
        if combined and self._contains_destructive_keyword(combined):
            return True
        if not combined and observation:
            for block in observation.text_blocks[:10]:
                if self._contains_destructive_keyword(self._normalize_text(block)):
                    return True
        return False

    def _contains_destructive_keyword(self, text: str) -> bool:
        return any(keyword in text for keyword in DESTRUCTIVE_KEYWORDS)

    def _normalize_text(self, text: str) -> str:
        lowered = text.lower()
        return re.sub(r"\s+", " ", lowered).strip()
