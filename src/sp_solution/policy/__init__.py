from __future__ import annotations

from ..models import Action, PolicyDecision


SAFE_ACTIONS = {"observe", "describe", "scroll", "wait", "screenshot"}


class PolicyGate:
    def assess(self, action: Action) -> PolicyDecision:
        if action.kind in SAFE_ACTIONS:
            return PolicyDecision(allow=True, requires_confirmation=False)
        return PolicyDecision(
            allow=False,
            requires_confirmation=True,
            reason=f"Action '{action.kind}' requires confirmation",
        )
