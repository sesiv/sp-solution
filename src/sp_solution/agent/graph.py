from __future__ import annotations

import logging
import shlex
from typing import Any, Dict, Optional

from ..browser.mcp import MCPBrowserClient
from ..config import Settings
from ..models import Action, Event, Observation, ServerMessage, InteractiveElement
from ..observation import ObservationBuilder
from ..policy import PolicyGate
from ..session import RunState, Session
from .llm import ActionContext, LLMActionPlanner


MAX_TOOL_STEPS = 10
EDITABLE_ROLES = {"textbox", "searchbox", "combobox", "textarea", "input"}


class AgentRunner:
    def __init__(self, session: Session, settings: Settings) -> None:
        self._session = session
        self._settings = settings
        self._browser: MCPBrowserClient | None = None
        self._observer = ObservationBuilder()
        self._policy = PolicyGate()
        self._planner = LLMActionPlanner(settings)
        self._last_observation: Observation | None = None
        self._logger = logging.getLogger(__name__)

    async def handle_user_message(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if self._session.pending_kind == "confirm":
            await self._handle_text_confirmation(text)
            return
        if self._session.pending_kind == "manual":
            await self._handle_manual_done(text)
            return
        if text.startswith("/"):
            await self._handle_command(text)
            return
        if self._session.mode != "chat":
            await self._session.send_message(
                ServerMessage(type="agent_message", payload={"text": "Use /chat to enter chat mode."})
            )
            await self._set_status("waiting_user")
            return
        await self._handle_chat_message(text)

    async def handle_user_confirm(self, reference: str, confirmed: bool) -> None:
        if not self._session.pending_action_id or reference != self._session.pending_action_id:
            await self._session.send_message(
                ServerMessage(type="agent_message", payload={"text": "No pending action found."})
            )
            return
        pending = self._session.pending_action
        run_state = self._session.run_state
        self._session.pending_action = None
        self._session.pending_action_id = None
        self._session.pending_kind = "none"
        await self._session.publish_event(
            Event.create(
                "policy_result",
                self._session.session_id,
                {"reference": reference, "confirmed": confirmed},
            )
        )
        if not confirmed:
            self._session.run_state = None
            await self._session.send_message(
                ServerMessage(
                    type="agent_message",
                    payload={"text": "Action cancelled. Tell me what to do next."},
                )
            )
            await self._set_status("waiting_user")
            return
        if pending:
            try:
                await self._execute_action(pending, run_state=run_state)
            except Exception as exc:
                await self._session.send_message(
                    ServerMessage(type="agent_message", payload={"text": f"Tool failed: {exc}"})
                )
                self._session.run_state = None
                await self._set_status("waiting_user")
                return
            if run_state and pending.kind in {"click", "type", "scroll", "wait"}:
                run_state.needs_observe = True
        if self._session.mode == "chat" and run_state:
            await self._run_chat_loop(user_message=None, run_state=run_state)
            return
        await self._set_status("waiting_user")

    async def _handle_text_confirmation(self, text: str) -> None:
        if text not in {"/yes", "/no"}:
            await self._session.send_message(
                ServerMessage(type="agent_message", payload={"text": "Reply /yes or /no to confirm."})
            )
            await self._set_status("waiting_user")
            return
        confirmed = text == "/yes"
        reference = self._session.pending_action_id or ""
        await self.handle_user_confirm(reference, confirmed)

    async def _handle_manual_done(self, text: str) -> None:
        if text.startswith("/"):
            await self._handle_manual_command(text)
            return
        run_state = self._session.run_state
        self._session.pending_kind = "none"
        await self._session.send_message(
            ServerMessage(type="agent_message", payload={"text": "Thanks. Continuing from the new state."})
        )
        if run_state is None:
            run_state = RunState()
        run_state.needs_observe = True
        await self._run_chat_loop(user_message=None, run_state=run_state)

    async def _handle_manual_command(self, text: str) -> None:
        action = self._parse_action(text)
        if action and action.kind == "observe":
            observation = await self._observe(count_step=False, run_state=None, take_screenshot=False)
            await self._session.send_message(
                ServerMessage(type="agent_message", payload={"text": self._summary(observation)})
            )
            await self._set_status("waiting_user")
            return
        if action and action.kind == "screenshot":
            await self._call_tool("screenshot", {}, run_state=None, action=action)
            await self._set_status("waiting_user")
            return
        await self._session.send_message(
            ServerMessage(
                type="agent_message",
                payload={"text": "Finish the manual step, then send any message to continue."},
            )
        )
        await self._set_status("waiting_user")

    async def _handle_command(self, text: str) -> None:
        command = text.split()[0].lstrip("/").lower()
        if command == "chat":
            self._session.mode = "chat"
            await self._session.send_message(
                ServerMessage(type="agent_message", payload={"text": "Chat mode enabled. Tell me what to do."})
            )
            await self._set_status("waiting_user")
            return
        if command == "exit":
            self._session.mode = "cmd"
            await self._session.send_message(
                ServerMessage(type="agent_message", payload={"text": "Chat mode disabled."})
            )
            await self._set_status("waiting_user")
            return
        if self._session.mode == "chat":
            self._session.mode = "cmd"
        action = self._parse_action(text)
        if action:
            await self._handle_action(action)
            return
        await self._session.send_message(
            ServerMessage(type="agent_message", payload={"text": "Unknown command."})
        )
        await self._set_status("waiting_user")

    async def _handle_chat_message(self, text: str) -> None:
        await self._run_chat_loop(user_message=text, run_state=None)

    async def _run_chat_loop(self, user_message: Optional[str], run_state: RunState | None) -> None:
        if not self._planner.available():
            await self._session.send_message(
                ServerMessage(
                    type="agent_message",
                    payload={"text": "LLM is not configured. Set OPENROUTER_API_KEY to use chat mode."},
                )
            )
            await self._set_status("waiting_user")
            return
        if run_state is None:
            run_state = RunState()
        if user_message is not None:
            run_state.last_user_message = user_message
        self._session.run_state = run_state
        await self._set_status("running")

        while run_state.tool_steps < MAX_TOOL_STEPS:
            if run_state.needs_observe or not self._last_observation:
                await self._observe(count_step=True, run_state=run_state, take_screenshot=True)
                run_state.needs_observe = False
                if run_state.tool_steps >= MAX_TOOL_STEPS:
                    break

            context = ActionContext(
                user_message=run_state.last_user_message,
                observation=self._last_observation,
                steps_taken=run_state.tool_steps,
                max_steps=MAX_TOOL_STEPS,
                recent_steps=run_state.steps[-5:],
            )
            action = await self._planner.next_action(context)
            if action.kind == "stop":
                final_response = action.final_response or "Stopping."
                await self._finalize(final_response)
                self._session.run_state = None
                await self._set_status("waiting_user")
                return
            if action.kind == "need_user":
                reason = action.reason or "Manual action required."
                self._session.pending_kind = "manual"
                await self._session.send_message(
                    ServerMessage(
                        type="agent_message",
                        payload={"text": f"{reason} Please handle it and reply when done."},
                    )
                )
                await self._set_status("waiting_user")
                return
            validation_error = self._validate_action(action)
            if validation_error:
                run_state.invalid_actions += 1
                self._record_validation_failure(run_state, action, validation_error)
                if run_state.invalid_actions >= 3:
                    await self._session.send_message(
                        ServerMessage(
                            type="agent_message",
                            payload={
                                "text": (
                                    f"Blocked action: {validation_error}. "
                                    "Please clarify the goal or specify the element."
                                )
                            },
                        )
                    )
                    await self._set_status("waiting_user")
                    self._session.run_state = None
                    return
                continue

            decision = self._policy.assess(action, self._last_observation)
            if not decision.allow and not decision.requires_confirmation:
                await self._session.send_message(
                    ServerMessage(
                        type="agent_message",
                        payload={"text": decision.reason or "Action denied by policy."},
                    )
                )
                await self._set_status("waiting_user")
                self._session.run_state = None
                return
            if decision.requires_confirmation:
                self._session.pending_action = action
                self._session.pending_action_id = action.id
                self._session.pending_kind = "confirm"
                await self._session.publish_event(
                    Event.create(
                        "policy_request",
                        self._session.session_id,
                        {"reference": action.id, "reason": decision.reason, "action": action.model_dump()},
                    )
                )
                await self._session.send_message(
                    ServerMessage(
                        type="agent_question",
                        payload={
                            "text": f"Confirm action {action.kind} (reference {action.id}). Reply /yes or /no.",
                            "reference": action.id,
                        },
                    )
                )
                await self._set_status("waiting_user")
                return

            try:
                await self._execute_action(action, run_state=run_state)
            except Exception as exc:
                await self._session.send_message(
                    ServerMessage(type="agent_message", payload={"text": f"Tool failed: {exc}"})
                )
                self._session.run_state = None
                await self._set_status("waiting_user")
                return

            if action.kind in {"click", "type", "scroll", "wait"}:
                run_state.needs_observe = True

        summary = self._build_limit_summary(run_state)
        await self._finalize(summary)
        await self._session.send_message(
            ServerMessage(
                type="agent_question",
                payload={"text": "Continue?", "reference": None},
            )
        )
        await self._set_status("waiting_user")
        self._session.run_state = None

    async def _handle_action(self, action: Action) -> None:
        if action.kind == "observe":
            observation = await self._observe(count_step=False, run_state=None, take_screenshot=False)
            await self._session.send_message(
                ServerMessage(type="agent_message", payload={"text": self._summary(observation)})
            )
            await self._set_status("waiting_user")
            return
        decision = self._policy.assess(action, self._last_observation)
        if not decision.allow and not decision.requires_confirmation:
            await self._session.send_message(
                ServerMessage(
                    type="agent_message",
                    payload={"text": decision.reason or "Action denied by policy."},
                )
            )
            return
        if decision.requires_confirmation:
            self._session.pending_action = action
            self._session.pending_action_id = action.id
            self._session.pending_kind = "confirm"
            await self._session.publish_event(
                Event.create(
                    "policy_request",
                    self._session.session_id,
                    {"reference": action.id, "reason": decision.reason, "action": action.model_dump()},
                )
            )
            await self._session.send_message(
                ServerMessage(
                    type="agent_question",
                    payload={
                        "text": f"Confirm action {action.kind} (reference {action.id}). Reply /yes or /no.",
                        "reference": action.id,
                    },
                )
            )
            await self._set_status("waiting_user")
            return
        await self._execute_action(action, run_state=None)
        await self._set_status("waiting_user")

    async def _execute_action(self, action: Action, run_state: RunState | None) -> None:
        if action.kind == "observe":
            await self._observe(
                count_step=run_state is not None,
                run_state=run_state,
                take_screenshot=run_state is not None,
            )
            return
        if action.kind == "launch":
            await self._call_tool("launch", {}, run_state=run_state, action=action)
            return
        if action.kind == "click":
            await self._call_tool(
                "click",
                {"eid": action.eid, "element": self._describe_element(action.eid)},
                run_state=run_state,
                action=action,
            )
        elif action.kind == "type":
            await self._call_tool(
                "type",
                {
                    "eid": action.eid,
                    "value": action.value,
                    "element": self._describe_element(action.eid),
                },
                run_state=run_state,
                action=action,
            )
        elif action.kind == "scroll":
            await self._call_tool("scroll", action.args, run_state=run_state, action=action)
        elif action.kind == "wait":
            await self._call_tool("wait", action.args, run_state=run_state, action=action)
        elif action.kind == "screenshot":
            await self._call_tool("screenshot", {}, run_state=run_state, action=action)

    async def _observe(
        self,
        count_step: bool,
        run_state: RunState | None,
        take_screenshot: bool,
    ) -> Observation:
        step_state = run_state if count_step else None
        if take_screenshot:
            await self._call_tool(
                "screenshot",
                {},
                run_state=step_state,
                action=Action(kind="screenshot"),
            )
        raw = await self._call_tool("observe", {}, run_state=step_state, action=None)
        self._logger.info("Observe raw response: %s", raw)
        observation = self._observer.build(raw)
        self._last_observation = observation
        await self._session.publish_event(
            Event.create(
                "observation",
                self._session.session_id,
                {
                    "summary": self._summary(observation),
                    "observation": observation.model_dump(),
                },
            )
        )
        return observation

    async def _call_tool(
        self,
        name: str,
        args: Dict[str, Any],
        run_state: RunState | None,
        action: Action | None,
    ) -> Dict[str, Any]:
        await self._session.publish_event(
            Event.create(
                "tool_call",
                self._session.session_id,
                {"name": name, "args": args},
            )
        )
        if run_state is not None:
            run_state.tool_steps += 1
        try:
            browser = self._get_browser()
            if name == "observe":
                result = await browser.observe(**args)
            elif name == "launch":
                result = await browser.launch()
            elif name == "click":
                result = await browser.click(args["eid"], args.get("element"))
            elif name == "type":
                result = await browser.type(args["eid"], args["value"], args.get("element"))
            elif name == "scroll":
                result = await browser.scroll(**args)
            elif name == "wait":
                result = await browser.wait(**args)
            elif name == "screenshot":
                result = await browser.screenshot()
            else:
                raise ValueError(f"Unknown tool: {name}")
        except Exception as exc:
            if run_state is not None:
                self._record_failure(run_state, action, str(exc))
            await self._session.publish_event(
                Event.create(
                    "error",
                    self._session.session_id,
                    {"stage": name, "message": str(exc)},
                )
            )
            raise
        summary = self._short_result(result)
        await self._session.publish_event(
            Event.create(
                "tool_result",
                self._session.session_id,
                {"name": name, "success": True, "result": summary},
            )
        )
        if run_state is not None:
            self._record_step(run_state, action, summary)
        return result

    def _record_step(self, run_state: RunState, action: Action | None, summary: Dict[str, Any]) -> None:
        label = self._format_action_label(action)
        run_state.steps.append(f"{label} -> ok")
        self._logger.info("Step ok: %s", {"action": label, "summary": summary})

    def _record_failure(self, run_state: RunState, action: Action | None, error: str) -> None:
        label = self._format_action_label(action)
        run_state.failures.append(f"{label} -> {error}")
        self._logger.info("Step failed: %s", {"action": label, "error": error})

    def _record_validation_failure(self, run_state: RunState, action: Action, error: str) -> None:
        label = self._format_action_label(action)
        run_state.failures.append(f"{label} -> {error}")
        run_state.steps.append(f"{label} -> skipped ({error})")
        self._logger.info("Validation blocked action: %s", {"action": label, "error": error})

    def _format_action_label(self, action: Action | None) -> str:
        if action is None:
            return "observe"
        if action.kind == "click":
            return f"click {action.eid}"
        if action.kind == "type":
            return f"type {action.eid}"
        if action.kind == "scroll":
            direction = action.args.get("direction", "down")
            amount = action.args.get("amount", 600)
            return f"scroll {direction} {amount}"
        if action.kind == "wait":
            timeout_ms = action.args.get("timeout_ms", 1000)
            return f"wait {timeout_ms}ms"
        if action.kind == "screenshot":
            return "screenshot"
        if action.kind == "launch":
            return "launch"
        return action.kind

    def _validate_action(self, action: Action) -> str | None:
        if action.kind not in {"type", "click"}:
            return None
        element = self._get_element(action.eid)
        if not element:
            return "Target element not found in observation."
        if element.disabled:
            return "Target element is disabled."
        if action.kind == "type":
            role = (element.role or "").lower()
            if role and role not in EDITABLE_ROLES:
                return f"Target element role '{role}' is not editable."
        return None

    async def _finalize(self, message: str) -> None:
        await self._session.publish_event(
            Event.create("final", self._session.session_id, {"result": message})
        )
        await self._session.send_message(ServerMessage(type="agent_message", payload={"text": message}))

    async def _set_status(self, status: str) -> None:
        self._session.status = status
        await self._session.send_message(ServerMessage(type="status", payload={"value": status}))

    def _parse_action(self, text: str) -> Optional[Action]:
        if not text.startswith("/"):
            return None
        tokens = shlex.split(text)
        if not tokens:
            return None
        command = tokens[0].lstrip("/").lower()
        if command == "observe":
            return Action(kind="observe")
        if command == "launch":
            return Action(kind="launch")
        if command == "click" and len(tokens) >= 2:
            return Action(kind="click", eid=tokens[1])
        if command == "type" and len(tokens) >= 3:
            return Action(kind="type", eid=tokens[1], value=" ".join(tokens[2:]))
        if command == "scroll":
            direction = tokens[1] if len(tokens) >= 2 else "down"
            amount = int(tokens[2]) if len(tokens) >= 3 else 600
            return Action(kind="scroll", args={"direction": direction, "amount": amount})
        if command == "wait":
            timeout_ms = int(tokens[1]) if len(tokens) >= 2 else 1000
            return Action(kind="wait", args={"timeout_ms": timeout_ms})
        if command == "screenshot":
            return Action(kind="screenshot")
        return None

    def _describe_element(self, eid: str | None) -> str | None:
        if not eid or not self._last_observation:
            return None
        for element in self._last_observation.interactive:
            if element.eid == eid:
                parts = [element.role, element.name, element.value]
                return " ".join([part for part in parts if part])
        return None

    def _get_element(self, eid: str | None) -> InteractiveElement | None:
        if not eid or not self._last_observation:
            return None
        for element in self._last_observation.interactive:
            if element.eid == eid:
                return element
        return None

    def _get_browser(self) -> MCPBrowserClient:
        if not self._browser:
            self._browser = MCPBrowserClient(self._session.session_id, self._settings)
        return self._browser

    def _build_limit_summary(self, run_state: RunState) -> str:
        done = "; ".join(run_state.steps[:6])
        if len(run_state.steps) > 6:
            done = f"{done}; ..."
        if not done:
            done = "No tool actions executed."
        if run_state.failures:
            failures = "; ".join(run_state.failures[:3])
            if len(run_state.failures) > 3:
                failures = f"{failures}; ..."
            failure_line = f"Failures: {failures}."
        else:
            failure_line = "Failures: none."
        next_step = "Next step: continue with the current task."
        return f"Steps: {done}. {failure_line} {next_step}"

    @staticmethod
    def _summary(observation: Observation) -> str:
        return (
            f"{observation.page.title} ({observation.page.url}) | "
            f"interactive={len(observation.interactive)} text_blocks={len(observation.text_blocks)} "
            f"overlays={len(observation.overlays)}"
        )

    @staticmethod
    def _short_result(result: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return {"result": str(result)}
        content = result.get("content")
        if isinstance(content, list):
            type_counts: Dict[str, int] = {}
            text_chars = 0
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type", "unknown"))
                type_counts[item_type] = type_counts.get(item_type, 0) + 1
                if item_type == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        text_chars += len(text)
            summary: Dict[str, Any] = {
                "content_items": len(content),
                "content_types": type_counts,
            }
            if text_chars:
                summary["text_chars"] = text_chars
            return summary
        keys = list(result.keys())[:5]
        return {key: AgentRunner._redact_log_value(result[key]) for key in keys}

    @staticmethod
    def _redact_log_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: AgentRunner._redact_log_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [AgentRunner._redact_log_value(item) for item in value]
        if isinstance(value, str) and len(value) > 500:
            return f"<omitted {len(value)} chars>"
        return value
