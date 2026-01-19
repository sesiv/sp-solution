from __future__ import annotations

import logging
import shlex
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # pragma: no cover - compatibility with older langgraph
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver

from ..browser.mcp import MCPBrowserClient
from ..config import Settings
from ..models import Action, Event, Observation, PolicyDecision, ServerMessage, InteractiveElement
from ..observation import ObservationBuilder
from ..policy import PolicyGate
from .llm import ActionContext, LLMActionPlanner


MAX_TOOL_STEPS = 10
EDITABLE_ROLES = {"textbox", "searchbox", "combobox", "textarea", "input"}

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    last_user_message: Optional[str]
    last_observation: Optional[Observation]
    last_screenshot: Optional[dict[str, str]]
    planned_action: Optional[Action]
    steps: List[str]
    failures: List[str]
    tool_steps: int
    invalid_actions: int
    needs_observe: bool
    final_response: Optional[str]
    policy_decision: Optional[PolicyDecision]
    confirm_result: Optional[bool]
    single_step: bool


class ToolError(RuntimeError):
    pass


class LangGraphAgent:
    def __init__(self, session: Any, settings: Settings) -> None:
        self._session = session
        self._settings = settings
        self._browser: MCPBrowserClient | None = None
        self._observer = ObservationBuilder()
        self._policy = PolicyGate()
        self._planner = LLMActionPlanner(settings)
        self._graph = self._build_graph()

    def planner_available(self) -> bool:
        return self._planner.available()

    async def ainvoke(self, payload: Any) -> Any:
        config = {"configurable": {"thread_id": self._session.session_id}}
        return await self._graph.ainvoke(payload, config=config)

    def _build_graph(self) -> Any:
        builder = StateGraph(AgentState)
        builder.add_node("ingest_user_input", self.ingest_user_input)
        builder.add_node("maybe_observe", self.maybe_observe)
        builder.add_node("plan_action", self.plan_action)
        builder.add_node("validate_action", self.validate_action)
        builder.add_node("policy_gate", self.policy_gate)
        builder.add_node("confirm_interrupt", self.confirm_interrupt)
        builder.add_node("manual_interrupt", self.manual_interrupt)
        builder.add_node("execute_action", self.execute_action)
        builder.add_node("finalize_done", self.finalize_done)
        builder.add_node("finalize_denied", self.finalize_denied)
        builder.add_node("finalize_cancelled", self.finalize_cancelled)
        builder.add_node("finalize_blocked", self.finalize_blocked)
        builder.add_node("finalize_limit", self.finalize_limit)

        builder.set_entry_point("ingest_user_input")
        builder.add_edge("ingest_user_input", "maybe_observe")
        builder.add_conditional_edges(
            "maybe_observe",
            self._route_after_observe,
            {
                "plan_action": "plan_action",
                "validate_action": "validate_action",
                "finalize_limit": "finalize_limit",
                "finalize_done": "finalize_done",
            },
        )
        builder.add_edge("plan_action", "validate_action")
        builder.add_conditional_edges(
            "validate_action",
            self._route_after_validate,
            {
                "policy_gate": "policy_gate",
                "manual_interrupt": "manual_interrupt",
                "finalize_blocked": "finalize_blocked",
            },
        )
        builder.add_conditional_edges(
            "policy_gate",
            self._route_after_policy,
            {
                "execute_action": "execute_action",
                "confirm_interrupt": "confirm_interrupt",
                "finalize_denied": "finalize_denied",
            },
        )
        builder.add_conditional_edges(
            "confirm_interrupt",
            self._route_after_confirm,
            {
                "execute_action": "execute_action",
                "finalize_cancelled": "finalize_cancelled",
            },
        )
        builder.add_edge("manual_interrupt", "maybe_observe")
        builder.add_conditional_edges(
            "execute_action",
            self._route_after_execute,
            {
                "maybe_observe": "maybe_observe",
                "finalize_done": "finalize_done",
                "finalize_limit": "finalize_limit",
            },
        )

        builder.add_edge("finalize_done", END)
        builder.add_edge("finalize_denied", END)
        builder.add_edge("finalize_cancelled", END)
        builder.add_edge("finalize_blocked", END)
        builder.add_edge("finalize_limit", END)

        return builder.compile(checkpointer=InMemorySaver())

    def _state_with_defaults(self, state: Dict[str, Any]) -> Dict[str, Any]:
        needs_observe = state.get("needs_observe")
        if needs_observe is None:
            needs_observe = True
        return {
            "messages": list(state.get("messages") or []),
            "last_user_message": state.get("last_user_message"),
            "last_observation": state.get("last_observation"),
            "last_screenshot": state.get("last_screenshot"),
            "planned_action": state.get("planned_action"),
            "steps": list(state.get("steps") or []),
            "failures": list(state.get("failures") or []),
            "tool_steps": int(state.get("tool_steps") or 0),
            "invalid_actions": int(state.get("invalid_actions") or 0),
            "needs_observe": bool(needs_observe),
            "final_response": state.get("final_response"),
            "policy_decision": state.get("policy_decision"),
            "confirm_result": state.get("confirm_result"),
            "single_step": bool(state.get("single_step", False)),
        }

    async def ingest_user_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        merged = self._state_with_defaults(state)
        text = state.get("user_message")
        if isinstance(text, str):
            cleaned = text.strip()
            if cleaned:
                merged["messages"].append(HumanMessage(content=cleaned))
                merged["last_user_message"] = cleaned
                merged["planned_action"] = None
                merged["final_response"] = None
                merged["tool_steps"] = 0
                merged["steps"] = []
                merged["failures"] = []
                merged["invalid_actions"] = 0
                merged["needs_observe"] = True
                merged["single_step"] = False
                if cleaned.startswith("/"):
                    action = self._parse_action(cleaned)
                    if action:
                        merged["planned_action"] = action
                        merged["needs_observe"] = action.kind in {"observe", "click", "type"}
                        merged["single_step"] = True
                    else:
                        merged["final_response"] = "Unknown command."
                        merged["needs_observe"] = False
                        merged["single_step"] = True
        return merged

    async def maybe_observe(self, state: Dict[str, Any]) -> Dict[str, Any]:
        merged = self._state_with_defaults(state)
        if not merged.get("single_step"):
            merged["planned_action"] = None
        if merged.get("final_response"):
            return merged
        action = merged.get("planned_action")
        if merged.get("single_step") and action and action.kind in {"screenshot", "scroll", "wait", "launch"}:
            merged["needs_observe"] = False
            return merged
        if merged.get("needs_observe") or merged.get("last_observation") is None:
            merged = await self._observe(merged)
            merged["needs_observe"] = False
        return merged

    async def plan_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        merged = self._state_with_defaults(state)
        if not self._planner.available():
            merged["final_response"] = "LLM is not configured. Set OPENROUTER_API_KEY to use chat mode."
            return merged
        context = ActionContext(
            user_message=merged.get("last_user_message"),
            observation=merged.get("last_observation"),
            steps_taken=merged.get("tool_steps", 0),
            max_steps=MAX_TOOL_STEPS,
            recent_steps=merged.get("steps", [])[-5:],
            messages=merged.get("messages", []),
            screenshot=merged.get("last_screenshot"),
        )
        action = await self._planner.next_action(context)
        merged["planned_action"] = action
        return merged

    async def validate_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        merged = self._state_with_defaults(state)
        action = merged.get("planned_action")
        if not action:
            merged["final_response"] = "No action planned."
            return merged
        error = self._validate_action(action, merged.get("last_observation"))
        if error:
            merged["invalid_actions"] = merged.get("invalid_actions", 0) + 1
            self._record_validation_failure(merged, action, error)
            if merged["invalid_actions"] >= 3:
                merged["final_response"] = (
                    f"Blocked action: {error}. Please clarify the goal or specify the element."
                )
        return merged

    async def policy_gate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        merged = self._state_with_defaults(state)
        action = merged.get("planned_action")
        if not action:
            merged["final_response"] = "No action planned."
            return merged
        decision = self._policy.assess(action, merged.get("last_observation"))
        merged["policy_decision"] = decision
        return merged

    async def confirm_interrupt(self, state: Dict[str, Any]) -> Dict[str, Any]:
        merged = self._state_with_defaults(state)
        action = merged.get("planned_action")
        decision = merged.get("policy_decision")
        if isinstance(decision, dict):
            decision = PolicyDecision.model_validate(decision)
            merged["policy_decision"] = decision
        if not action or not decision:
            merged["final_response"] = "No action pending confirmation."
            return merged
        reason_text = decision.reason or "Action requires confirmation."
        prompt = f"{reason_text} Confirm action {action.kind} (reference {action.id}). Reply /yes or /no."
        merged["messages"].append(AIMessage(content=prompt))
        await self._session.publish_event(
            Event.create(
                "policy_request",
                self._session.session_id,
                {"reference": action.id, "reason": decision.reason, "action": action.model_dump()},
            )
        )
        response = interrupt(
            {
                "kind": "confirm",
                "reference": action.id,
                "reason": reason_text,
                "action": action.model_dump(),
                "text": prompt,
            }
        )
        confirmed = False
        if isinstance(response, dict):
            reference = response.get("reference")
            confirmed = bool(response.get("confirmed")) and reference == action.id
        await self._session.publish_event(
            Event.create(
                "policy_result",
                self._session.session_id,
                {"reference": action.id, "confirmed": confirmed},
            )
        )
        merged["confirm_result"] = confirmed
        return merged

    async def manual_interrupt(self, state: Dict[str, Any]) -> Dict[str, Any]:
        merged = self._state_with_defaults(state)
        action = merged.get("planned_action")
        reason = action.reason if action else None
        text = reason or "Manual action required. Please handle it and reply when done."
        merged["messages"].append(AIMessage(content=text))
        response = interrupt({"kind": "manual", "text": text})
        message = None
        if isinstance(response, dict):
            message = response.get("message")
        elif isinstance(response, str):
            message = response
        if message:
            merged["messages"].append(HumanMessage(content=str(message).strip()))
            merged["last_user_message"] = str(message).strip() or None
        merged["planned_action"] = None
        merged["needs_observe"] = True
        merged["single_step"] = False
        return merged

    async def execute_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        merged = self._state_with_defaults(state)
        action = merged.get("planned_action")
        if not action:
            merged["final_response"] = "No action planned."
            return merged
        try:
            if action.kind == "observe":
                if merged.get("single_step"):
                    observation = merged.get("last_observation")
                    if observation:
                        merged["final_response"] = self._summary(observation)
                else:
                    merged = await self._observe(merged)
            elif action.kind == "launch":
                await self._call_tool(merged, "launch", {}, action)
            elif action.kind == "click":
                await self._call_tool(
                    merged,
                    "click",
                    {"eid": action.eid, "element": self._describe_element(action.eid, merged)},
                    action,
                )
                merged["needs_observe"] = True
            elif action.kind == "type":
                await self._call_tool(
                    merged,
                    "type",
                    {
                        "eid": action.eid,
                        "value": action.value,
                        "element": self._describe_element(action.eid, merged),
                    },
                    action,
                )
                merged["needs_observe"] = True
            elif action.kind == "scroll":
                await self._call_tool(merged, "scroll", action.args, action)
                merged["needs_observe"] = True
            elif action.kind == "wait":
                await self._call_tool(merged, "wait", action.args, action)
                merged["needs_observe"] = True
            elif action.kind == "screenshot":
                await self._call_tool(merged, "screenshot", {}, action)
            elif action.kind == "stop":
                merged["final_response"] = action.final_response or "Stopping."
            else:
                merged["final_response"] = f"Unknown action '{action.kind}'."
        except ToolError as exc:
            merged["final_response"] = f"Tool failed: {exc}"
        return merged

    async def finalize_done(self, state: Dict[str, Any]) -> Dict[str, Any]:
        merged = self._state_with_defaults(state)
        message = merged.get("final_response")
        if message:
            await self._session.send_message(
                ServerMessage(type="agent_message", payload={"text": message})
            )
            merged["messages"].append(AIMessage(content=message))
        return merged

    async def finalize_denied(self, state: Dict[str, Any]) -> Dict[str, Any]:
        merged = self._state_with_defaults(state)
        message = merged.get("final_response") or "Action denied by policy."
        merged["final_response"] = message
        await self._session.send_message(ServerMessage(type="agent_message", payload={"text": message}))
        merged["messages"].append(AIMessage(content=message))
        return merged

    async def finalize_cancelled(self, state: Dict[str, Any]) -> Dict[str, Any]:
        merged = self._state_with_defaults(state)
        message = merged.get("final_response") or "Action cancelled. Tell me what to do next."
        merged["final_response"] = message
        await self._session.send_message(ServerMessage(type="agent_message", payload={"text": message}))
        merged["messages"].append(AIMessage(content=message))
        return merged

    async def finalize_blocked(self, state: Dict[str, Any]) -> Dict[str, Any]:
        merged = self._state_with_defaults(state)
        message = merged.get("final_response") or "Blocked action. Please clarify the goal."
        merged["final_response"] = message
        await self._session.send_message(ServerMessage(type="agent_message", payload={"text": message}))
        merged["messages"].append(AIMessage(content=message))
        return merged

    async def finalize_limit(self, state: Dict[str, Any]) -> Dict[str, Any]:
        merged = self._state_with_defaults(state)
        summary = self._build_limit_summary(merged)
        merged["final_response"] = summary
        await self._session.send_message(ServerMessage(type="agent_message", payload={"text": summary}))
        merged["messages"].append(AIMessage(content=summary))
        await self._session.send_message(
            ServerMessage(type="agent_question", payload={"text": "Continue?", "reference": None})
        )
        return merged

    def _route_after_observe(self, state: Dict[str, Any]) -> str:
        tool_steps = int(state.get("tool_steps") or 0)
        if state.get("final_response"):
            return "finalize_done"
        if tool_steps >= MAX_TOOL_STEPS:
            return "finalize_limit"
        if state.get("planned_action") is not None:
            return "validate_action"
        return "plan_action"

    def _route_after_validate(self, state: Dict[str, Any]) -> str:
        if int(state.get("invalid_actions") or 0) >= 3:
            return "finalize_blocked"
        action = state.get("planned_action")
        if action and action.kind == "need_user":
            return "manual_interrupt"
        return "policy_gate"

    def _route_after_policy(self, state: Dict[str, Any]) -> str:
        decision = state.get("policy_decision")
        if isinstance(decision, dict):
            decision = PolicyDecision.model_validate(decision)
        if not isinstance(decision, PolicyDecision):
            return "finalize_denied"
        if decision.requires_confirmation:
            return "confirm_interrupt"
        if not decision.allow:
            return "finalize_denied"
        return "execute_action"

    def _route_after_confirm(self, state: Dict[str, Any]) -> str:
        if state.get("confirm_result"):
            return "execute_action"
        return "finalize_cancelled"

    def _route_after_execute(self, state: Dict[str, Any]) -> str:
        tool_steps = int(state.get("tool_steps") or 0)
        if state.get("final_response"):
            return "finalize_done"
        if state.get("single_step"):
            return "finalize_done"
        action = state.get("planned_action")
        if action and action.kind == "stop":
            return "finalize_done"
        if tool_steps >= MAX_TOOL_STEPS:
            return "finalize_limit"
        return "maybe_observe"

    async def _observe(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            screenshot = await self._call_tool(state, "screenshot", {}, Action(kind="screenshot"))
            state["last_screenshot"] = self._extract_screenshot(screenshot)
        except ToolError:
            # Screenshot failures should not block observation or the agent loop.
            state["last_screenshot"] = None
        raw = await self._call_tool(state, "observe", {}, Action(kind="observe"))
        logger.info("Observe raw response: %s", raw)
        observation = self._observer.build(raw)
        state["last_observation"] = observation
        await self._session.publish_event(
            Event.create(
                "observation",
                self._session.session_id,
                {"summary": self._summary(observation), "observation": observation.model_dump()},
            )
        )
        return state

    async def _call_tool(
        self,
        state: Dict[str, Any],
        name: str,
        args: Dict[str, Any],
        action: Action | None,
    ) -> Dict[str, Any]:
        await self._session.publish_event(
            Event.create("tool_call", self._session.session_id, {"name": name, "args": args})
        )
        state["tool_steps"] = int(state.get("tool_steps") or 0) + 1
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
            self._record_failure(state, action, str(exc))
            await self._session.publish_event(
                Event.create("error", self._session.session_id, {"stage": name, "message": str(exc)})
            )
            raise ToolError(str(exc)) from exc
        summary = self._short_result(result)
        await self._session.publish_event(
            Event.create(
                "tool_result",
                self._session.session_id,
                {"name": name, "success": True, "result": summary},
            )
        )
        self._record_step(state, action, summary)
        return result

    def _record_step(self, state: Dict[str, Any], action: Action | None, summary: Dict[str, Any]) -> None:
        label = self._format_action_label(action)
        state["steps"].append(f"{label} -> ok")
        logger.info("Step ok: %s", {"action": label, "summary": summary})

    def _record_failure(self, state: Dict[str, Any], action: Action | None, error: str) -> None:
        label = self._format_action_label(action)
        state["failures"].append(f"{label} -> {error}")
        logger.info("Step failed: %s", {"action": label, "error": error})

    def _record_validation_failure(self, state: Dict[str, Any], action: Action, error: str) -> None:
        label = self._format_action_label(action)
        state["failures"].append(f"{label} -> {error}")
        state["steps"].append(f"{label} -> skipped ({error})")
        logger.info("Validation blocked action: %s", {"action": label, "error": error})

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

    def _validate_action(self, action: Action, observation: Observation | None) -> str | None:
        if action.kind not in {"type", "click"}:
            return None
        element = self._get_element(action.eid, observation)
        if not element:
            return "Target element not found in observation."
        if element.disabled:
            return "Target element is disabled."
        if action.kind == "type":
            role = (element.role or "").lower()
            if role and role not in EDITABLE_ROLES:
                return f"Target element role '{role}' is not editable."
        return None

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

    def _describe_element(self, eid: str | None, state: Dict[str, Any]) -> str | None:
        observation = state.get("last_observation")
        if not eid or not observation:
            return None
        for element in observation.interactive:
            if element.eid == eid:
                parts = [element.role, element.name, element.value]
                return " ".join([part for part in parts if part])
        return None

    def _get_element(
        self, eid: str | None, observation: Observation | None
    ) -> InteractiveElement | None:
        if not eid or not observation:
            return None
        for element in observation.interactive:
            if element.eid == eid:
                return element
        return None

    def _get_browser(self) -> MCPBrowserClient:
        if not self._browser:
            self._browser = MCPBrowserClient(self._session.session_id, self._settings)
        return self._browser

    @staticmethod
    def _extract_screenshot(result: Dict[str, Any]) -> dict[str, str] | None:
        if not isinstance(result, dict):
            return None
        content = result.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "image":
                    continue
                url = item.get("url")
                if isinstance(url, str) and url:
                    return {"url": url}
                data = item.get("data") or item.get("base64")
                if isinstance(data, str) and data:
                    mime_type = item.get("mimeType") or item.get("mime_type") or "image/png"
                    return {"data": data, "mime_type": str(mime_type)}
        url = result.get("url")
        if isinstance(url, str) and url:
            return {"url": url}
        data = result.get("data") or result.get("base64")
        if isinstance(data, str) and data:
            mime_type = result.get("mimeType") or result.get("mime_type") or "image/png"
            return {"data": data, "mime_type": str(mime_type)}
        return None

    def _build_limit_summary(self, state: Dict[str, Any]) -> str:
        steps = state.get("steps") or []
        failures = state.get("failures") or []
        done = "; ".join(steps[:6])
        if len(steps) > 6:
            done = f"{done}; ..."
        if not done:
            done = "No tool actions executed."
        if failures:
            failure_text = "; ".join(failures[:3])
            if len(failures) > 3:
                failure_text = f"{failure_text}; ..."
            failure_line = f"Failures: {failure_text}."
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
        return {key: LangGraphAgent._redact_log_value(result[key]) for key in keys}

    @staticmethod
    def _redact_log_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: LangGraphAgent._redact_log_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [LangGraphAgent._redact_log_value(item) for item in value]
        if isinstance(value, str) and len(value) > 500:
            return f"<omitted {len(value)} chars>"
        return value
