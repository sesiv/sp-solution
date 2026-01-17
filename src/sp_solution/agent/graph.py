from __future__ import annotations

import shlex
from typing import Any, Dict, Optional, TypedDict

from langgraph.graph import END, StateGraph

from ..browser.mcp import MCPBrowserClient
from ..config import Settings
from ..models import Action, Event, Observation, ServerMessage
from ..observation import ObservationBuilder
from ..policy import PolicyGate
from ..session import Session
from .llm import LLMDescriber


class AgentState(TypedDict, total=False):
    session_id: str
    last_user_message: str | None
    observation: Observation
    description: str


class AgentRunner:
    def __init__(self, session: Session, settings: Settings) -> None:
        self._session = session
        self._settings = settings
        self._browser: MCPBrowserClient | None = None
        self._observer = ObservationBuilder()
        self._policy = PolicyGate()
        self._describer = LLMDescriber(settings)
        self._graph = self._build_graph()
        self._last_observation: Observation | None = None

    async def handle_user_message(self, text: str) -> None:
        action = self._parse_action(text)
        if action:
            await self._handle_action(action)
            return
        await self._run_cycle(text)

    async def handle_user_confirm(self, reference: str, confirmed: bool) -> None:
        if not self._session.pending_action_id or reference != self._session.pending_action_id:
            await self._session.send_message(
                ServerMessage(type="agent_message", payload={"text": "No pending action found."})
            )
            return
        pending = self._session.pending_action
        self._session.pending_action = None
        self._session.pending_action_id = None
        await self._session.publish_event(
            Event.create(
                "policy_result",
                self._session.session_id,
                {"reference": reference, "confirmed": confirmed},
            )
        )
        if not confirmed:
            await self._session.send_message(
                ServerMessage(type="agent_message", payload={"text": "Action cancelled."})
            )
            return
        if pending:
            await self._execute_action(pending)
        await self._run_cycle(None)

    async def _run_cycle(self, user_message: Optional[str]) -> None:
        state: AgentState = {"session_id": self._session.session_id, "last_user_message": user_message}
        await self._graph.ainvoke(state)

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("observe", self._node_observe)
        graph.add_node("describe", self._node_describe)
        graph.add_node("wait_user", self._node_wait_user)
        graph.add_edge("observe", "describe")
        graph.add_edge("describe", "wait_user")
        graph.add_edge("wait_user", END)
        graph.set_entry_point("observe")
        return graph.compile()

    async def _node_observe(self, state: AgentState) -> AgentState:
        await self._set_status("running")
        raw = await self._call_tool("observe", {})
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
        state["observation"] = observation
        return state

    async def _node_describe(self, state: AgentState) -> AgentState:
        observation = state.get("observation")
        if not observation:
            state["description"] = "Observation unavailable."
            return state
        description = await self._describer.describe(observation)
        state["description"] = description
        return state

    async def _node_wait_user(self, state: AgentState) -> AgentState:
        description = state.get("description") or "Waiting for input."
        await self._session.send_message(
            ServerMessage(type="agent_message", payload={"text": description})
        )
        await self._set_status("waiting_user")
        return state

    async def _set_status(self, status: str) -> None:
        self._session.status = status
        await self._session.send_message(ServerMessage(type="status", payload={"value": status}))

    async def _call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        await self._session.publish_event(
            Event.create(
                "tool_call",
                self._session.session_id,
                {"name": name, "args": args},
            )
        )
        try:
            browser = self._get_browser()
            if name == "observe":
                result = await browser.observe(**args)
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
            await self._session.publish_event(
                Event.create(
                    "error",
                    self._session.session_id,
                    {"stage": name, "message": str(exc)},
                )
            )
            raise
        await self._session.publish_event(
            Event.create(
                "tool_result",
                self._session.session_id,
                {"name": name, "success": True, "result": self._short_result(result)},
            )
        )
        return result

    async def _handle_action(self, action: Action) -> None:
        if action.kind == "describe":
            await self._describe_last()
            return
        decision = self._policy.assess(action)
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
        await self._execute_action(action)
        await self._run_cycle(None)

    async def _describe_last(self) -> None:
        if not self._last_observation:
            await self._session.send_message(
                ServerMessage(
                    type="agent_message",
                    payload={"text": "No observation yet. Use /observe first."},
                )
            )
            await self._set_status("waiting_user")
            return
        description = await self._describer.describe(self._last_observation)
        await self._session.send_message(
            ServerMessage(type="agent_message", payload={"text": description})
        )
        await self._set_status("waiting_user")

    async def _execute_action(self, action: Action) -> None:
        if action.kind == "click":
            await self._call_tool(
                "click",
                {"eid": action.eid, "element": self._describe_element(action.eid)},
            )
        elif action.kind == "type":
            await self._call_tool(
                "type",
                {
                    "eid": action.eid,
                    "value": action.value,
                    "element": self._describe_element(action.eid),
                },
            )
        elif action.kind == "scroll":
            await self._call_tool("scroll", action.args)
        elif action.kind == "wait":
            await self._call_tool("wait", action.args)
        elif action.kind == "screenshot":
            await self._call_tool("screenshot", {})

    def _parse_action(self, text: str) -> Optional[Action]:
        if not text:
            return None
        if not text.startswith("/"):
            return None
        tokens = shlex.split(text)
        if not tokens:
            return None
        command = tokens[0].lstrip("/").lower()
        if command == "observe":
            return Action(kind="observe")
        if command == "describe":
            return Action(kind="describe")
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

    def _get_browser(self) -> MCPBrowserClient:
        if not self._browser:
            self._browser = MCPBrowserClient(self._session.session_id, self._settings)
        return self._browser

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
        return {key: result[key] for key in keys}
