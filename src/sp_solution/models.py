from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class PageInfo(BaseModel):
    url: str | None = None
    title: str | None = None


class InteractiveElement(BaseModel):
    eid: str
    role: str | None = None
    name: str | None = None
    value: str | None = None
    placeholder: str | None = None
    disabled: bool | None = None
    visible: bool | None = None
    bbox: Dict[str, float] | None = None


class Observation(BaseModel):
    page: PageInfo = Field(default_factory=PageInfo)
    interactive: List[InteractiveElement] = Field(default_factory=list)
    text_blocks: List[str] = Field(default_factory=list)
    overlays: List[str] = Field(default_factory=list)


class Action(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    kind: Literal[
        "observe",
        "describe",
        "launch",
        "click",
        "type",
        "scroll",
        "wait",
        "screenshot",
        "stop",
        "need_user",
    ]
    eid: str | None = None
    value: str | None = None
    args: Dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None
    final_response: str | None = None


class Event(BaseModel):
    type: str
    session_id: str
    ts: str
    payload: Dict[str, Any]

    @classmethod
    def create(cls, event_type: str, session_id: str, payload: Dict[str, Any]) -> "Event":
        return cls(
            type=event_type,
            session_id=session_id,
            ts=datetime.now(timezone.utc).isoformat(),
            payload=payload,
        )


class ClientMessage(BaseModel):
    type: Literal["user_message", "user_confirm", "control"]
    payload: Dict[str, Any]


class ServerMessage(BaseModel):
    type: Literal["agent_message", "agent_question", "status"]
    payload: Dict[str, Any]


class PolicyDecision(BaseModel):
    allow: bool
    requires_confirmation: bool
    reason: str | None = None
