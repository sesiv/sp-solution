from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .models import Action, Event, ServerMessage


class EventBroker:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[Event]] = set()

    def subscribe(self) -> asyncio.Queue[Event]:
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=200)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Event]) -> None:
        self._subscribers.discard(queue)

    async def publish(self, event: Event) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop oldest to keep flow moving.
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    pass


@dataclass
class Session:
    session_id: str
    events: EventBroker = field(default_factory=EventBroker)
    messages: asyncio.Queue[ServerMessage] = field(default_factory=lambda: asyncio.Queue(maxsize=200))
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending_action_id: Optional[str] = None
    pending_action: Optional[Action] = None
    status: str = "idle"
    runner: Optional[Any] = None

    async def publish_event(self, event: Event) -> None:
        await self.events.publish(event)

    async def send_message(self, message: ServerMessage) -> None:
        try:
            self.messages.put_nowait(message)
        except asyncio.QueueFull:
            try:
                self.messages.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self.messages.put_nowait(message)
            except asyncio.QueueFull:
                pass


class SessionManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}

    def get_or_create(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)
        return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)
