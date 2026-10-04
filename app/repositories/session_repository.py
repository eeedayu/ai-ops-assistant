from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Session:
    session_id: str

    messages: list[dict[str, Any]] = field(
        default_factory=list
    )

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class SessionRepository:

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def create(
        self,
        session_id: str,
    ) -> Session:

        session = Session(
            session_id=session_id
        )

        self._sessions[session_id] = session

        return session

    def get(
        self,
        session_id: str,
    ) -> Session | None:

        return self._sessions.get(session_id)

    def get_or_create(
        self,
        session_id: str,
    ) -> Session:

        session = self.get(session_id)

        if session is None:
            return self.create(session_id)

        return session

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:

        session = self.get_or_create(
            session_id
        )

        session.messages.append(
            {
                "role": role,
                "content": content,
            }
        )

        session.updated_at = datetime.now(
            timezone.utc
        )

    def get_messages(
        self,
        session_id: str,
    ) -> list[dict[str, Any]]:

        session = self.get(session_id)

        if session is None:
            return []

        return list(session.messages)

    def delete(
        self,
        session_id: str,
    ) -> bool:

        if session_id not in self._sessions:
            return False

        del self._sessions[session_id]

        return True

    def clear(
        self,
        session_id: str,
    ) -> bool:

        session = self.get(session_id)

        if session is None:
            return False

        session.messages.clear()

        session.updated_at = datetime.now(
            timezone.utc
        )

        return True