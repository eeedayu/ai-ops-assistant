from dataclasses import dataclass, field
from typing import Any, Literal


AgentStatus = Literal[
    "running",
    "completed",
    "failed",
    "max_steps",
    "waiting_for_approval",
]


@dataclass
class ToolCallRecord:

    tool_call_id: str

    tool_name: str

    arguments: dict[str, Any]

    result: Any = None

    success: bool = True


@dataclass
class AgentEvent:

    type: str

    step: int

    message: str

    data: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class AgentState:

    session_id: str

    messages: list[dict[str, Any]] = field(
        default_factory=list
    )

    step_count: int = 0

    status: AgentStatus = "running"

    final_answer: str | None = None

    diagnosis: Any | None = None

    error: str | None = None

    repeated_tool_calls: dict[
        str,
        int,
    ] = field(
        default_factory=dict
    )

    pending_approval_tool: dict[str, Any] | None = None

    trace: list[AgentEvent] = field(
        default_factory=list
    )


