from typing import Any, Literal

from pydantic import BaseModel, Field

Severity = Literal[
    "low",
    "medium",
    "high",
    "critical",
]


class AgentRequest(BaseModel):
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    message: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
    )


class AgentGraphRequest(BaseModel):
    device_id: str | None = None

    message: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
    )


class DiagnosisResult(BaseModel):

    summary: str = Field(
        description="故障现象和初步判断"
    )

    fault_type: str = Field(
        description="故障类型"
    )

    confidence: float = Field(
        ge=0,
        le=1,
        description="诊断置信度"
    )

    evidence: list[str] = Field(
        description="诊断依据"
    )

    recommended_actions: list[str] = Field(
        description="建议处理措施"
    )

    risk_level: Literal[
        "low",
        "medium",
        "high",
        "critical",
    ]

    requires_human_confirmation: bool

    sources: list[str] = Field(
        default_factory=list
    )


class ToolCallInfo(BaseModel):
    tool_name: str

    arguments: dict[str, Any]

    success: bool


class AgentEventInfo(BaseModel):

    type: str

    step: int

    message: str

    data: dict[str, Any] = Field(
        default_factory=dict
    )


class AgentResponse(BaseModel):

    session_id: str

    status: Literal[
        "completed",
        "failed",
        "max_steps",
        "waiting_for_approval",
    ]

    diagnosis: DiagnosisResult | None = None

    steps: int

    tool_calls: list[ToolCallInfo]

    trace: list[AgentEventInfo]

    error: str | None = None