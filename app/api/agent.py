import uuid
from typing import Any, Mapping

from fastapi import APIRouter, Depends
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from app.agent.graph import graph
from app.agent.graph_state import AgentState, ResumeRequest
from app.agents.agent import AgentService
from app.agents.state import AgentState as LegacyAgentState
from app.dependencies import get_agent_service
from app.schemas.agent import (
    AgentEventInfo,
    AgentGraphRequest,
    AgentRequest,
    AgentResponse,
    ToolCallInfo,
)

router = APIRouter(
    prefix="/agent",
    tags=["agent"],
)


async def resopnse_construct(state: LegacyAgentState) -> AgentResponse:
    tool_calls = []
    trace = []
    for event in state.trace:
        if event.type == "tool_result":
            tool_info = event.data.get("tool_info")
            if tool_info:
                tool_calls.append(
                    ToolCallInfo(
                        tool_name=tool_info.tool_name,
                        arguments=tool_info.arguments,
                        success=tool_info.success,
                    )
                )
        trace.append(
            AgentEventInfo(
                type=event.type,
                step=event.step,
                message=event.message,
                data=event.data,
            )
        )

    return AgentResponse(
        session_id=state.session_id,
        status=state.status,
        diagnosis=state.diagnosis,
        steps=state.step_count,
        tool_calls=tool_calls,
        trace=trace,
        error=state.error,
    )


@router.post(
    "/chat",
    response_model=AgentResponse,
)
async def agent_chat(
    request: AgentRequest,
    agent_service: AgentService = Depends(
        get_agent_service
    ),
):

    state = await agent_service.run(
        session_id=request.session_id,
        user_message=request.message,
    )

    return await resopnse_construct(state)


@router.post(
    "/{session_id}/approve",
    response_model=AgentResponse,
)
async def agent_approve(
    request: AgentRequest,
    agent_service: AgentService = Depends(
        get_agent_service
    ),
):

    state = await agent_service.approve(
        session_id=request.session_id,
    )

    return await resopnse_construct(state)


@router.post(
    "/{session_id}/reject",
    response_model=AgentResponse,
)
async def agent_reject(
    request: AgentRequest,
    agent_service: AgentService = Depends(
        get_agent_service
    ),
):

    state = await agent_service.reject(
        session_id=request.session_id,
    )

    return await resopnse_construct(state)




def graph_response(
    thread_id: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    interrupts = result.get("__interrupt__", ())
    pending_tool = interrupts[0].value if interrupts else None

    return {
        "session_id": thread_id,
        "status": (
            "waiting_for_approval"
            if pending_tool is not None
            else "completed"
        ),
        "pending_tool": pending_tool,
        "diagnosis": result.get("diagnosis"),
        "trace": result.get("trace", []),
    }


@router.post("/run")
async def run_agent(
    request: AgentGraphRequest,
):

    thread_id = str(
        uuid.uuid4()
    )

    config: RunnableConfig = {
        "configurable": {
            "thread_id":
                thread_id,
        }
    }

    initial_state: AgentState = {
        "messages": [
            HumanMessage(content=request.message),
        ],
        "user_input": request.message,
        "device_id": request.device_id,
        "memories": [],
        "diagnosis": None,
        "trace": [],
    }

    result = await graph.ainvoke(
        initial_state,
        config=config,
    )

    return graph_response(thread_id, result)




@router.post(
    "/{thread_id}/resume"
)
async def resume_agent(
    thread_id: str,
    request: ResumeRequest,
):

    config: RunnableConfig = {
        "configurable": {
            "thread_id":
                thread_id,
        }
    }

    result = await graph.ainvoke(
        Command(
            resume={
                "approved":
                    request.approved,

                "reason":
                    request.reason,
            }
        ),
        config,
    )

    return graph_response(thread_id, result)