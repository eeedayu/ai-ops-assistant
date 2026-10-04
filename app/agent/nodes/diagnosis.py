from app.schemas.agent import (
    DiagnosisResult,
)
from app.agent.graph_state import AgentState
from app.agent.nodes.agent import llm


async def diagnosis_node(
    state: AgentState,
):

    result = await llm.with_structured_output(
        DiagnosisResult
    ).ainvoke(
        state["messages"]
    )

    diagnosis = result.model_dump()

    return {
        "diagnosis": diagnosis,

        "trace": [
            {
                "event": "diagnosis",
                "data": diagnosis,
            }
        ],
    }