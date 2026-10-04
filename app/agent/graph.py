from typing import Literal

from langgraph.checkpoint.memory import (
    InMemorySaver,
)
from langgraph.graph import (
    END,
    START,
    StateGraph,
)
from langgraph.store.memory import (
    InMemoryStore,
)

from app.agent.graph_state import (
    AgentState,
)
from app.agent.nodes.agent import (
    agent_node,
)
from app.agent.nodes.diagnosis import (
    diagnosis_node,
)
from app.agent.nodes.memory import (
    load_memory,
    save_memory,
)
from app.agent.nodes.tools import (
    tool_node,
)


def should_continue(
    state: AgentState,
) -> Literal["tools", "diagnose"]:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "diagnose"


# ==========================================================
# Graph
# ==========================================================

builder = StateGraph(
    AgentState
)


builder.add_node(
    "load_memory",
    load_memory,
)

builder.add_node(
    "agent",
    agent_node,
)

builder.add_node(
    "tools",
    tool_node,
)

builder.add_node(
    "diagnose",
    diagnosis_node,
)

builder.add_node(
    "save_memory",
    save_memory,
)


# ==========================================================
# Edges
# ==========================================================

builder.add_edge(
    START,
    "load_memory",
)

builder.add_edge(
    "load_memory",
    "agent",
)


builder.add_conditional_edges(
    "agent",

    should_continue,

    {
        "tools":
            "tools",

        "diagnose":
            "diagnose",
    },
)


builder.add_edge(
    "tools",
    "agent",
)

builder.add_edge(
    "diagnose",
    "save_memory",
)

builder.add_edge(
    "save_memory",
    END,
)


# ==========================================================
# Compile
# ==========================================================

checkpointer = InMemorySaver()

store = InMemoryStore()


graph = builder.compile(
    checkpointer=checkpointer,
    store=store,
)