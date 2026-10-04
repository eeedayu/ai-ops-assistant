from app.agents.state import AgentState


class AgentStateStore:

    def __init__(self):
        self.states: dict[str, AgentState] = {}

    async def save(
        self,
        state: AgentState,
    ):
        self.states[state.session_id] = state

    async def get(
        self,
        session_id: str,
    ) -> AgentState | None:

        return self.states.get(session_id)

    async def delete(
        self,
        session_id: str,
    ):
        self.states.pop(
            session_id,
            None,
        )