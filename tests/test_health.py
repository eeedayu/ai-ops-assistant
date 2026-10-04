from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import (
    get_chat_service,
)
from app.services.chat_service import (
    ChatService,
)


class FakeChatService:

    async def chat(
        self,
        session_id: str,
        user_message: str,
    ) -> str:

        return (
            f"mock answer: {user_message}"
        )


app.dependency_overrides[
    get_chat_service
] = lambda: FakeChatService()


client = TestClient(app)


def test_chat():

    response = client.post(
        "/chat",
        json={
            "session_id": "test-001",
            "message": "CPU很高怎么办？",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["answer"]
        == "mock answer: CPU很高怎么办？"
    )


def test_chat_validation():

    response = client.post(
        "/chat",
        json={
            "session_id": "",
            "message": "",
        },
    )

    assert response.status_code == 422