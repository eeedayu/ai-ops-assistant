from app.repositories.session_repository import (
    SessionRepository,
)


def test_create_session():

    repository = SessionRepository()

    session = repository.create(
        "test-001"
    )

    assert session.session_id == "test-001"


def test_add_message():

    repository = SessionRepository()

    repository.create(
        "test-001"
    )

    repository.add_message(
        session_id="test-001",
        role="user",
        content="你好",
    )

    messages = repository.get_messages(
        "test-001"
    )

    assert len(messages) == 1

    assert (
        messages[0]["content"]
        == "你好"
    )