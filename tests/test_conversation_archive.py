from domain.models import ResearchConversationMessage, ResearchConversationState
from storage.conversations import ConversationStore


def test_archive_survives_restart_without_expanding_model_context(tmp_path):
    path = tmp_path / "chats.sqlite3"
    transcript = [ResearchConversationMessage(role="user", content=f"topic {i}") for i in range(20)]
    ConversationStore(path).save("one", ResearchConversationState(messages=transcript[-8:]), transcript)
    store = ConversationStore(path)
    state, history = store.load("one")
    assert len(state.messages) == 8
    assert len(history) == 20
    assert store.list("topic 0")[0]["id"] == "one"
    assert store.load("other")[1] == []


def test_forget_context_preserves_archive_and_delete_is_scoped(tmp_path):
    store = ConversationStore(tmp_path / "chats.sqlite3")
    message = ResearchConversationMessage(role="user", content="private project")
    state = ResearchConversationState(messages=[message], summary="old facts")
    for key in ("one", "two"):
        store.save(key, state, [message])
    store.forget_context("one")
    context, transcript = store.load("one")
    assert context == ResearchConversationState()
    assert transcript == [message]
    assert store.load("two")[0] == state
    store.delete("one")
    assert [item["id"] for item in store.list()] == ["two"]
