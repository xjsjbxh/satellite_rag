from __future__ import annotations

from satellite_rag.env import load_dotenv


def test_load_dotenv_reads_key_value_lines(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# comment",
                "LITELLM_API_KEY=abc123",
                'RAG_EMBEDDING_MODEL="embed-model"',
                "ANSWER_LLM_MODEL=chat-model # inline comment",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.delenv("RAG_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("ANSWER_LLM_MODEL", raising=False)

    loaded = load_dotenv(env_path)

    assert loaded == env_path
    assert "LITELLM_API_KEY" in __import__("os").environ
    assert __import__("os").environ["LITELLM_API_KEY"] == "abc123"
    assert __import__("os").environ["RAG_EMBEDDING_MODEL"] == "embed-model"
    assert __import__("os").environ["ANSWER_LLM_MODEL"] == "chat-model"


def test_load_dotenv_does_not_override_existing_values(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("LITELLM_API_KEY=file-value\n", encoding="utf-8")
    monkeypatch.setenv("LITELLM_API_KEY", "process-value")

    load_dotenv(env_path)

    assert __import__("os").environ["LITELLM_API_KEY"] == "process-value"
