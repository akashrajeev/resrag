from __future__ import annotations

from src.providers import get_provider_client, get_provider_config, get_provider_model


def test_provider_configs_have_expected_endpoints():
    assert get_provider_config("openai").base_url is None
    assert get_provider_config("openrouter").base_url == "https://openrouter.ai/api/v1"
    assert get_provider_config("groq").base_url == "https://api.groq.com/openai/v1"


def test_provider_model_prefers_specific_environment(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "test/groq-model")
    monkeypatch.setenv("LLM_MODEL", "fallback-model")
    assert get_provider_model("groq") == "test/groq-model"


def test_provider_model_uses_generic_fallback(monkeypatch):
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.setenv("LLM_MODEL", "fallback-model")
    assert get_provider_model("openrouter") == "fallback-model"


def test_provider_client_uses_provider_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "test-model")
    client, model = get_provider_client("groq")
    assert model == "test-model"
    assert client.base_url.host == "api.groq.com"
    assert client.base_url.path == "/openai/v1/"
