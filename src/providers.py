from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    name: str
    api_key_env: str
    model_env: str
    base_url: str | None = None
    default_model: str = ""


PROVIDERS: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        name="OpenAI",
        api_key_env="OPENAI_API_KEY",
        model_env="OPENAI_MODEL",
        default_model="",
    ),
    "openrouter": ProviderConfig(
        name="OpenRouter",
        api_key_env="OPENROUTER_API_KEY",
        model_env="OPENROUTER_MODEL",
        base_url="https://openrouter.ai/api/v1",
        default_model="",
    ),
    "groq": ProviderConfig(
        name="Groq",
        api_key_env="GROQ_API_KEY",
        model_env="GROQ_MODEL",
        base_url="https://api.groq.com/openai/v1",
        default_model="openai/gpt-oss-20b",
    ),
}


def provider_keys() -> list[str]:
    return list(PROVIDERS)


def get_provider_config(provider: str | None = None) -> ProviderConfig:
    selected = (provider or os.getenv("LLM_PROVIDER", "openai")).strip().lower()
    if selected not in PROVIDERS:
        supported = ", ".join(provider_keys())
        raise ValueError(f"Unsupported LLM provider '{selected}'. Choose one of: {supported}.")
    return PROVIDERS[selected]


def get_provider_model(provider: str, override: str | None = None) -> str:
    config = get_provider_config(provider)
    model = (override or os.getenv(config.model_env) or os.getenv("LLM_MODEL") or config.default_model).strip()
    if not model:
        raise RuntimeError(f"Add {config.model_env} or LLM_MODEL to your .env file for {config.name}.")
    return model


def get_provider_client(provider: str, *, model_override: str | None = None) -> tuple[OpenAI, str]:
    config = get_provider_config(provider)
    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"Add {config.api_key_env} to your .env file for {config.name}.")

    kwargs: dict[str, object] = {"api_key": api_key}

    if config.base_url:
        kwargs["base_url"] = config.base_url
    elif os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]

    if provider == "openrouter":
        headers: dict[str, str] = {}
        if os.getenv("OPENROUTER_SITE_URL"):
            headers["HTTP-Referer"] = os.environ["OPENROUTER_SITE_URL"]
        if os.getenv("OPENROUTER_APP_NAME"):
            headers["X-Title"] = os.environ["OPENROUTER_APP_NAME"]
        if headers:
            kwargs["default_headers"] = headers

    return OpenAI(**kwargs), get_provider_model(provider, model_override)


def get_completion_extras(provider: str, model: str | None = None) -> dict[str, object]:
    """Provider-specific request hints focused on latency."""
    extra_body: dict[str, object] = {}
    if provider == "openrouter":
        extra_body["provider"] = {"sort": "latency"}
    if provider == "groq" and model and "gpt-oss" in model.lower():
        # Simple PDF QA rarely needs medium/high reasoning. Low effort retains
        # reasoning capability while avoiding unnecessary reasoning tokens.
        extra_body["reasoning_effort"] = "low"
        extra_body["include_reasoning"] = False
    return {"extra_body": extra_body} if extra_body else {}
