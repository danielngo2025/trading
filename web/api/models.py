"""LLM providers/models the UI can offer, filtered to what this machine can actually use."""

import os
import shutil

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.model_catalog import MODEL_OPTIONS

PROVIDERS = {
    "claude-cli": ("Claude (local CLI, uses your subscription)", lambda: shutil.which("claude") is not None),
    "anthropic": ("Anthropic API", lambda: bool(os.getenv("ANTHROPIC_API_KEY"))),
    "openai": ("OpenAI API", lambda: bool(os.getenv("OPENAI_API_KEY"))),
    "google": ("Google Gemini API", lambda: bool(os.getenv("GOOGLE_API_KEY"))),
    "xai": ("xAI Grok API", lambda: bool(os.getenv("XAI_API_KEY"))),
    "ollama": ("Ollama (local)", lambda: shutil.which("ollama") is not None),
}


def _options(provider: str, mode: str) -> list[dict]:
    return [{"value": value, "label": label} for label, value in MODEL_OPTIONS[provider][mode]]


def list_models() -> dict:
    providers = [
        {"key": key, "label": label, "available": available(),
         "deep": _options(key, "deep"), "quick": _options(key, "quick")}
        for key, (label, available) in PROVIDERS.items()
        if key in MODEL_OPTIONS
    ]
    return {
        "providers": providers,
        "default": {"provider": DEFAULT_CONFIG["llm_provider"], "deep": DEFAULT_CONFIG["deep_think_llm"],
                    "quick": DEFAULT_CONFIG["quick_think_llm"]},
    }


def validate(provider: str, deep: str, quick: str) -> str | None:
    if provider not in PROVIDERS or provider not in MODEL_OPTIONS:
        return f"Unknown provider {provider}"
    if not PROVIDERS[provider][1]():
        return f"{PROVIDERS[provider][0]} is not set up on this machine"
    for mode, model in (("deep", deep), ("quick", quick)):
        if model not in {o["value"] for o in _options(provider, mode)}:
            return f"{model} is not a {mode} model for {provider}"
    return None
