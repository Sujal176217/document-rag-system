"""Builds the embeddings object and the LLM from settings + environment.

Features:
  * choose providers by name (EMBEDDING_PROVIDER / LLM_PROVIDER)
  * several keys per provider (KEY1,KEY2) -> automatic key fallback
  * backup providers (LLM_FALLBACKS=anthropic,openai) -> automatic failover
Everything takes `settings` and `env` as arguments so tests can inject fakes.
"""
import os
import re
from collections.abc import Mapping

from ..config import Settings, settings as default_settings
from .registry import ConfigError, available_providers, get_api_keys, get_provider


def embedding_id(settings: Settings = default_settings) -> str:
    """e.g. 'google-gemini-embedding-001'. Used to name the vector collection:
    vectors from different embedding models must NEVER be mixed."""
    spec = get_provider(settings.embedding_provider)
    model = settings.embedding_model or spec.default_embedding_model
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", f"{spec.name}-{model}").strip("-_")


def get_embeddings(settings: Settings = default_settings, env: Mapping[str, str] | None = None):
    env = os.environ if env is None else env
    spec = get_provider(settings.embedding_provider)
    if spec.embeddings_factory is None:
        options = [p.name for p in available_providers() if p.embeddings_factory]
        raise ConfigError(
            f"Provider '{spec.name}' has no embeddings API. Set EMBEDDING_PROVIDER to one of {options}."
        )
    keys = get_api_keys(spec, env)
    if not keys:
        raise ConfigError(f"{spec.key_env} is missing. Add it to your .env file.")
    model = settings.embedding_model or spec.default_embedding_model
    return spec.embeddings_factory(model, keys[0])


def get_llm(settings: Settings = default_settings, env: Mapping[str, str] | None = None):
    env = os.environ if env is None else env

    primary = get_provider(settings.llm_provider)
    if primary.llm_factory is None:
        options = [p.name for p in available_providers() if p.llm_factory]
        raise ConfigError(f"Provider '{primary.name}' has no chat models. Choose one of {options}.")
    primary_keys = get_api_keys(primary, env)
    if not primary_keys:
        raise ConfigError(f"{primary.key_env} is missing. Add it to your .env file.")

    model = settings.llm_model or primary.default_llm_model
    llms = [primary.llm_factory(model, key) for key in primary_keys]

    # Optional backups: skipped quietly if the provider has no key configured.
    for name in settings.llm_fallbacks:
        spec = get_provider(name)
        if spec.llm_factory is None:
            continue
        for key in get_api_keys(spec, env):
            llms.append(spec.llm_factory(spec.default_llm_model, key))

    return llms[0] if len(llms) == 1 else llms[0].with_fallbacks(llms[1:])


def describe_providers(settings: Settings = default_settings, env: Mapping[str, str] | None = None) -> dict:
    """Safe summary for /providers and the CLI. NEVER includes key values."""
    env = os.environ if env is None else env
    emb = get_provider(settings.embedding_provider)
    llm = get_provider(settings.llm_provider)
    return {
        "active": {
            "embedding": f"{emb.name}:{settings.embedding_model or emb.default_embedding_model}",
            "llm": f"{llm.name}:{settings.llm_model or llm.default_llm_model}",
            "llm_fallbacks": list(settings.llm_fallbacks),
        },
        "providers": [
            {
                "name": p.name,
                "key_env": p.key_env,
                "keys_configured": len(get_api_keys(p, env)),
                "embeddings": p.embeddings_factory is not None,
                "llm": p.llm_factory is not None,
            }
            for p in available_providers()
        ],
    }
