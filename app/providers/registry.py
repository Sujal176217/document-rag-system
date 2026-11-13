"""The provider REGISTRY: the one place that knows which AI providers exist.

To support a new provider you add ONE entry (see builtin.py). Nothing else
in the project changes.
"""
from collections.abc import Callable, Mapping
from dataclasses import dataclass


class ConfigError(RuntimeError):
    """Raised for setup problems (missing key, unknown provider, ...)."""


# factory signature: (model_name, api_key) -> LangChain object
Factory = Callable[[str, str], object]


@dataclass(frozen=True)
class ProviderSpec:
    name: str                          # e.g. "google"
    key_env: str                       # env var holding the key(s), e.g. GOOGLE_API_KEY
    embeddings_factory: Factory | None = None   # None = provider has no embeddings API
    llm_factory: Factory | None = None          # None = provider has no chat models
    default_embedding_model: str = ""
    default_llm_model: str = ""


_REGISTRY: dict[str, ProviderSpec] = {}


def register_provider(spec: ProviderSpec) -> None:
    _REGISTRY[spec.name.lower()] = spec


def get_provider(name: str) -> ProviderSpec:
    try:
        return _REGISTRY[name.lower()]
    except KeyError:
        raise ConfigError(
            f"Unknown provider '{name}'. Available: {sorted(_REGISTRY)}"
        ) from None


def available_providers() -> list[ProviderSpec]:
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def parse_keys(raw: str) -> list[str]:
    """One variable can hold several keys, comma-separated: KEY1,KEY2."""
    return [k.strip() for k in (raw or "").split(",") if k.strip()]


def get_api_keys(spec: ProviderSpec, env: Mapping[str, str]) -> list[str]:
    return parse_keys(env.get(spec.key_env, ""))
