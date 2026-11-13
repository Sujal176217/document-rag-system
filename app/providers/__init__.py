from . import builtin  # noqa: F401  (registers google / openai / anthropic)
from .factory import describe_providers, embedding_id, get_embeddings, get_llm
from .registry import ConfigError, ProviderSpec, register_provider

__all__ = [
    "ConfigError",
    "ProviderSpec",
    "describe_providers",
    "embedding_id",
    "get_embeddings",
    "get_llm",
    "register_provider",
]
