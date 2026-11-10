"""External services (API-based). Swap providers here and nowhere else.

Note: Anthropic does not offer an embeddings API, so embeddings come from
OpenAI here. You could use Voyage, Cohere or Google instead - only
get_embeddings() needs to change.
"""
from .config import settings


def get_embeddings():
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)


def get_llm():
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is missing. Add it to your .env file.")
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key,
        temperature=0,  # factual answers, no creativity
        max_tokens=settings.max_tokens,
    )
