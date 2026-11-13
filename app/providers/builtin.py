"""Built-in providers. Each factory imports its library lazily, so you only
need the packages for providers you actually use.

ADD A NEW PROVIDER = write two small functions + one register_provider(...).
"""
from .registry import ProviderSpec, register_provider

# ----------------------------- Google (Gemini) -----------------------------
# Note: Google deprecated temperature/top_p/top_k for Gemini 3.x models, so we
# deliberately do not pass them.


def _google_embeddings(model: str, api_key: str):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    if not model.startswith("models/"):
        model = f"models/{model}"
    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)


def _google_llm(model: str, api_key: str):
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, timeout=30, max_retries=1)


# ----------------------------------- OpenAI --------------------------------


def _openai_embeddings(model: str, api_key: str):
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=model, api_key=api_key)


def _openai_llm(model: str, api_key: str):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, api_key=api_key)


# ---------------------------------- Anthropic ------------------------------
# Anthropic has no embeddings API, so it is LLM-only.


def _anthropic_llm(model: str, api_key: str):
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=model, api_key=api_key, temperature=0, max_tokens=800)


register_provider(
    ProviderSpec(
        name="google",
        key_env="GOOGLE_API_KEY",
        embeddings_factory=_google_embeddings,
        llm_factory=_google_llm,
        default_embedding_model="gemini-embedding-001",
        default_llm_model="gemini-3.8-flash",
    )
)
register_provider(
    ProviderSpec(
        name="openai",
        key_env="OPENAI_API_KEY",
        embeddings_factory=_openai_embeddings,
        llm_factory=_openai_llm,
        default_embedding_model="text-embedding-3-small",
        default_llm_model="gpt-4o-mini",  # set LLM_MODEL if your account uses another
    )
)
register_provider(
    ProviderSpec(
        name="anthropic",
        key_env="ANTHROPIC_API_KEY",
        llm_factory=_anthropic_llm,
        default_llm_model="claude-sonnet-4-6",
    )
)
