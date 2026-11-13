"""Provider-system tests. No network, no real keys."""
import dataclasses

import pytest
from langchain_core.runnables import RunnableLambda

from app.config import Settings
from app.providers import (
    ConfigError,
    ProviderSpec,
    describe_providers,
    embedding_id,
    get_embeddings,
    get_llm,
    register_provider,
)
from app.providers.registry import parse_keys


def make_settings(**kw) -> Settings:
    # Hermetic: ignore whatever is in the developer's real .env / environment.
    base = Settings(
        embedding_provider="google", embedding_model="",
        llm_provider="google", llm_model="", llm_fallbacks=(),
    )
    return dataclasses.replace(base, **kw)


def test_builtin_providers_are_registered():
    names = {p["name"] for p in describe_providers(env={})["providers"]}
    assert {"google", "openai", "anthropic"} <= names


def test_parse_keys_handles_commas_and_spaces():
    assert parse_keys(" a , b,,c ") == ["a", "b", "c"]
    assert parse_keys("") == []


def test_missing_key_gives_clear_error():
    s = make_settings(llm_provider="google")
    with pytest.raises(ConfigError, match="GOOGLE_API_KEY is missing"):
        get_llm(s, env={})


def test_unknown_provider_lists_options():
    with pytest.raises(ConfigError, match="Available"):
        get_llm(make_settings(llm_provider="nope"), env={})


def test_anthropic_cannot_be_used_for_embeddings():
    s = make_settings(embedding_provider="anthropic")
    with pytest.raises(ConfigError, match="no embeddings API"):
        get_embeddings(s, env={"ANTHROPIC_API_KEY": "x"})


def test_embedding_id_is_collection_safe():
    assert embedding_id(make_settings(embedding_provider="google")) == "google-gemini-embedding-001"
    s = make_settings(embedding_provider="openai", embedding_model="text-embedding-3-small")
    assert embedding_id(s) == "openai-text-embedding-3-small"


def test_describe_never_leaks_key_values():
    info = describe_providers(env={"GOOGLE_API_KEY": "SECRET-1,SECRET-2"})
    assert "SECRET" not in str(info)
    google = next(p for p in info["providers"] if p["name"] == "google")
    assert google["keys_configured"] == 2


# ---- extensibility: a brand-new provider needs only one register call ----

def _fake_llm_factory(model, api_key):
    def run(_):
        if api_key == "bad":
            raise RuntimeError("quota exceeded")
        return f"{model}:{api_key}"
    return RunnableLambda(run)


@pytest.fixture
def fake_provider():
    register_provider(
        ProviderSpec(name="fake", key_env="FAKE_API_KEY",
                     llm_factory=_fake_llm_factory, default_llm_model="fake-1")
    )


def test_new_provider_works_without_touching_other_code(fake_provider):
    llm = get_llm(make_settings(llm_provider="fake"), env={"FAKE_API_KEY": "k1"})
    assert llm.invoke("hi") == "fake-1:k1"


def test_second_key_is_used_when_first_fails(fake_provider):
    llm = get_llm(make_settings(llm_provider="fake"), env={"FAKE_API_KEY": "bad,good"})
    assert llm.invoke("hi") == "fake-1:good"


def test_fallback_provider_is_used_when_primary_fails(fake_provider):
    s = make_settings(llm_provider="fake", llm_fallbacks=("fake",))
    env = {"FAKE_API_KEY": "bad"}  # primary key bad; fallback uses same var -> also bad
    with pytest.raises(RuntimeError):
        get_llm(s, env).invoke("hi")

    register_provider(ProviderSpec(name="backup", key_env="BACKUP_KEY",
                                   llm_factory=_fake_llm_factory, default_llm_model="b-1"))
    s = make_settings(llm_provider="fake", llm_fallbacks=("backup",))
    assert get_llm(s, {"FAKE_API_KEY": "bad", "BACKUP_KEY": "ok"}).invoke("hi") == "b-1:ok"


def test_fallback_without_key_is_skipped(fake_provider):
    s = make_settings(llm_provider="fake", llm_fallbacks=("openai",))
    assert get_llm(s, {"FAKE_API_KEY": "k1"}).invoke("hi") == "fake-1:k1"


# ---- real LangChain classes build (offline: just constructing, no API call) ----

def test_google_objects_construct():
    pytest.importorskip("langchain_google_genai")
    env = {"GOOGLE_API_KEY": "dummy-key"}
    s = make_settings(embedding_provider="google", llm_provider="google")
    assert get_embeddings(s, env) is not None
    assert get_llm(s, env) is not None
