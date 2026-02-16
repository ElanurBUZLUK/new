import json
import os
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"

Difficulty = Literal["beginner", "intermediate", "advanced"]


@lru_cache(maxsize=64)
def _load_entries(language: str) -> list[dict]:
    path = CLEAN_DIR / language / "word_list_clean.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Önce scripts/build_wordlists.py çalıştır."
        )
    return json.loads(path.read_text(encoding="utf-8"))


@tool
def get_n_random_words(language: str, n: int = 10) -> list[str]:
    """Return n random words from the cleaned word list for the given language."""
    entries = _load_entries(language)
    words = [e["word"] for e in entries]
    n = min(n, len(words))
    return random.sample(words, n)


@tool
def get_n_random_words_by_difficulty_level(
    language: str,
    word_difficulty: Difficulty,
    n: int = 10,
) -> list[str]:
    """
    Return n random words from the cleaned list filtered by difficulty.
    word_difficulty must be one of: beginner, intermediate, advanced.
    """
    entries = _load_entries(language)
    filtered = [e["word"] for e in entries if e.get("word_difficulty") == word_difficulty]
    if not filtered:
        raise ValueError(f"No words for {language=} {word_difficulty=}.")
    n = min(n, len(filtered))
    return random.sample(filtered, n)


def _translation_llm():
    provider = os.getenv("TRANSLATION_PROVIDER", "ollama").lower()
    if provider == "openai":
        return ChatOpenAI(model=os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4o-mini"), temperature=0)
    # dokümandaki gibi: ayrı bir modelle (örn llama 3.2) çeviri :contentReference[oaicite:26]{index=26}
    return ChatOllama(model=os.getenv("OLLAMA_TRANSLATION_MODEL", "llama3.2"), temperature=0)


@tool
def translate_words(
    words: list[str],
    source_language: str,
    target_language: str,
) -> dict:
    """
    Translate words from source_language to target_language.
    Returns a JSON dict mapping each source word to its translation.
    """
    llm = _translation_llm()
    prompt = (
        f"Translate the following {source_language} words into {target_language}.\n"
        f"Return ONLY a valid JSON object mapping each source word to its translation.\n"
        f"No extra text.\n\nWords: {words}"
    )
    resp = llm.invoke(prompt)
    text = resp.content if hasattr(resp, "content") else str(resp)

    # Dokümanda: model bazen ekstra text basar; { ... } yakalayıp parse ediyoruz :contentReference[oaicite:27]{index=27}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"Could not find JSON in model output: {text[:200]}")
    data = json.loads(m.group(0))

    # sanity-check
    missing = [w for w in words if w not in data]
    if missing:
        raise ValueError(f"Missing translations for: {missing}")
    return data
