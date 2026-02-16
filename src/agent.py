from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from .tools import (
    get_n_random_words,
    get_n_random_words_by_difficulty_level,
    translate_words,
)

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


SYSTEM_PROMPT = """You are a language-learning assistant.
When the user asks for words:
- Use get_n_random_words or get_n_random_words_by_difficulty_level.
When the user asks for translations:
- Use translate_words and return a clean mapping.
If the user wants flashcards (Anki):
- Use MCP tools after you have word->translation pairs.
- Ensure deck exists, then create cards one by one.
Workflow examples:
1) "Give me 10 beginner German words, translate to English, add to deck German Easy":
   - get_n_random_words_by_difficulty_level(language="German", word_difficulty="beginner", n=10)
   - translate_words(words=[...], source_language="German", target_language="English")
   - clanki create deck (if needed)
   - clanki create card for each pair
2) "Create deck + add provided pairs":
   - create/find deck
   - create card for each provided front/back pair
Be concise and structured.
"""


def _reasoning_llm():
    provider = os.getenv("REASONING_PROVIDER", "ollama").lower()
    if provider == "openai":
        return ChatOpenAI(model=os.getenv("OPENAI_REASONING_MODEL", "gpt-4o"), temperature=0)
    # Dokümanda yerel open-source modele geçiş var (Qwen 3) :contentReference[oaicite:31]{index=31}
    return ChatOllama(model=os.getenv("OLLAMA_REASONING_MODEL", "qwen3"), temperature=0)


def _mcp_tools() -> list:
    clanki_js = os.getenv("CLANKI_JS_PATH", "").strip()
    if not clanki_js:
        return []

    js_path = Path(clanki_js)
    if not js_path.exists():
        raise FileNotFoundError(f"CLANKI_JS_PATH does not exist: {js_path}")

    client = MultiServerMCPClient(
        {
            "clanki": {
                "command": "node",
                "args": [str(js_path)],
                "transport": "stdio",
            }
        }
    )
    return asyncio.run(client.get_tools())


def build_graph(extra_tools: list | None = None):
    tools = [
        get_n_random_words,
        get_n_random_words_by_difficulty_level,
        translate_words,
    ]
    tools.extend(_mcp_tools())
    if extra_tools:
        tools.extend(extra_tools)

    llm = _reasoning_llm().bind_tools(tools)

    def assistant(state: AgentState):
        msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        resp = llm.invoke(msgs)
        return {"messages": [resp]}

    g = StateGraph(AgentState)
    g.add_node("assistant", assistant)
    g.add_node("tools", ToolNode(tools))

    g.add_edge(START, "assistant")
    g.add_conditional_edges("assistant", tools_condition)
    g.add_edge("tools", "assistant")

    return g.compile()
