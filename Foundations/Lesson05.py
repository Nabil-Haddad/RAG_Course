#LESSON 5: Type Hints, Structured Logging & Project Layout

# Lesson Contenta:
#   1. Type hints, the RAG-specific patterns you will use daily
#   2. Type aliases, making complex signatures readable
#   3. JSON structured logging, the production standard
#   4. Context managers, safe resource handling

from __future__ import annotations
from collections.abc import Generator, AsyncIterator
from dataclasses import dataclass, field
from contextlib import contextmanager
from datetime import datetime, UTC
from typing import Optional, Any
from pathlib import Path
import logging
import json
import time


# PART 1 — TYPE HINTS: THE RAG-SPECIFIC PATTERNS

# Type hints do not change runtime behaviour. They are documentation that
# tools (mypy, pyright, your IDE) can check automatically.

# LangChain's entire codebase uses type hints. FastAPI uses them to generate
# API documentation automatically. Pydantic uses them for validatio


def count_words(text: str)->int:
    return len(text.split())

def get_page_label(page_num: Optional[int]) -> str:
    # Optional[int] means the value is either an int or None.
    if page_num is None:
        return "unknown page"
    return f"page {page_num}"


def average_vector(vectors: list[list[float]])-> list[float]:
    if not vectors:
        return []
    n_dim = len(vectors[0])
    return [sum(vec[i] for vec in vectors) / len(vectors) for i in range(n_dim)]


def build_metadata(source: str, page: int) -> dict[str, Any]:
    # dict[str, Any] — string keys, values of any type.
    return {"source": source, "page": page, "indexed_at": time.time()}


# --- TYPE ALIASES — give complex types a name for readability ---
# Without aliases:
#   def retrieve(query_vec: list[float], corpus: list[tuple[str, list[float]]]) -> list[tuple[str, float]]: ...
#
# With aliases:

Vector = list[str]
ChunkText = str
Corpus = list[tuple[ChunkText, Vector]]
Scored  = list[tuple[ChunkText, float]] 


def retrieve_top_k(query_vec: Vector, corpus: Corpus, k: int = 5) -> Scored:
    # Type aliases make this signature readable at a glance.
    import math
    def cosine(a: Vector, b: Vector) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x**2 for x in a))
        mag_b = math.sqrt(sum(x**2 for x in b))
        return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0
 
    scored = [(text, cosine(query_vec, vec)) for text, vec in corpus]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]



# --- Callable and AsyncIterator ---
from collections.abc import Callable
 
# A function that takes a string and returns a float
Scorer = Callable[[str, str], float]
 
# An async generator that streams tokens — used for streaming LLM responses
async def stream_tokens(text: str) -> AsyncIterator[str]:
    #Yields one word at a time, simulating an LLM streaming response.
    import asyncio
    for word in text.split():
        await asyncio.sleep(0)  # yield control
        yield word + " "
 
# Verify basic type hint functions work
print(f"count_words : {count_words('hello world RAG')}")
print(f"get_page_label : {get_page_label(None)}, {get_page_label(42)}")
avg = average_vector([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
print(f"average_vector : {avg}")
meta = build_metadata("fca.pdf", 7)
print(f"build_metadata : {list(meta.keys())}")
 
corpus: Corpus = [
    ("RAG reduces hallucinations", [0.9, 0.1]),
    ("Weather in London is rainy", [0.1, 0.9]),
    ("Embeddings encode semantics", [0.8, 0.2]),
]
top = retrieve_top_k([0.85, 0.15], corpus, k=2)
print(f"retrieve_top_k  : {[text[:25] for text, _ in top]}")