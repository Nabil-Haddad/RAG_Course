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



# PART 2 — STRUCTURED JSON LOGGING

# print() is fine for debugging. For production you need logs that are:
#   - Machine-readable (CloudWatch, Datadog, Grafana can parse JSON)
#   - Filterable ("show me all queries slower than 2000ms")
#   - Auditable ("who queried what, at what time, from which source")

# Every log line emitted by a RAG pipeline should include:
#   timestamp, level, logger name, message, and domain-specific fields.



class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp" : datetime.now(UTC).isoformat(),  # ISO 8601, UTC, timezone-aware
            "level" : record.levelname,
            "logger" : record.name,
            "message" : record.getMessage(),
        }
        # Merge any extra fields the caller passed via `extra={...}`
        if hasattr(record, "extra"):
            entry.update(record.extra)
        # Attach exception info if present
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
 
        return json.dumps(entry, default=str)  # default=str handles non-serialisable types
 
 
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    return logger
 
 
logger = get_logger("rag_pipeline")
 
# Standard log call — message only
logger.info("Pipeline started")
 
# Log with extra domain-specific fields
def log_query_complete(logger: logging.Logger, query : str, n_chunks: int, ret_ms: float,gen_ms:float) -> None:
    record = logging.LogRecord(
        name="rag_pipeline", level=logging.INFO,
        pathname="", lineno=0,
        msg="query_complete", args=(), exc_info=None
    )
    record.extra = {
        "query_preview" : query[:80],
        "chunks_used" : n_chunks,
        "retrieval_ms" : round(ret_ms),
        "generation_ms" : round(gen_ms),
        "total_ms" : round(ret_ms + gen_ms),
    }
    logger.handle(record)
 
log_query_complete(logger, "What is the FCA consumer duty?", 5, 142.3, 887.6)
 
# Log a warning — for non-critical issues like short chunks being skipped
logger.warning("Short chunk skipped", extra={"chunk_id": "fca_p3_c007", "word_count": 4})



# PART 3 — CONTEXT MANAGERS: SAFE RESOURCE HANDLING


# A context manager guarantees that cleanup code runs even if an exception
# is raised inside the `with` block. You already use them:
#   `with open(...) as f:`  → file is closed even if you raise inside
#   `async with semaphore:` → semaphore is released even if you raise inside


# For RAG, write context managers for:
#   - Vector DB connections (must be closed cleanly)
#   - Temporary directories (must be deleted after use)
#   - Timing blocks (measure latency of pipeline segments)
 
print("\n--- Part 3: Context managers ---")
 
@contextmanager
def timer(label: str) -> Generator[None, None, None]:
    start = time.perf_counter()
    try:
        yield   # execution enters the `with` block here
    finally:
        # `finally` runs even if an exception was raised inside the block
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("timer", extra={"label": label, "elapsed_ms": round(elapsed_ms, 1)})
        print(f"  [{label}] {elapsed_ms:.1f}ms")
 
# Demonstrate the timer
with timer("fake_embedding_step"):
    time.sleep(0.05)   # simulate work
 
with timer("fake_retrieval_step"):
    _ = [i**2 for i in range(100_000)]
 
@contextmanager
def temp_output_dir(base: Path) -> Generator[Path, None, None]:
    # Create a temporary directory
    import shutil
    import uuid
    tmp_dir = base / f"tmp_{uuid.uuid4().hex[:8]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created temp dir: {tmp_dir}")
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Cleaned up temp dir: {tmp_dir}")
 
with temp_output_dir(Path("/tmp")) as out_dir:
    (out_dir / "test.json").write_text('{"status": "ok"}')
    print(f"Wrote to: {out_dir / 'test.json'}")
# Directory is deleted here automatically