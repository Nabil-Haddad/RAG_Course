
# PHASE 0 — PILLAR 1 | LESSON 4: Async Python for RAG Pipelines

# WHY THIS MATTERS:
#   Embedding 5,000 chunks one-by-one takes ~14 minutes.
#   Embedding them concurrently with asyncio takes under 2 minutes.
#   FastAPI, the standard RAG serving framework is fully async.
#   LangChain's async methods (ainvoke, aget_relevant_documents) require async.


# Lesson Content:
#   1. The core mental model, async vs sync, I/O-bound vs CPU-bound
#   2. async def, await, and asyncio.run()
#   3. asyncio.gather(): running coroutines concurrently
#   4. asyncio.Semaphore(): rate-limit protection
#   5. Exponential backoff retry: essential for API calls at scale
#   6. How NOT to mix sync and async (the most common mistake)


import asyncio
import random
import time


# PART 1 — THE MENTAL MODEL: ASYNC vs SYNC

# Sync code: each line waits for the previous to finish.
#   embed(chunk_1)    # waits 0.1s
#   embed(chunk_2)    # waits 0.1s (starts AFTER chunk_1 finishes)
#   embed(chunk_3)    # waits 0.1s (starts AFTER chunk_2 finishes)
#   Total: 0.3s

# Async code: coroutines yield control while waiting for I/O.
#   await embed(chunk_1)  # starts, yields while waiting for network
#   await embed(chunk_2)  # starts IMMEDIATELY while chunk_1 waits
#   await embed(chunk_3)  # starts IMMEDIATELY while chunk_1 and chunk_2 wait
#   Total: 0.1s (all three overlap)

# KEY INSIGHT: async is only faster for I/O-bound work (network calls, disk).
# For CPU-bound work (heavy maths, compression), use multiprocessing instead.
# Embedding API calls are I/O-bound, async wins massively.


# a fake embed function taht returns a 3 dimention array

async def fake_embed(text: str, latency : int = 0.85)->list[float]:
    await asyncio.sleep(latency)
    return [hash(text) % 100 / 100, 0.5, 0.3]

async def embed_sequential(texts : list[str])-> list[list[float]]:
    results = []
    for txt in texts:
        vector = await fake_embed(txt)
        results.append(txt)
    return results

async def embed_concurrent(texts : list[str])->list[list[float]]:
    # Embed all at once — asyncio.gather() runs all coroutines together
    results = [fake_embed(text) for text in texts]
    return await asyncio.gather(*results)


texts = [f"chunk number {i}" for i in range(20)]

start = time.perf_counter()
seq_results = asyncio.run(embed_sequential(texts))
seq_time = time.perf_counter() - start
 
start = time.perf_counter()
con_results = asyncio.run(embed_concurrent(texts))
con_time = time.perf_counter() - start


print(f"Sequential : {seq_time:.3f}s  ({len(seq_results)} embeddings)")
print(f"Concurrent : {con_time:.3f}s  ({len(con_results)} embeddings)")
print(f"Speedup : {seq_time / con_time:.1f}x faster")



# PART 2 — async def, await, asyncio.run()

# async def  — marks a function as a coroutine. Calling it returns a coroutine
#              object; it does NOT run the function. You must await it.
# await      — runs a coroutine and pauses the current one until it finishes.
#              Can only be used inside an async def function.
# asyncio.run() — the entry point. Creates the event loop, runs one top-level
#                 coroutine, then closes the loop. Use once at the top level.



async def fetch_answer(query: str, latency : int = 0.85)->str:
    await asyncio.sleep(latency)
    return f"Answer to: {query}"


async def rag_query(question:str)-> dict:
    query_vec = await fake_embed(question)
    answer = await fetch_answer(query=question)
    return {"question": question, "answer": answer, "query_vec": query_vec}

result = asyncio.run(rag_query("What is RAG?"))

print(f"question : {result['question']}")
print(f"answer : {result['answer']}")
print(f"vec dims : {len(result['query_vec'])}")



# PART 3 — asyncio.Semaphore: RATE-LIMIT PROTECTION

# Problem: asyncio.gather() with 5,000 tasks fires ALL 5,000 API calls at once.
# OpenAI Tier 1 allows ~3,000 requests per minute (50/second).
# 5,000 simultaneous calls → mass 429 Too Many Requests errors.

# Solution: asyncio.Semaphore(N) limits concurrent coroutines to N at any time.
# The others wait in queue. When one finishes, the next starts.
# This is like a bouncer at a club — only N people inside at once.


MAX_CONCURRENT = 5
semaphore = asyncio.Semaphore(MAX_CONCURRENT)

# Tarck how many calls are running simultaneously 
active_calls = 0
max_observed = 0

async def rate_limited_embed(text:str)->list[float]:
    global active_calls, max_observed
    async with semaphore:
        active_calls += 1
        max_observed = max(max_observed, active_calls)
        result = await fake_embed(text, latency= 0.02)
        active_calls -=1
        return result


async def embed_with_limit(texts: list[str])-> list[list[float]]:
    tasks = [rate_limited_embed(text) for text in texts]
    return await asyncio.gather(*tasks)


chunks_30 = [f"chunk {i}" for i in range(30)]
vectors = asyncio.run(embed_with_limit(chunks_30))


print(f"Embedded  : {len(vectors)} chunks")
print(f"Max concurrent : {max_observed} (limit was {MAX_CONCURRENT})")
print(f"Limit respected : {max_observed <= MAX_CONCURRENT}")
 