
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