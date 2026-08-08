"""
main.py — FastAPI proxy server for semantic LLM caching.

Flow for POST /query:
  1. Check the semantic cache (vector_store.search_cache) for a similar past question.
  2. If found -> return the cached answer immediately (cache hit, fast path).
  3. If not found -> compress the prompt (compressor.compress_prompt), call Groq,
     store the new question/answer pair in t`he cache, and return the fresh answer.
"""

import os
import time

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Teammate-built modules (assumed to be importable from the project root)
from vector_store import search_cache, add_to_cache
from compressor import compress_prompt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

app = FastAPI(title="LLM Semantic Cache Proxy")

# Allow a local frontend (e.g. React dev server) to call this API directly.
# Tighten allow_origins to your actual frontend URL before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g. ["http://localhost:3000"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    cache_hit: bool
    response_time_ms: float


# ---------------------------------------------------------------------------
# Groq call helper
# ---------------------------------------------------------------------------


async def call_groq(prompt: str) -> str:
    """Send the (compressed) prompt to Groq's OpenAI-compatible chat endpoint
    and return the model's text response."""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(GROQ_API_URL, headers=headers, json=payload)

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Groq API error ({resp.status_code}): {resp.text}",
        )

    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    start_time = time.perf_counter()
    question = request.question

    # Step 1: Check the semantic cache for a sufficiently similar past question.
    cached_answer = search_cache(question)

    if cached_answer is not None:
        # Step 2: Cache hit — skip compression and the LLM call entirely.
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return QueryResponse(
            answer=cached_answer,
            cache_hit=True,
            response_time_ms=round(elapsed_ms, 2),
        )

    # Step 3: Cache miss — compress the prompt to cut down on tokens sent to Groq.
    compressed_question = compress_prompt(question)

    # Step 4: Call the Groq API with the compressed prompt.
    answer = await call_groq(compressed_question)

    # Step 5: Store the original question and the new answer in the cache
    # so future semantically-similar questions can be served instantly.
    add_to_cache(question, answer)

    # Step 6: Return the fresh answer along with cache/timing metadata.
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return QueryResponse(
        answer=answer,
        cache_hit=False,
        response_time_ms=round(elapsed_ms, 2),
    )


# ---------------------------------------------------------------------------
# Local dev entrypoint: `python main.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
