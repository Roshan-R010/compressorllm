import chromadb

from backend.cache.embeddings import embed_text
from backend.cache.threshold_config import DEFAULT_THRESHOLD, DISTANCE_METRIC

# ---------------------------------------------------------------------------
# Set up ChromaDB. We use a persistent client so the cache survives
# restarts (stored on disk in ./chroma_db). Swap for chromadb.Client()
# if you only want an in-memory cache.
# ---------------------------------------------------------------------------
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# A "collection" is like a table. get_or_create so re-running the script
# doesn't error out if it already exists.
collection = chroma_client.get_or_create_collection(
    name="semantic_cache",
    metadata={"hnsw:space": DISTANCE_METRIC},
)

# Chroma needs unique string IDs per entry. We just keep an incrementing counter.
_next_id = collection.count()


def add_to_cache(question: str, answer: str) -> None:
    """
    Embed `question` (via embeddings.py) and store it in ChromaDB alongside
    its `answer`.

    The answer is stored in the `metadatas` field (Chroma only indexes
    embeddings/documents for search; metadata is just attached payload).
    """
    global _next_id

    embedding = embed_text(question)

    entry_id = str(_next_id)
    _next_id += 1

    collection.add(
        ids=[entry_id],
        embeddings=[embedding],
        documents=[question],            # original question text (for debugging/inspection)
        metadatas=[{"answer": answer}],  # cached answer, retrievable on lookup
    )


def search_cache(question: str, threshold: float = DEFAULT_THRESHOLD):
    """
    Embed `question`, find the most similar stored question, and return its
    cached answer if similarity >= threshold. Otherwise return None.

    threshold is a similarity score in [0, 1], where 1 = identical meaning.
    Defaults to DEFAULT_THRESHOLD from threshold_config.py, but callers can
    override it (e.g. pass LOOSE_THRESHOLD or STRICT_THRESHOLD).
    """
    # Nothing stored yet -> nothing to match against.
    if collection.count() == 0:
        return None

    query_embedding = embed_text(question)

    # Ask Chroma for the single closest match (n_results=1).
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
    )

    # No results returned (shouldn't normally happen if count() > 0, but be safe).
    if not results["ids"] or not results["ids"][0]:
        return None

    # Chroma returns "distance", not similarity. With cosine space,
    # distance = 1 - cosine_similarity, so we convert it back.
    distance = results["distances"][0][0]
    similarity = 1 - distance

    if similarity >= threshold:
        cached_answer = results["metadatas"][0][0]["answer"]
        return cached_answer

    return None


def get_cache_size() -> int:
    """Return how many question/answer pairs are currently stored."""
    return collection.count()


# ---------------------------------------------------------------------------
# Test script — run this file directly to sanity-check the cache works.
#   python vector_store.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from threshold_config import LOOSE_THRESHOLD

    print("Adding 3 example questions to cache...\n")

    add_to_cache("What is AI?", "AI stands for Artificial Intelligence.")
    add_to_cache(
        "How do I reverse a list in Python?",
        "Use my_list.reverse() or my_list[::-1].",
    )
    add_to_cache(
        "What is the capital of France?",
        "The capital of France is Paris.",
    )

    print(f"Cache size after adding: {get_cache_size()}\n")

    # Reworded versions of the same 3 questions — should still hit the cache
    # if the embeddings + threshold are working correctly.
    test_queries = [
        "Explain artificial intelligence",           # reword of Q1
        "How can I reverse the order of a list?",    # reword of Q2
        "Which city is France's capital?",           # reword of Q3
        "What's the weather like today?",             # unrelated — should MISS
    ]

    for query in test_queries:
        result = search_cache(query, threshold=LOOSE_THRESHOLD)
        print(f"Query: {query!r}")
        if result:
            print(f"  -> CACHE HIT: {result}")
        else:
            print("  -> CACHE MISS (no similar question found)")
        print()