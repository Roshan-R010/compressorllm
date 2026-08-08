from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Load the model once at import time. Loading it fresh on every call would
# be slow (it has to read weights off disk / initialize torch each time).
# Because Python caches imports, every other file that imports this module
# reuses this same loaded model instead of loading a new copy.
# ---------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
_model = SentenceTransformer(MODEL_NAME)


def embed_text(text: str) -> list[float]:
    """
    Convert a piece of text into an embedding vector.

    Returns a plain Python list of floats (not a numpy array), because
    downstream consumers like ChromaDB expect plain lists.
    """
    embedding = _model.encode(text)
    return embedding.tolist()


# ---------------------------------------------------------------------------
# Test script — run this file directly to sanity-check embeddings work.
#   python embeddings.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = "What is artificial intelligence?"
    vector = embed_text(sample)

    print(f"Text: {sample!r}")
    print(f"Vector length: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")