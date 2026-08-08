

# Default similarity cutoff used by search_cache() if no threshold is
# passed in explicitly. Similarity is a cosine similarity score in [0, 1],
# where 1.0 = identical meaning and 0.0 = completely unrelated.
DEFAULT_THRESHOLD = 0.85

# A looser threshold — use this if you'd rather get more cache hits and
# are okay with occasional near-matches instead of exact ones.
LOOSE_THRESHOLD = 0.75

# A stricter threshold — use this when wrong cached answers are costly
# and you only want to reuse answers for near-identical questions.
STRICT_THRESHOLD = 0.92

# Distance metric used when creating the ChromaDB collection. "cosine" is
# the standard choice for sentence-transformer embeddings.
DISTANCE_METRIC = "cosine"