# cost_tracker.py
# Cost tracking module for the LLM proxy system.
#
# This module calculates how much money is saved when a request
# is answered from the cache instead of making an LLM API call.


def calculate_savings(cache_hit, api_cost_per_call=0.002):
    """
    Calculate the money saved for a single request.

    If the request is a cache hit, no LLM API call is made,
    so the full API cost is saved.

    If the request is a cache miss, an LLM API call is made,
    so there is no saving.

    Args:
        cache_hit (bool): True if the request was served from cache.
        api_cost_per_call (float): Cost of one LLM API call in dollars.

    Returns:
        float: Amount of money saved for the request.
    """

    if cache_hit:
        return api_cost_per_call

    return 0.0


def get_total_savings(history):
    """
    Calculate the total money saved from a list of past requests.

    Each request in history should contain a 'cache_hit' key
    with a True or False value.

    Example:
        history = [
            {"cache_hit": True},
            {"cache_hit": False},
            {"cache_hit": True}
        ]

    Returns:
        float: Total amount of money saved in dollars.
    """

    total_savings = 0.0

    for request in history:
        total_savings += calculate_savings(request["cache_hit"])

    return total_savings


# ---------------------------------------------------------------------------
# Test script
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Simulate 10 requests.
    # True  = cache hit  -> money saved
    # False = cache miss -> LLM API call made
    history = [
        {"cache_hit": True},
        {"cache_hit": False},
        {"cache_hit": True},
        {"cache_hit": True},
        {"cache_hit": False},
        {"cache_hit": False},
        {"cache_hit": True},
        {"cache_hit": False},
        {"cache_hit": True},
        {"cache_hit": True},
    ]

    print("Cost Tracker Test")
    print("-----------------")

    for i, request in enumerate(history, start=1):
        savings = calculate_savings(request["cache_hit"])

        if request["cache_hit"]:
            print(f"Request {i}: Cache HIT  -> Saved ${savings:.3f}")
        else:
            print(f"Request {i}: Cache MISS -> Saved $0.000")

    total = get_total_savings(history)

    print("-----------------")
    print(f"Total savings: ${total:.3f}")