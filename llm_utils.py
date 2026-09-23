"""Shared helper for resilient Anthropic API calls."""

import anthropic

def safe_claude_call(client, **kwargs):
    """Calls client.messages.create, returning None on failure instead
    of raising. Callers should check for None and fall back gracefully
    (e.g. treat a failed grade() as "done" rather than crashing the graph).
    """
    try:
        return client.messages.create(**kwargs)
    except anthropic.APIError as e:
        print(f"Anthropic API call failed: {e}")
        return None