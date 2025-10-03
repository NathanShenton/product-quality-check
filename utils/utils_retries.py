# utils/utils_retries.py
import time
from typing import Any, Dict, List, Optional
from openai import OpenAI

RETRIABLE_SUBSTRINGS = (
    "timeout", "temporarily unavailable", "overloaded", "rate limit",
    "socket", "read timed out", "429", "5xx", "connection reset",
)

def _is_retriable(err: Exception) -> bool:
    s = str(err).lower()
    return any(k in s for k in RETRIABLE_SUBSTRINGS)

def safe_chat_completion(
    client: OpenAI,
    *,
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float = 0.0,
    top_p: float = 0.0,
    max_retries: int = 6,
    base_delay: float = 1.5,
    max_delay: float = 20.0,
    **kwargs: Any,
):
    """
    Call Chat Completions with exponential backoff and clear exceptions.
    IMPORTANT: Pass an already-initialised `OpenAI(api_key=...)` client from app.py.
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )
        except Exception as e:
            last_err = e
            if attempt >= max_retries or not _is_retriable(e):
                # Bubble up the last error with context
                raise RuntimeError(
                    f"OpenAI call failed after {attempt}/{max_retries} attempts: {e}"
                ) from e
            # exponential backoff (jitterless to keep logs predictable)
            delay = min(max_delay, base_delay ** attempt)
            time.sleep(delay)
