# utils/utils_retries.py
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from openai import OpenAI

client = OpenAI()  # you'll still pass api_key in your main app (OpenAI(api_key=...))

class GPTRetryable(Exception):
    """Used to mark transient errors for retry."""
    pass

@retry(
    reraise=True,
    stop=stop_after_attempt(6),                      # up to 6 tries
    wait=wait_exponential_jitter(initial=1, max=30), # 1s → 30s with jitter
    retry=retry_if_exception_type(GPTRetryable)
)
def safe_chat_completion(*, model, messages, temperature=0, top_p=0, timeout=90):
    """OpenAI chat completion with timeouts + robust retries for transient failures."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        msg = str(e).lower()
        transient = any(k in msg for k in [
            "429", "rate", "timeout", "temporar", "overload", "bad gateway", "service", "5"
        ])
        if transient:
            raise GPTRetryable(e)
        raise
