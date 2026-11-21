"""HTTP client with retries, exponential backoff and per-host rate limiting."""
from __future__ import annotations

import threading
import time
from urllib.parse import urlparse

import httpx

from quantai.logging_setup import get_logger, redact

log = get_logger(__name__)


class ProviderError(RuntimeError):
    pass


class RateLimiter:
    def __init__(self, min_interval_s: float = 1.0):
        self.min_interval_s = min_interval_s
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str) -> None:
        with self._lock:
            now = time.monotonic()
            last = self._last.get(host, 0.0)
            delay = self.min_interval_s - (now - last)
            if delay > 0:
                time.sleep(delay)
            self._last[host] = time.monotonic()


_limiter = RateLimiter(1.0)


def get(url: str, params: dict | None = None, retries: int = 3, timeout: float = 15.0,
        headers: dict | None = None, backoff: float = 1.5) -> httpx.Response:
    host = urlparse(url).netloc
    hdrs = {"User-Agent": "quantai-research/0.1"}
    hdrs.update(headers or {})
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        _limiter.wait(host)
        try:
            r = httpx.get(url, params=params, timeout=timeout, headers=hdrs, follow_redirects=True)
            if r.status_code == 429 or r.status_code >= 500:
                raise ProviderError(f"HTTP {r.status_code} from {host}")
            r.raise_for_status()
            return r
        except (httpx.HTTPError, ProviderError) as e:
            last_exc = e
            log.warning("http retry", extra={"ctx": {"host": host, "attempt": attempt, "err": redact(str(e))}})
            if attempt < retries:
                time.sleep(backoff ** attempt)
    raise ProviderError(redact(f"request to {host} failed after {retries + 1} attempts: {last_exc}"))
