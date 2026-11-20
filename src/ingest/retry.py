"""Exponential backoff for external fetches.

Every call to a remote host can fail for reasons that fix themselves: a dropped
connection, a timeout, a 503 while the server is busy. Those are retried with a
growing delay, so a brief outage costs seconds instead of the whole run.

Only transient failures are retried. A 404 or a 403 is a real answer from the
server and fails immediately, rather than burning five attempts over roughly half
a minute to learn the same thing.
"""

from __future__ import annotations

import http.client
import logging
import random
import socket
import time
import urllib.error
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_ATTEMPTS = 5
DEFAULT_BASE_DELAY_S = 2.0
DEFAULT_MAX_DELAY_S = 60.0

#: HTTP statuses worth retrying: server-side faults and explicit rate limiting.
TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class PermanentIngestError(Exception):
    """Failure that will not be fixed by trying again."""


def _status_code(exc: BaseException) -> int | None:
    """HTTP status carried by `exc`, if any.

    Reads urllib's `HTTPError.code`, and the `response.status_code` attribute
    that HTTP client libraries such as requests attach, without depending on them.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code
    response = getattr(exc, "response", None)
    if response is None:
        return None
    return getattr(response, "status_code", None)


def is_transient(exc: BaseException) -> bool:
    """True when retrying `exc` could plausibly succeed."""
    if isinstance(exc, PermanentIngestError):
        return False

    status = _status_code(exc)
    if status is not None:
        return status in TRANSIENT_STATUS

    # No status attached: fall back on the exception family. Connection and
    # timeout errors are transient by nature, and so is a body cut short.
    if isinstance(exc, (TimeoutError, ConnectionError, http.client.IncompleteRead)):
        return True

    # urllib wraps socket-level failures (refused, reset, DNS hiccup) in a URLError.
    if isinstance(exc, urllib.error.URLError):
        return isinstance(exc.reason, (TimeoutError, ConnectionError, socket.gaierror))

    name = type(exc).__name__.lower()
    return any(token in name for token in ("timeout", "connection", "chunked"))


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    description: str,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay_s: float = DEFAULT_BASE_DELAY_S,
    max_delay_s: float = DEFAULT_MAX_DELAY_S,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
) -> T:
    """Call `fn`, retrying transient failures with jittered exponential backoff.

    `sleep` and `rng` are injectable so that tests can assert the delay
    schedule without actually waiting.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    last_exc: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - classified immediately below
            last_exc = exc

            if not is_transient(exc):
                logger.error(
                    "fetch_failed_permanent description=%r attempt=%d error=%s",
                    description,
                    attempt,
                    exc,
                )
                raise

            if attempt == attempts:
                logger.error(
                    "fetch_failed_exhausted description=%r attempts=%d error=%s",
                    description,
                    attempts,
                    exc,
                )
                break

            delay = min(base_delay_s * (2 ** (attempt - 1)), max_delay_s)
            delay += rng() * base_delay_s  # full-width jitter on the base unit
            logger.warning(
                "fetch_retry description=%r attempt=%d/%d delay_s=%.2f error=%s",
                description,
                attempt,
                attempts,
                delay,
                exc,
            )
            sleep(delay)

    assert last_exc is not None  # unreachable: loop always sets it before break
    raise last_exc
