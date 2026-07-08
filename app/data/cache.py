"""Tiny thread-safe in-memory TTL cache so we stay polite to upstream APIs."""

import threading
import time


class TTLCache:
    def __init__(self):
        self._data = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            hit = self._data.get(key)
            if hit is None:
                return None
            expires, value = hit
            if time.time() > expires:
                del self._data[key]
                return None
            return value

    def set(self, key, value, ttl):
        with self._lock:
            self._data[key] = (time.time() + ttl, value)


cache = TTLCache()


def cached(key, ttl, fetch):
    """Return the cached value for `key`, calling `fetch()` on a miss.

    A `fetch` returning None is treated as a failure and is not cached,
    so transient upstream errors retry on the next request.
    """
    value = cache.get(key)
    if value is None:
        value = fetch()
        if value is not None:
            cache.set(key, value, ttl)
    return value
