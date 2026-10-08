"""Stage 1 - ingest: fetch companyfacts from the SEC and cache it on disk.

One request per company returns every XBRL fact it has ever reported. The raw
response is stored gzipped (about a tenth of the JSON size) and reused on
later runs unless ``refresh`` is set.
"""

from __future__ import annotations

import gzip
import json
import threading
import time
from pathlib import Path
from typing import Callable, Iterable

import requests

from . import config


class RateLimiter:
    """Spaces calls so the SEC's requests-per-second cap is never exceeded."""

    def __init__(self, per_second: float):
        self._interval = 1.0 / per_second
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            if delay > 0:
                time.sleep(delay)
            self._next = max(now, self._next) + self._interval


def raw_path(cik: int) -> Path:
    return config.RAW_DIR / f"CIK{cik:010d}.json.gz"


def load_raw(cik: int) -> dict:
    with gzip.open(raw_path(cik), "rt", encoding="utf-8") as fh:
        return json.load(fh)


def fetch_companyfacts(
    cik: int,
    session: requests.Session,
    limiter: RateLimiter,
    max_retries: int = 5,
) -> bytes | None:
    """Return the raw JSON bytes, or None when the SEC has no XBRL for the CIK."""
    url = config.SEC_COMPANYFACTS_URL.format(cik=cik)
    for attempt in range(max_retries):
        limiter.wait()
        try:
            resp = session.get(url, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 200:
            return resp.content
        if resp.status_code == 404:
            return None
        # 403/429 mean we are being throttled; 5xx is transient. Back off.
        time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError(f"Could not fetch companyfacts for CIK {cik} after {max_retries} attempts")


def ingest(
    ciks: Iterable[int],
    refresh: bool = False,
    progress: Callable[[int, int, int, str], None] | None = None,
) -> dict[str, list[int]]:
    """Download companyfacts for every CIK that is not already cached."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    ciks = list(dict.fromkeys(int(c) for c in ciks))
    session = requests.Session()
    session.headers.update({
        "User-Agent": config.sec_user_agent(),
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    })
    limiter = RateLimiter(config.SEC_MAX_REQUESTS_PER_SECOND)
    result: dict[str, list[int]] = {"cached": [], "downloaded": [], "missing": [], "failed": []}
    for i, cik in enumerate(ciks, 1):
        path = raw_path(cik)
        if path.exists() and not refresh:
            status = "cached"
        else:
            try:
                body = fetch_companyfacts(cik, session, limiter)
            except RuntimeError:
                body = None
                status = "failed"
            else:
                status = "downloaded" if body is not None else "missing"
            if body is not None:
                json.loads(body)  # never cache a truncated or non-JSON response
                tmp = path.with_suffix(".tmp")
                with gzip.open(tmp, "wb", compresslevel=6) as fh:
                    fh.write(body)
                tmp.replace(path)
        result[status].append(cik)
        if progress is not None:
            progress(i, len(ciks), cik, status)
    return result
