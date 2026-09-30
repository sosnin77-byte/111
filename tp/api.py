"""REST-клиент TRADER.PRO Data API с ограничением скорости и постраничной выгрузкой.

Ключ берётся из переменной окружения TRADERPRO_API_KEY и передаётся в
заголовке X-TP-API-Key. Лимит тарифа Pro 600 запросов в минуту на аккаунт,
около половины занимает сервер пользователя, поэтому по умолчанию берём 240.
"""
from __future__ import annotations

import os
import threading
import time

import requests

BASE = "https://trader-pro.org/data/v1"

INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


class ApiError(RuntimeError):
    pass


class RateLimiter:
    """Равномерный темп: не больше rpm запросов в минуту на все потоки."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            at = max(now, self.next_at)
            self.next_at = at + self.interval
        delay = at - time.monotonic()
        if delay > 0:
            time.sleep(delay)


class Client:
    def __init__(self, key: str | None = None, rpm: int = 240, timeout: int = 60):
        key = (key or os.environ.get("TRADERPRO_API_KEY", "")).strip()
        if not key:
            raise ApiError("TRADERPRO_API_KEY не задан")
        self.session = requests.Session()
        self.session.headers["X-TP-API-Key"] = key
        self.limiter = RateLimiter(rpm)
        self.timeout = timeout
        self.calls = 0

    def get(self, path: str, **params) -> dict:
        params = {k: v for k, v in params.items() if v is not None}
        for attempt in range(8):
            self.limiter.wait()
            try:
                r = self.session.get(f"{BASE}{path}", params=params, timeout=self.timeout)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            self.calls += 1
            if r.status_code == 200:
                return r.json()
            if r.status_code in (202, 429, 500, 502, 503, 504, 507):
                time.sleep(float(r.headers.get("Retry-After", 2 ** min(attempt, 5))))
                continue
            raise ApiError(f"{r.status_code} {path} {params}: {r.text[:300]}")
        raise ApiError(f"не удалось получить {path} {params}")

    def account(self) -> dict:
        return self.get("/account")


def rows_of(resp: dict) -> list[dict]:
    """Список точек из ответа: bars / rates / points / data — берём первый список словарей."""
    for key in ("bars", "rates", "points", "data", "items", "candles"):
        v = resp.get(key)
        if isinstance(v, list):
            return v
    for v in resp.values():
        if isinstance(v, list) and (not v or isinstance(v[0], dict)):
            return v
    return []


def fetch_range(client: Client, path: str, start_ms: int, end_ms: int, step_ms: int,
                limit: int, **params) -> list[dict]:
    """Выгружает ряд окнами по limit точек.

    Окна считаются от шага ряда. Если сервер пометил ответ как обрезанный,
    недостающие края окна дозапрашиваются, в какую бы сторону он ни резал.
    """
    out: dict[int, dict] = {}
    span = step_ms * limit
    stack = []
    a = start_ms
    while a <= end_ms:
        b = min(a + span - 1, end_ms)
        stack.append((a, b))
        a = b + 1
    stack.reverse()
    while stack:
        a, b = stack.pop()
        resp = client.get(path, startTime=a, endTime=b, limit=limit, **params)
        rows = rows_of(resp)
        times = []
        for row in rows:
            t = int(row.get("time", row.get("ts", 0)))
            if a <= t <= b:
                out[t] = row
                times.append(t)
        if resp.get("truncated") and times:
            lo, hi = min(times), max(times)
            if hi + step_ms <= b:
                stack.append((hi + 1, b))
            if lo - step_ms >= a:
                stack.append((a, lo - 1))
    return [out[t] for t in sorted(out)]
