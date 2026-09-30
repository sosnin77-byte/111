"""Выгрузки вне сетки баров: контекст рынка, события ликвидаций с ценой, живые снимки.

    python -m tp.collect context               # макро, календарь, ETF, onchain -> data/raw/context/
    python -m tp.collect events --universe 400 # события ликвидаций от $5k (5 бирж)
    python -m tp.collect snapshots             # бесконечный цикл: снимки раз в 5 / 15 / 60 минут

Истории у снимков нет (предсказанный фандинг, толпа по всем парам, пульс, дельты OI), поэтому их
надо собирать самим с первого дня. События ликвидаций хранятся на сервере около 60 дней:
запускать сборщик хотя бы раз в сутки, иначе дни пропадут (см. docs/reports/api_audit.md).
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .api import ApiError, Client, rows_of
from .fetch import RAW, load_universe

CTX = RAW / "context"
SNAP = RAW / "snap"
EV_EXCHANGES = ("binancef", "bybitf", "okx", "bitget", "gate")
EV_DAYS = 62
ETF_ASSETS = ("BTC", "ETH", "SOL", "XRP", "LTC", "DOGE", "HBAR", "LINK", "AVAX", "DOT", "ADA", "SUI")


def _save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        json.dump(obj, fh)


def context(c: Client):
    """Всё дневное и справочное: несколько десятков запросов."""
    done = 0
    ev = c.get("/macro/events")
    _save_json(CTX / "macro_events.json.gz", ev)
    for e in rows_of(ev):
        try:
            _save_json(CTX / "macro" / f"{e['id']}.json.gz", c.get(f"/macro/events/{e['id']}", limit=1000))
            done += 1
        except ApiError as x:
            print("macro", e.get("id"), x)
    _save_json(CTX / "macro_calendar.json.gz", c.get("/macro/calendar"))
    for a in ETF_ASSETS:
        try:
            _save_json(CTX / "etf" / f"{a}.json.gz", c.get(f"/etf/flows/{a}", limit=1000))
            done += 1
        except ApiError as x:
            print("etf", a, str(x)[:80])
    charts = c.get("/onchain/charts")
    _save_json(CTX / "onchain_charts.json.gz", charts)
    for ch in charts.get("charts", []):
        try:
            _save_json(CTX / "onchain" / f"{ch['chart']}.json.gz", c.get(f"/onchain/{ch['chart']}", limit=1000))
            done += 1
        except ApiError as x:
            print("onchain", ch.get("chart"), str(x)[:80])
    print(f"context: сохранено {done} рядов в {CTX}", flush=True)


def _events(c: Client, ex: str, sym: str, a: int, b: int, out: dict, depth: int = 0):
    r = c.get(f"/liquidations/{ex}/{sym}/events", startTime=a, endTime=b, limit=1000)
    evs = r.get("events") or rows_of(r)
    for e in evs:
        out[(e["time"], e.get("side"), e.get("price"), e.get("quantity"))] = e
    if r.get("truncated") and evs and depth < 40:
        ts = [int(e["time"]) for e in evs]
        lo, hi = min(ts), max(ts)
        if hi < b:
            _events(c, ex, sym, hi, b, out, depth + 1)
        if lo > a:
            _events(c, ex, sym, a, lo, out, depth + 1)


def events_one(c: Client, ex: str, sym: str, end: int) -> int:
    f = RAW / f"liq_ev_{ex}" / f"{sym}.parquet"
    old = pd.read_parquet(f) if f.exists() else None
    start = end - EV_DAYS * 86_400_000
    if old is not None and len(old):
        start = max(start, int(old["time"].max()))
    out: dict = {}
    _events(c, ex, sym, start, end, out)
    if not out:
        return 0
    df = pd.DataFrame(list(out.values()))
    if old is not None:
        df = pd.concat([old, df])
    df = df.drop_duplicates(["time", "side", "price", "quantity"]).sort_values("time")
    f.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(f, index=False)
    return len(out)


def events(c: Client, universe: str, workers: int):
    inst = c.get("/liquidations/instruments").get("exchanges", {})
    syms = set(load_universe(universe))
    jobs = []
    for ex in EV_EXCHANGES:
        have = inst.get(ex) or []
        names = {(i if isinstance(i, str) else i.get("symbol", "")) for i in have}
        jobs += [(ex, s) for s in sorted(syms & names)]
    end = int(time.time() * 1000)
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(events_one, c, ex, s, end): (ex, s) for ex, s in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            ex, s = futs[fu]
            try:
                msg = f"{fu.result()} events"
            except ApiError as e:
                msg = f"ERROR {str(e)[:120]}"
            print(f"[{i}/{len(jobs)} {time.time() - t0:.0f}s calls={c.calls}] {ex} {s}: {msg}", flush=True)


EVERY = {  # имя -> (путь, период в минутах, parquet из списка или сырой json)
    "predicted": ("/funding/predicted", 5, "predicted"),
    "ls_markets": ("/long-short/markets", 5, "markets"),
    "pulse": ("/liquidations/pulse", 15, None),
    "oi_deltas": ("/open-interest/deltas", 60, None),
    "market_overview": ("/market-overview", 60, None),
    "altcoin_season": ("/altcoin-season", 60, None),
    "global_metrics": ("/global-metrics", 60, None),
    "market_indices": ("/market-indices", 60, None),
}


def snapshots(c: Client):
    last: dict[str, float] = {}
    while True:
        now = time.time()
        stamp = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M")
        for name, (path, minutes, key) in EVERY.items():
            if now - last.get(name, 0) < minutes * 60 - 5:
                continue
            last[name] = now
            try:
                r = c.get(path)
            except ApiError as e:
                print(stamp, name, str(e)[:100], flush=True)
                continue
            if key and isinstance(r.get(key), list):
                f = SNAP / name / stamp[:8] / f"{stamp}.parquet"
                f.parent.mkdir(parents=True, exist_ok=True)
                df = pd.DataFrame(r[key])
                df["snap_ms"] = int(now * 1000)
                df.to_parquet(f, index=False)
            else:
                _save_json(SNAP / name / stamp[:8] / f"{stamp}.json.gz", r)
        time.sleep(max(5, 60 - (time.time() - now)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["context", "events", "snapshots"])
    ap.add_argument("--universe", default="400")
    ap.add_argument("--rpm", type=int, default=560)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    c = Client(rpm=a.rpm)
    if a.cmd == "context":
        context(c)
    elif a.cmd == "events":
        events(c, a.universe, a.workers)
    else:
        snapshots(c)


if __name__ == "__main__":
    main()
