"""Выгрузка истории TRADER.PRO по топ-100 фьючерсам Binance USDT-M в data/raw.

    python -m tp.fetch universe            # список топ-100 по обороту за 30 дней
    python -m tp.fetch all                 # все наборы данных по списку
    python -m tp.fetch all --only BTCUSDT  # одна монета

Каждый ряд пишется в data/raw/<набор>/<SYMBOL>.parquet. Повторный запуск
дописывает только недостающий хвост.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .api import INTERVAL_MS, ApiError, Client, fetch_range, rows_of

EX = "binancef"
ROOT = Path(__file__).resolve().parent.parent / "data"
RAW = ROOT / "raw"
UNIVERSE = ROOT / "universe.json"

# Начало истории на тарифе Pro (см. PLAN.md). Раньше сервер всё равно не отдаст.
HISTORY_START = "2025-10-02"

STABLE = {"USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "USDPUSDT", "DAIUSDT", "BUSDUSDT",
          "USDEUSDT", "EURUSDT", "BTCDOMUSDT", "DEFIUSDT", "XUSDUSDT", "USD1USDT",
          # токенизированное золото ведёт себя как золото, не как крипта
          "PAXGUSDT", "XAUTUSDT"}


def _ms(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


# имя набора -> (путь, шаг ряда, limit, доп. параметры, начало)
def datasets(sym: str) -> dict:
    coin = sym.removesuffix("USDT")
    return {
        "candles_1h": (f"/candles/{EX}/{sym}", "1h", 1000, {"interval": "1h"}, HISTORY_START),
        "candles_5m": (f"/candles/{EX}/{sym}", "5m", 1000, {"interval": "5m"}, "2026-04-01"),
        "funding": (f"/funding/{EX}/{sym}", "1h", 1000, {}, HISTORY_START),
        "oi_1h": (f"/open-interest/{EX}/{sym}", "1h", 1000, {"interval": "1h"}, "2026-03-15"),
        "oi_5m": (f"/open-interest/{EX}/{sym}", "5m", 1000, {"interval": "5m"}, "2026-04-01"),
        "ls_global": (f"/long-short/{EX}/{sym}", "5m", 5000,
                      {"ratio_type": "global_account", "period": "5m"}, "2026-04-01"),
        "ls_top": (f"/long-short/{EX}/{sym}", "5m", 5000,
                   {"ratio_type": "top_position", "period": "5m"}, "2026-04-01"),
        # ликвидации по всем биржам, где торгуется монета
        "liq_5m": (f"/liquidations/{sym}/bars", "5m", 5000, {"interval": "5m"}, "2026-06-01"),
    }


def path_for(name: str, sym: str) -> Path:
    return RAW / name / f"{sym}.parquet"


def fetch_one(client: Client, sym: str, name: str, end_ms: int) -> int:
    path, step, limit, params, start = datasets(sym)[name]
    f = path_for(name, sym)
    old = pd.read_parquet(f) if f.exists() else None
    start_ms = _ms(start)
    if old is not None and len(old):
        start_ms = int(old["time"].max()) + 1
    if start_ms > end_ms:
        return 0
    # фандинг идёт раз в 8 (иногда 4 или 1) часов, окно в limit часов заведомо без обрезки
    rows = fetch_range(client, path, start_ms, end_ms, INTERVAL_MS[step], limit, **params)
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    if "time" not in df and "ts" in df:
        df = df.rename(columns={"ts": "time"})
    df["time"] = df["time"].astype("int64")
    if old is not None:
        df = pd.concat([old, df]).drop_duplicates("time", keep="last")
    df = df.sort_values("time").reset_index(drop=True)
    f.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(f, index=False)
    return len(rows)


def build_universe(client: Client, top: int = 100) -> list[str]:
    resp = client.get("/instruments", exchange=EX)
    # ответ: {"count": N, "exchanges": {"binancef": [{"symbol", "status", "has_data", ...}]}}
    items = (resp.get("exchanges") or {}).get(EX) or rows_of(resp) or resp.get("instruments", [])
    syms = set()
    for it in items:
        s = it if isinstance(it, str) else (it.get("symbol") or it.get("id") or "")
        if isinstance(it, dict) and (it.get("status", "active") != "active"
                                     or it.get("has_data") is False):
            continue
        if s.endswith("USDT") and s not in STABLE:
            syms.add(s)
    end = int(time.time() * 1000)
    vols = {}

    def qv(s):
        r = client.get(f"/candles/{EX}/{s}", interval="1d", startTime=end - 31 * 86_400_000,
                       endTime=end, limit=40)
        return sum(float(b.get("quote_volume") or 0) for b in rows_of(r))

    with ThreadPoolExecutor(4) as pool:
        futs = {pool.submit(qv, s): s for s in sorted(syms)}
        for fu in as_completed(futs):
            try:
                vols[futs[fu]] = fu.result()
            except ApiError as e:
                print("skip", futs[fu], e)
    ranked = sorted(vols, key=vols.get, reverse=True)[:top]
    ROOT.mkdir(parents=True, exist_ok=True)
    UNIVERSE.write_text(json.dumps(
        {"asof": pd.Timestamp.now("UTC").isoformat(), "symbols": ranked,
         "quote_volume_30d": {s: vols[s] for s in ranked}}, indent=1))
    return ranked


def load_universe() -> list[str]:
    return json.loads(UNIVERSE.read_text())["symbols"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["universe", "all", "account"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--sets", nargs="*", help="какие наборы качать (по умолчанию все)")
    ap.add_argument("--rpm", type=int, default=580)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    client = Client(rpm=a.rpm)
    if a.cmd == "account":
        print(json.dumps(client.account(), indent=1, ensure_ascii=False))
        return
    if a.cmd == "universe":
        print(build_universe(client))
        return
    syms = a.only or load_universe()
    names = a.sets or list(datasets("BTCUSDT"))
    end = int(time.time() * 1000)
    jobs = [(s, n) for s in syms for n in names]
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as pool:
        futs = {pool.submit(fetch_one, client, s, n, end): (s, n) for s, n in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            s, n = futs[fu]
            try:
                got = fu.result()
                msg = f"{got} rows"
            except ApiError as e:
                msg = f"ERROR {e}"
            print(f"[{i}/{len(jobs)} {time.time() - t0:.0f}s calls={client.calls}] {s} {n}: {msg}",
                  flush=True)


if __name__ == "__main__":
    main()
