"""Проверка файла гипотез перед перебором параметров.

    python -m tp.hyp.check b01                 # все сетапы файла на 25 монетах
    python -m tp.hyp.check b01 --names h003_x  # один сетап
    python -m tp.hyp.check b01 --coins 10 --json

Что проверяется для каждого сетапа и каждого из 3 вариантов:
1. код отрабатывает без исключений на всех монетах, колонки из needs существуют;
2. нет заглядывания вперёд: сигналы (бар, направление, стоп), посчитанные на обрезанной
   истории, совпадают с сигналами на полной истории до точки обрезки (4 точки, 3 монеты);
3. направление ±1, структурный стоп (если задан) по правильную сторону от close сигнального бара;
   у входа лимиткой или стоп-ордером задана цена; лимитка ниже close для лонга (иначе это
   рыночный вход) и стоп-ордер выше close для лонга — нарушения считаются как предупреждение;
4. частота: сигналов на оптимизации и на отложенном периоде, сколько монет дали сигналы,
   плотность (доля баров с сигналом).
Итог OK, если 1–3 без нарушений, у каждого варианта хотя бы MIN_IS сигналов на оптимизации
и плотность не выше MAX_DENSITY.
"""
from __future__ import annotations

import argparse
import importlib
import json
import traceback

import numpy as np
import pandas as pd

from ..data import load
from ..fetch import RAW, load_universe
from ..setups import F

MIN_IS = 40            # минимум сигналов на оптимизации по проверочным монетам (на вариант)
MAX_DENSITY = 0.05     # сигнал не чаще, чем на 5 % баров: иначе это не сетап, а «всегда в рынке»
CUTS = (0.55, 0.75, 0.9, 0.97)
LOOK_COINS = 3

_cache: dict = {}


def _coins(tf: str, n: int) -> list[str]:
    src = "candles_5m" if tf in ("5m", "15m") else "candles_1h"
    out = []
    for s in load_universe():
        if (RAW / src / f"{s}.parquet").exists():
            out.append(s)
        if len(out) == n:
            break
    return out


def _frame(sym: str, tf: str):
    k = (sym, tf)
    if k not in _cache:
        try:
            df = load(sym, tf)
        except Exception:          # файл может переписываться выгрузкой прямо сейчас
            df = None
        _cache[k] = df if df is not None and len(df) >= 300 else None
    return _cache[k]


def _key(sig):
    """(бар, направление) -> (стоп, способ входа, цена входа, ttl, wait)."""
    return {(int(i), int(d)): (s, int(k), p, int(t), int(w)) for i, d, s, k, p, t, w in
            zip(sig.i, sig.dir, sig.stop, sig.kind, sig.px, sig.ttl, sig.wait)}


def _same(a, b):
    return all((np.isnan(x) and np.isnan(y)) if isinstance(x, float) and np.isnan(x)
               else (np.isclose(x, y, rtol=1e-9, atol=0) if isinstance(x, float) else x == y)
               for x, y in zip(a, b))


def check_setup(setup, n_coins: int) -> dict:
    res = {"name": setup.name, "tf": setup.tf, "needs": list(setup.needs), "errors": [],
           "lookahead": [], "bad_stop": 0, "bad_dir": 0, "no_px": 0, "px_marketable": 0,
           "variants": []}
    oos = pd.Timestamp(setup.oos_start, tz="UTC")
    coins = _coins(setup.tf, n_coins)
    for var in setup.variants:
        res["variants"].append({"params": var, "is": 0, "oos": 0, "coins": 0, "bars": 0})
    for ci, sym in enumerate(coins):
        df = _frame(sym, setup.tf)
        if df is None:
            continue
        miss = [c for c in setup.needs if c not in df.columns]
        if miss:
            res["errors"].append(f"нет колонок {miss}")
            break
        f = F(df, sym)
        for vi, var in enumerate(setup.variants):
            try:
                sig = setup.signals(f, **var)
            except Exception:
                res["errors"].append(f"{sym} {var}: {traceback.format_exc(limit=3)}")
                continue
            if len(sig.i) and (sig.i.min() < 0 or sig.i.max() >= len(df)):
                res["errors"].append(f"{sym} {var}: индекс сигнала вне таблицы")
                continue
            res["bad_dir"] += int((~np.isin(sig.dir, (1, -1))).sum())
            c = df["close"].to_numpy()[sig.i]
            st = sig.stop
            bad = ((sig.dir == 1) & (st >= c)) | ((sig.dir == -1) & (st <= c))
            res["bad_stop"] += int(np.nansum(bad & ~np.isnan(st)))
            pend = np.isin(sig.kind, (2, 3))
            res["no_px"] += int((pend & ~(sig.px > 0)).sum())
            wrong = ((sig.kind == 2) & (sig.dir * (sig.px - c) >= 0)) | \
                    ((sig.kind == 3) & (sig.dir * (c - sig.px) >= 0))
            res["px_marketable"] += int((wrong & (sig.px > 0)).sum())
            v = res["variants"][vi]
            t = df.index[sig.i]
            v["is"] += int((t < oos).sum())
            v["oos"] += int((t >= oos).sum())
            v["coins"] += int(len(sig.i) > 0)
            v["bars"] += len(df)
            # заглядывание вперёд: пересчёт на обрезанной истории
            if ci < LOOK_COINS:
                full = _key(sig)
                for q in CUTS:
                    cut = int(len(df) * q)
                    try:
                        s2 = setup.signals(F(df.iloc[:cut].copy(), sym), **var)
                    except Exception:
                        res["errors"].append(f"{sym} {var} cut={cut}: {traceback.format_exc(limit=3)}")
                        continue
                    part = _key(s2)
                    want = {k: s for k, s in full.items() if k[0] < cut}
                    diff = set(want) ^ set(part)
                    stop_diff = [k for k in set(want) & set(part) if not _same(want[k], part[k])]
                    if diff or stop_diff:
                        ex = sorted(diff)[:3] or stop_diff[:3]
                        res["lookahead"].append(
                            f"{sym} {var} cut={q}: {len(diff)} сигналов и {len(stop_diff)} стопов/цен входа "
                            f"расходятся, напр. бары {[df.index[k[0]].isoformat() for k in ex]}")
    res["coins_checked"] = len(coins)
    for v in res["variants"]:
        v["density"] = round((v["is"] + v["oos"]) / max(v["bars"], 1), 4)
    res["ok"] = (not res["errors"] and not res["lookahead"] and not res["bad_stop"]
                 and not res["bad_dir"] and not res["no_px"]
                 and all(v["is"] >= MIN_IS and v["density"] <= MAX_DENSITY for v in res["variants"]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module", help="имя файла гипотез, напр. b01")
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--coins", type=int, default=25)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    mod = importlib.import_module(f"tp.hyp.{a.module}")
    out = []
    for s in mod.SETUPS:
        if a.names and s.name not in a.names:
            continue
        r = check_setup(s, a.coins)
        out.append(r)
        if not a.json:
            flag = "OK  " if r["ok"] else "FAIL"
            cnt = "  ".join(f"{v['params']}: IS {v['is']} OOS {v['oos']} монет {v['coins']} "
                            f"плотн {v['density']:.3f}"
                            for v in r["variants"])
            print(f"{flag} {s.name} [{s.tf}] {cnt}", flush=True)
            for k in ("errors", "lookahead"):
                for m in r[k][:5]:
                    print(f"     {k}: {m}")
            if r["bad_stop"] or r["bad_dir"] or r["no_px"]:
                print(f"     стоп не с той стороны: {r['bad_stop']}, неверное направление: "
                      f"{r['bad_dir']}, лимитка/стоп-ордер без цены: {r['no_px']}")
            if r["px_marketable"]:
                print(f"     предупреждение: {r['px_marketable']} ордеров на вход уже в рынке "
                      f"(исполнятся по open как рыночные)")
    names = [s.name for s in mod.SETUPS]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        print("ДУБЛИ ИМЁН:", dup)
    if a.json:
        print(json.dumps(out, ensure_ascii=False, default=str, indent=1))
    else:
        print(f"итого: {sum(r['ok'] for r in out)}/{len(out)} OK")


if __name__ == "__main__":
    main()
