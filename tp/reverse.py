"""Обратная инженерия: от сильных движений к сетапам.

1. Разметка. Для каждого бара каждой монеты: началось ли после его закрытия чистое движение
   вверх (цена прошла +k ATR раньше, чем −k/2 ATR, за H баров) или вниз (зеркально).
   k ∈ {2, 4}, H ∈ {6, 24} — 8 меток. Метка — только цель, в признаки не попадает.
2. Признаки состояния на закрытии бара (~35: доходности и положение в диапазоне, волатильность,
   оборот, дельта агрессора, форма бара, OI, фандинг, доли лонгов всех и топов, ликвидации,
   BTC, время суток).
3. Поиск предвестников на ранней части истории (discovery): для каждого признака 7 корзин по
   квантилям; лифт = частота движения в корзине / базовая частота. Затем пары лучших корзин.
   Не «что было перед движениями» (почти всё), а «насколько чаще движение после признака».
   Асимметрия: движение в сторону правила должно быть чаще, чем в обратную (иначе правило
   предсказывает размах, а не направление: у тихой монеты 4 ATR проходятся в обе стороны).
4. Проверка вперёд: лучшие правила становятся сигналами (корзина признака → направление
   движения) и проходят карту геометрий с подменами на проверочном отрезке, который поиск не
   видел (tp.screen, span), затем отложенный период.

Отрезки: правила только на свечах/фандинге/BTC — поиск 2025-10-02 … 2026-04-01, проверка
2026-04-01 … 2026-07-01, отложенный с 2026-07-01; правила с OI, long/short, ликвидациями —
поиск 2026-04-11 … 2026-07-01, проверка 2026-07-01 … 2026-08-10, отложенный с 2026-08-10.

    python -m tp.reverse discover            # results/reverse/rules.json
    python -m tp.reverse validate --top 15   # карта с подменами на проверочном отрезке
    python -m tp.reverse full --rules 5 3 13 # полный перебор выходов с контролем (results/reverse/full)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from . import features as ft
from .data import load
from .fetch import load_universe
from .setups import OOS_CANDLES, OOS_OI, Setup, _sig

OUT = Path(__file__).resolve().parent.parent / "results" / "reverse"
TF = "1h"
KS = (2.0, 4.0)
HS = (6, 24)
BINS = (0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0)
MIN_N_SINGLE, MIN_N_PAIR = 300, 150

WIN = {  # группа признаков -> (поиск, проверка, начало отложенного)
    "A": (("2025-10-02", "2026-04-01"), ("2026-04-01", OOS_CANDLES), OOS_CANDLES),
    "B": (("2026-04-11", "2026-07-01"), ("2026-07-01", OOS_OI), OOS_OI),
}
GROUP_B = ("oi", "ls", "liq", "gap")          # признаки с этими префиксами — группа B
NEEDS = {"oi": "oi", "ls": "ls_global", "gap": "ls_top", "liq": "liq_long", "fund": "fund_last"}


def group_of(name: str) -> str:
    return "B" if name.split("_")[0] in GROUP_B else "A"


# ---------------------------------------------------------------- метки
@njit(cache=True)
def _labels(h, l, c, a, k, H):
    n = len(c)
    up = np.zeros(n, np.int8)
    dn = np.zeros(n, np.int8)
    for i in range(n - 1):
        if not (a[i] > 0):
            continue
        tu, su = c[i] + k * a[i], c[i] - 0.5 * k * a[i]
        td, sd = c[i] - k * a[i], c[i] + 0.5 * k * a[i]
        u_done = False
        d_done = False
        for j in range(i + 1, min(n, i + 1 + H)):
            if not u_done:
                if l[j] <= su:
                    u_done = True
                elif h[j] >= tu:
                    up[i] = 1
                    u_done = True
            if not d_done:
                if h[j] >= sd:
                    d_done = True
                elif l[j] <= td:
                    dn[i] = 1
                    d_done = True
            if u_done and d_done:
                break
        if i + H >= n:                      # путь не полон — метки нет
            up[i] = -1
            dn[i] = -1
    return up, dn


def labels(df: pd.DataFrame, atr: pd.Series) -> pd.DataFrame:
    out = {}
    h, l, c = (np.ascontiguousarray(df[x].to_numpy(float)) for x in ("high", "low", "close"))
    a = np.ascontiguousarray(atr.fillna(0).to_numpy(float))
    for k in KS:
        for H in HS:
            u, d = _labels(h, l, c, a, k, H)
            out[f"up_{k:g}_{H}"] = u
            out[f"dn_{k:g}_{H}"] = d
    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------- признаки
def feats(df: pd.DataFrame, btc: pd.DataFrame | None) -> pd.DataFrame:
    """Состояние на закрытии бара. Всё только по прошлому."""
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    a = ft.atr(df, 14)
    qv = df["qv"].replace(0, np.nan)
    rng = (h - l).replace(0, np.nan)
    X = {}
    for n in (1, 4, 24, 72):
        X[f"r{n}"] = (c - c.shift(n)) / a
    for n in (24, 168):
        hi, lo = h.rolling(n).max(), l.rolling(n).min()
        X[f"pos{n}"] = (c - lo) / (hi - lo).replace(0, np.nan)
    X["dist_hh168"] = (h.shift().rolling(168).max() - c) / a
    X["dist_ll168"] = (c - l.shift().rolling(168).min()) / a
    X["atr_pct"] = a / c * 100
    X["atr_regime"] = a / a.rolling(168).mean()
    X["vsp1"] = ft.spike(df["qv"], 48)
    X["vsp24"] = df["qv"].rolling(24).sum() / df["qv"].shift(24).rolling(168).sum() * 7
    X["dr1"] = df["delta"] / qv
    X["dr6"] = df["delta"].rolling(6).sum() / df["qv"].rolling(6).sum()
    X["dr24"] = df["delta"].rolling(24).sum() / df["qv"].rolling(24).sum()
    X["clv"] = (c - l) / rng
    X["body"] = (c - o) / rng
    X["uwick"] = (h - np.maximum(c, o)) / rng
    X["lwick"] = (np.minimum(c, o) - l) / rng
    X["fund"] = df["fund_last"] * 1e4
    X["fund_z"] = ft.zscore(df["fund_last"], 30 * 24)
    oi = df["oi"]
    for n in (1, 6, 24):
        X[f"oi_ch{n}"] = (oi / oi.shift(n) - 1) * 100
    X["oi_vs_px24"] = X["oi_ch24"] * np.sign(c - c.shift(24))
    X["ls_glob"] = df["ls_global"]
    X["ls_glob_ch24"] = df["ls_global"] - df["ls_global"].shift(24)
    X["gap_top"] = df["ls_top"] - df["ls_global"]
    X["gap_top_ch24"] = X["gap_top"] - X["gap_top"].shift(24)
    X["liq_l1"] = df["liq_long"] / qv * 100
    X["liq_s1"] = df["liq_short"] / qv * 100
    X["liq_l24"] = df["liq_long"].rolling(24).sum() / df["qv"].rolling(24).sum() * 100
    X["liq_s24"] = df["liq_short"].rolling(24).sum() / df["qv"].rolling(24).sum() * 100
    if btc is not None:
        bc = btc["close"].reindex(df.index)
        X["btc_r4"] = (bc / bc.shift(4) - 1) * 100
        X["btc_r24"] = (bc / bc.shift(24) - 1) * 100
        X["rel24"] = (c / c.shift(24) - 1) * 100 - X["btc_r24"]
    X["hour"] = pd.Series(df.index.hour, df.index).astype(float)
    return pd.DataFrame(X, index=df.index).replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------- поиск
def build(syms: list[str]) -> pd.DataFrame:
    btc = load("BTCUSDT", TF)
    parts = []
    for s in syms:
        df = load(s, TF)
        if df is None or len(df) < 500:
            continue
        X = feats(df, btc)
        Y = labels(df, ft.atr(df, 14))
        parts.append(pd.concat([X, Y], axis=1).assign(sym=s))
    return pd.concat(parts)


def _in(D: pd.DataFrame, w) -> pd.DataFrame:
    return D[(D.index >= pd.Timestamp(w[0], tz="UTC")) & (D.index < pd.Timestamp(w[1], tz="UTC"))]


def _singles(W: pd.DataFrame, fs: list[str], labs: list[str], group: str) -> list[dict]:
    out = []
    for f in fs:
        x = W[f]
        ok = x.notna()
        if ok.sum() < 5000:
            continue
        edges = np.unique(np.nanquantile(x[ok], BINS))
        if len(edges) < 4:
            continue
        b = np.digitize(x, edges[1:-1])
        for lab in labs:
            y = W[lab]
            m = ok & (y >= 0)
            base = y[m].mean()
            if not (base > 0):
                continue
            for q in range(len(edges) - 1):
                sel = m & (b == q)
                n = int(sel.sum())
                if n < MIN_N_SINGLE:
                    continue
                rate = y[sel].mean()
                out.append({"group": group, "label": lab, "conds": [[f, float(edges[q]), float(edges[q + 1])]],
                            "n": n, "rate": float(rate), "base": float(base), "lift": float(rate / base),
                            "z": float((rate - base) / np.sqrt(base * (1 - base) / n))})
    return out


def _pairs(W: pd.DataFrame, pool: list[dict], group: str, need_b: bool) -> list[dict]:
    out = []
    for lab in {r["label"] for r in pool}:
        top = sorted([r for r in pool if r["label"] == lab and r["lift"] > 1.2], key=lambda r: -r["z"])[:25]
        y = W[lab]
        base = y[y >= 0].mean()
        for i1 in range(len(top)):
            for i2 in range(i1 + 1, len(top)):
                (f1, a1, b1), (f2, a2, b2) = top[i1]["conds"][0], top[i2]["conds"][0]
                if f1 == f2 or (need_b and group_of(f1) != "B" and group_of(f2) != "B"):
                    continue
                sel = (W[f1] >= a1) & (W[f1] <= b1) & (W[f2] >= a2) & (W[f2] <= b2) & (y >= 0)
                n = int(sel.sum())
                if n < MIN_N_PAIR:
                    continue
                rate = y[sel].mean()
                out.append({"group": group, "label": lab, "conds": [[f1, a1, b1], [f2, a2, b2]], "n": n,
                            "rate": float(rate), "base": float(base), "lift": float(rate / base),
                            "z": float((rate - base) / np.sqrt(base * (1 - base) / n))})
    return out


def discover(D: pd.DataFrame) -> list[dict]:
    """Одиночные корзины и пары. Группа A — только свечи/фандинг/BTC на длинном окне поиска;
    группа B — правила с OI/long/short/ликвидациями на коротком окне (пары могут включать
    признак группы A, но хотя бы один признак — из B)."""
    fa = [c for c in D.columns if not c.startswith(("up_", "dn_")) and c != "sym" and group_of(c) == "A"]
    fb = [c for c in D.columns if not c.startswith(("up_", "dn_")) and c != "sym" and group_of(c) == "B"]
    labs = [c for c in D.columns if c.startswith(("up_", "dn_"))]
    WA, WB = _in(D, WIN["A"][0]), _in(D, WIN["B"][0])
    sa = _singles(WA, fa, labs, "A")
    sb = _singles(WB, fb, labs, "B")
    sa_on_b = _singles(WB, fa, labs, "B")          # только для пар с признаками B
    return sa + sb + _pairs(WA, sa, "A", False) + _pairs(WB, sb + sa_on_b, "B", True)


def _opposite(lab: str) -> str:
    return ("dn" if lab.startswith("up") else "up") + lab[2:]


def check_direction(D: pd.DataFrame, rules: list[dict]) -> list[dict]:
    """Асимметрия: во сколько раз движение в сторону правила чаще, чем в обратную, на отрезке
    поиска. Правило, которое предсказывает только размах (тихая монета — движение на 4 ATR
    в любую сторону), здесь даёт ≈ 1 и направления не несёт."""
    for r in rules:
        W = _in(D, WIN[r["group"]][0])
        sel = pd.Series(True, W.index)
        for f, a, b in r["conds"]:
            sel &= (W[f] >= a) & (W[f] <= b)
        y, yo = W[r["label"]], W[_opposite(r["label"])]
        m = sel & (y >= 0) & (yo >= 0)
        ro = yo[m].mean()
        r["opp_rate"] = float(ro)
        r["asym"] = float(y[m].mean() / ro) if ro > 0 else float("inf")
    return rules


def check_validation(D: pd.DataFrame, rules: list[dict]) -> list[dict]:
    """Лифт правила на проверочном отрезке (метки известны; поиск этот отрезок не видел)."""
    for r in rules:
        W = _in(D, WIN[r["group"]][1])
        y = W[r["label"]]
        sel = y >= 0
        for f, a, b in r["conds"]:
            sel &= (W[f] >= a) & (W[f] <= b)
        n = int(sel.sum())
        base = y[y >= 0].mean()
        r["val_n"] = n
        r["val_lift"] = float(y[sel].mean() / base) if n and base > 0 else None
    return rules


def describe(r: dict) -> str:
    lab = r["label"].split("_")
    side = "рост" if lab[0] == "up" else "падение"
    conds = " и ".join(f"{f} ∈ [{a:.3g}; {b:.3g}]" for f, a, b in r["conds"])
    return f"{conds} → {side} на {lab[1]} ATR за {lab[2]} ч"


# ---------------------------------------------------------------- правило -> сетап
def rule_setup(r: dict, idx: int) -> Setup:
    conds = [tuple(c) for c in r["conds"]]
    long_ = r["label"].startswith("up")
    btc_cache: dict = {}

    def fn(f, cooldown: int):
        from .hyp.common import ref
        X = f.get("revfeats", lambda: feats(f.df, ref(f, "BTCUSDT")))
        m = pd.Series(True, f.df.index)
        for name, a, b in conds:
            m &= (X[name] >= a) & (X[name] <= b)
        none = pd.Series(False, f.df.index)
        return _sig(m, none, cooldown=cooldown) if long_ else _sig(none, m, cooldown=cooldown)

    fn.__doc__ = describe(r)
    needs = tuple(sorted({NEEDS[c[0].split("_")[0]] for c in conds if c[0].split("_")[0] in NEEDS}))
    g = r["group"]
    return Setup(f"rev{idx:02d}", describe(r), fn, TF, needs, WIN[g][2],
                 [{"cooldown": 1}, {"cooldown": 6}, {"cooldown": 24}], doc=describe(r))


def _good(r: dict) -> bool:
    return (r["lift"] >= 1.5 and (r.get("val_lift") or 0) >= 1.3 and r["val_n"] >= 100
            and r.get("asym", 0) >= 1.5)


def main():
    from .placebo import Pool
    from .research import Universe
    from .screen import screen_variant, KINDS, SCREEN_P, edge_type
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["discover", "validate", "full"])
    ap.add_argument("--rules", nargs="*", help="full: номера правил из validated (revNN)")
    ap.add_argument("--universe", default="100")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--runs", type=int, default=30)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.cmd == "discover":
        D = build(load_universe(a.universe))
        rules = discover(D)
        rules = check_direction(D, check_validation(D, rules))
        rules.sort(key=lambda r: -r["z"])
        (OUT / f"rules_{a.universe}.json").write_text(json.dumps(rules, ensure_ascii=False))
        print(f"{len(D)} строк, {len(rules)} правил, {time.time() - t0:.0f}s")
        good = [r for r in rules if _good(r)]
        print(f"устойчивых (лифт поиска ≥ 1.5, проверки ≥ 1.3, асимметрия ≥ 1.5): {len(good)}")
        for r in good[:40]:
            print(f"  z={r['z']:5.1f} лифт {r['lift']:.2f}/{r['val_lift']:.2f} асим {r['asym']:.1f} "
                  f"n={r['n']}/{r['val_n']} база {r['base']:.3f}  {describe(r)}")
        return
    rules = json.loads((OUT / f"rules_{a.universe}.json").read_text())
    if a.cmd == "full":
        # полный перебор выходов с контролем для выбранных правил (results/reverse/full/)
        from . import research
        picks = json.loads((OUT / f"picks_{a.universe}.json").read_text())
        research.RESULTS = OUT / "full"
        uni = Universe(load_universe(a.universe))
        for q in a.rules:
            r = picks[int(q)]
            s_ = rule_setup(r, int(q))
            out = research.run_setup(s_, uni)
            print(f"rev{int(q):02d}: лучшие {len(out['best'])}, лонги {len(out['best_long'])}  {describe(r)}")
            for c in out["best"] + out["best_long"]:
                ctl = c.get("control", {})
                print(f"   OOS {c['oos']['net']:.0f}$ PF {c['oos']['pf']} n={c['oos']['trades']} | IS {c['is']['net']:.0f}$ | "
                      f"{c['label']} | p " + " ".join(f"{k}:{v['oos_p']:.2f}" for k, v in ctl.items())
                      + f" | {c.get('edge_type')}", flush=True)
        return
    good = [r for r in rules if _good(r)]
    # разнообразие: не больше 3 правил на метку; одинаковые условия с тем же направлением
    # (разные k или H) дают одинаковые сигналы — проверяем один раз
    pick, per, seen = [], {}, set()
    for r in good:
        key = (r["label"][:2], tuple(tuple(c) for c in sorted(r["conds"])))
        if key in seen:
            continue
        seen.add(key)
        if per.get(r["label"], 0) < 3:
            pick.append(r)
            per[r["label"]] = per.get(r["label"], 0) + 1
        if len(pick) == a.top:
            break
    (OUT / f"picks_{a.universe}.json").write_text(json.dumps(pick, ensure_ascii=False))
    uni = Universe(load_universe(a.universe))
    res = []
    for q, r in enumerate(pick):
        s = rule_setup(r, q)
        g = r["group"]
        pool = Pool(s, uni, cuts=[WIN[g][1][0]])
        best = None
        for vi in range(3):
            v = screen_variant(pool, vi, a.runs, span=WIN[g][1])
            et = edge_type({k: v[k]["plateau_t"]["p"] for k in KINDS}, SCREEN_P)
            row = {"rule": describe(r), "label": r["label"], "cooldown": s.variants[vi]["cooldown"],
                   "lift": r["lift"], "val_lift": r["val_lift"], "signals_val": v["signals_is"],
                   "plateau_t": v["real"]["plateau_t"], "p": {k: v[k]["plateau_t"]["p"] for k in KINDS},
                   "edge_type": et, "cell": v["real"]["plateau_cell"], "oos": v["oos_at_plateau"]}
            res.append(row)
            print(f"rev{q:02d} cd={row['cooldown']:2d} t={row['plateau_t']:.2f} p={row['p']} {et or '—'} "
                  f"OOS {row['oos']}  {row['rule']}", flush=True)
    (OUT / f"validated_{a.universe}.json").write_text(json.dumps(res, ensure_ascii=False, default=str))
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
