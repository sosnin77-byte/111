"""Перебор параметров сетапов, разбивка на оптимизацию и отложенный период, отбор.

    python -m tp.research                     # все сетапы по data/universe.json
    python -m tp.research --setups sweep --only BTCUSDT ETHUSDT

Контрольные проверки (tp/placebo.py): для всех вариантов — вход без сетки выходов против
подмены монеты, момента и направления; для отобранных комбинаций — то же с их выходами на
отложенном периоде (control_pass и edge_type: обыгрывают подмену направления и момента
и/или монеты с p <= CONTROL_P);
и вся процедура отбора на подменённых сигналах (null): сколько даёт отбор на шуме.

Результаты: results/<setup>.json (метрики всех комбинаций и 5 лучших) и
results/<setup>_trades.parquet (сделки лучших комбинаций, для кривых капитала).
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load
from .engine import REASONS, Bars, Costs, Exit, simulate
from .setups import BY_NAME, SETUPS, F, Setup

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Критерии отбора (PLAN.md)
MIN_TRADES = 50
MIN_PF = 1.3
MAX_DD_SHARE = 1 / 3        # просадка не больше трети прибыли
IS_TOP = 20                 # сколько лучших по оптимизации проверяем на отложенном периоде

# Контрольные проверки (tp/placebo.py)
CONTROL_RUNS = 100          # подмен на каждую отобранную комбинацию и каждый вид подмены
ENTRY_RUNS = 30             # подмен для контроля входа без выходов (все варианты всех сетапов)
NULL_RUNS = 20              # прогонов всей процедуры отбора на подменённых сигналах
CONTROL_P = 0.10            # строгий порог контроля на отложенном периоде (тип — screen.edge_type)

TIME_EXITS = {"5m": (24, 72, 144), "15m": (16, 48, 96), "1h": (12, 24, 48), "4h": (6, 12, 24)}

SCHEMES = {
    "tp1.5": dict(tp_r=(1.5, 0, 0), tp_frac=(1, 0, 0)),
    "tp3": dict(tp_r=(3, 0, 0), tp_frac=(1, 0, 0)),
    "1R½+BE+2.5R": dict(tp_r=(1, 2.5, 0), tp_frac=(0.5, 0.5, 0), be=True),
    "⅓:0.5/1.5/3R+BE": dict(tp_r=(0.5, 1.5, 3), tp_frac=(1 / 3, 1 / 3, 1 / 3), be=True),
    "1R½+BE+trail2.5": dict(tp_r=(1, 0, 0), tp_frac=(0.5, 0, 0), be=True, trail_atr=2.5),
}
STOP_HUNTER_NATIVE = dict(tp_r=(0.2, 0.7, 1.5), tp_frac=(1 / 3, 1 / 3, 1 / 3), be=True)


def exit_grid(setup: Setup) -> list[Exit]:
    """4 стопа × 5 схем выхода × 3 выхода по времени = 60 (плюс родная схема Stop Hunter)."""
    if setup.struct_stop:
        stops = [dict(sl_atr=0, sl_buf=b) for b in (0.1, 0.3, 0.6)] + [dict(sl_atr=1.5)]
    else:
        stops = [dict(sl_atr=v) for v in (1.0, 1.5, 2.0, 3.0)]
    schemes = list(SCHEMES.values())
    if setup.name == "stop_hunter":
        schemes = schemes + [STOP_HUNTER_NATIVE]
    out = []
    for st, sc, te in itertools.product(stops, schemes, TIME_EXITS[setup.tf]):
        out.append(Exit(**st, **sc, time_exit=te))
    return out


def metrics(tr: pd.DataFrame, n_coins: int | None = None) -> dict:
    if tr.empty:
        return {"trades": 0, "net": 0.0, "pf": 0.0, "win": 0.0, "max_dd": 0.0, "net_dd": 0.0}
    tr = tr.sort_values("exit_time")
    p = tr["pnl"].to_numpy()
    eq = np.cumsum(p)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:] - eq))
    gw, gl = p[p > 0].sum(), -p[p < 0].sum()
    net = float(p.sum())
    by_coin = tr.groupby("coin")["pnl"].sum()
    return {
        "trades": int(len(p)),
        "net": round(net, 2),
        "pf": round(float(gw / gl), 3) if gl > 0 else float("inf"),
        "win": round(float((p > 0).mean()), 3),
        "avg": round(float(p.mean()), 3),
        "max_dd": round(dd, 2),
        "net_dd": round(net / dd, 2) if dd > 0 else float("inf"),
        "coins": int(by_coin.size),
        "coins_pos": round(float((by_coin > 0).mean()), 3),
        "long_trades": int((tr["dir"] == 1).sum()),
        "long_net": round(float(tr.loc[tr["dir"] == 1, "pnl"].sum()), 2),
        "short_net": round(float(tr.loc[tr["dir"] == -1, "pnl"].sum()), 2),
    }


def passes(m: dict) -> bool:
    return (m["trades"] >= MIN_TRADES and m["pf"] > MIN_PF and m["net"] > 0
            and m["max_dd"] <= m["net"] * MAX_DD_SHARE)


def _score_is(m: dict) -> float:
    """Ранжирование на оптимизации: прибыль с поправкой на просадку, минимум сделок."""
    if m["trades"] < MIN_TRADES or m["net"] <= 0:
        return -np.inf
    return m["net"] * min(m["net_dd"], 10) / 10 * min(m["pf"], 3)


def select(combos: list[dict], key_is: str, key_oos: str) -> list[dict]:
    ranked = sorted(combos, key=lambda c: _score_is(c[key_is]), reverse=True)[:IS_TOP]
    ranked = [c for c in ranked if np.isfinite(_score_is(c[key_is]))]
    good = [c for c in ranked if passes(c[key_oos])]
    return sorted(good, key=lambda c: c[key_oos]["net"], reverse=True)[:5]


class Universe:
    """Кэш таблиц монет по ТФ."""

    def __init__(self, syms, raw=None):
        self.syms = syms
        self.raw = raw
        self._c = {}

    def frames(self, tf):
        for s in self.syms:
            k = (s, tf)
            if k not in self._c:
                df = load(s, tf) if self.raw is None else load(s, tf, self.raw)
                self._c[k] = None if df is None or len(df) < 300 else F(df, s)
            if self._c[k] is not None:
                yield s, self._c[k]


def run_control(setup, uni, exits, best, best_long, costs, log) -> dict:
    """Контрольные проверки: вход без выходов для всех вариантов, полный контроль отобранных
    комбинаций (подмена монеты, момента, направления), отбор на шуме."""
    from .placebo import KINDS, Pool, control, entry_control, null_selection
    from .screen import edge_type
    t0 = time.time()
    pool = Pool(setup, uni, costs)
    out = {"entry": [entry_control(pool, vi, ENTRY_RUNS) for vi in range(len(setup.variants))]}
    for lst, key in ((best, "oos"), (best_long, "oos_long")):
        for c in lst:
            ex = exits[c["exit"]]
            ctl = control(pool, c["variant"], ex, CONTROL_RUNS)
            c["control"] = {k: {"oos_p": ctl[k]["oos"]["p"], "oos_mean": ctl[k]["oos"]["mean"],
                                "oos_p95": ctl[k]["oos"]["p95"], "all_p": ctl[k]["all"]["p"],
                                "forward_p": ctl[k]["forward_p"]} for k in KINDS}
            c["edge_type"] = edge_type({k: c["control"][k]["oos_p"] for k in KINDS}, CONTROL_P)
            c["control_pass"] = c["edge_type"] is not None
    if best or best_long:
        nul = null_selection(pool, exits, "time", NULL_RUNS,
                             select_fn=lambda cs: select(cs, "is", "oos"))
        real_best = max((_score_is(c["is"]) for c in best), default=0.0)
        nul["real_best_is_score"] = real_best
        nul["p_is_score"] = float((1 + sum(x >= real_best for x in nul["best_is_score"]))
                                  / (1 + len(nul["best_is_score"])))
        nul["share_noise_runs_with_selection"] = float(np.mean([n > 0 for n in nul["n_selected"]]))
        out["null"] = nul
    log(f"  контроль: {time.time() - t0:.0f}s")
    return out


def run_setup(setup: Setup, uni: Universe, costs: Costs = Costs(), log=print,
              control_on: bool = True) -> dict:
    exits = exit_grid(setup)
    oos = np.datetime64(pd.Timestamp(setup.oos_start, tz="UTC").tz_convert(None))
    trades = {}       # (variant, exit) -> список DataFrame
    t0 = time.time()
    n_coins = 0
    for sym, f in uni.frames(setup.tf):
        n_coins += 1
        bars = Bars.from_frame(f.df, f.atr)
        for vi, var in enumerate(setup.variants):
            sig = setup.signals(f, **var)
            if not len(sig.i):
                continue
            for ei, ex in enumerate(exits):
                r = simulate(bars, sig, ex, costs)
                if len(r["pnl"]):
                    r["coin"] = np.full(len(r["pnl"]), sym)
                    trades.setdefault((vi, ei), []).append(pd.DataFrame(r))
    log(f"{setup.name}: {n_coins} монет, {time.time() - t0:.0f}s")
    combos = []
    for vi, var in enumerate(setup.variants):
        for ei, ex in enumerate(exits):
            parts = trades.get((vi, ei))
            tr = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
                columns=["entry_time", "exit_time", "dir", "pnl", "risk", "reason", "coin"])
            tr["entry_time"] = pd.to_datetime(tr["entry_time"])
            is_ = tr[tr["entry_time"].values < oos]
            oo = tr[tr["entry_time"].values >= oos]
            lg = tr[tr["dir"] == 1]
            combos.append({
                "variant": vi, "exit": ei, "signal": var, "exit_params": asdict(ex),
                "label": f"{var} | {ex.label()}",
                "is": metrics(is_), "oos": metrics(oo), "all": metrics(tr),
                "is_long": metrics(lg[lg["entry_time"].values < oos]),
                "oos_long": metrics(lg[lg["entry_time"].values >= oos]),
                "exit_reasons": {REASONS[k]: int(v) for k, v in
                                 tr["reason"].value_counts().items()} if len(tr) else {},
            })
    best = select(combos, "is", "oos")
    best_long = select(combos, "is_long", "oos_long")
    control = run_control(setup, uni, exits, best, best_long, costs, log) if control_on else {}
    # сделки лучших комбинаций для кривых капитала
    keep = {(c["variant"], c["exit"]) for c in best + best_long}
    if not keep:     # ничего не прошло — сохраняем лучшие по оптимизации для разбора
        keep = {(c["variant"], c["exit"]) for c in
                sorted(combos, key=lambda c: _score_is(c["is"]), reverse=True)[:5]}
    tr_keep = [pd.concat(trades[k]).assign(variant=k[0], exit=k[1]) for k in keep if k in trades]
    RESULTS.mkdir(exist_ok=True)
    if tr_keep:
        pd.concat(tr_keep, ignore_index=True).to_parquet(RESULTS / f"{setup.name}_trades.parquet")
    out = {
        "setup": setup.name, "title": setup.title, "doc": setup.doc, "tf": setup.tf,
        "oos_start": setup.oos_start, "coins": n_coins, "combos": len(combos),
        "criteria": {"min_trades": MIN_TRADES, "min_pf": MIN_PF, "max_dd_share": MAX_DD_SHARE},
        "best": best, "best_long": best_long, "control": control, "all_combos": combos,
    }
    (RESULTS / f"{setup.name}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return out


def signal_counts(setup: Setup, uni: Universe) -> list[dict]:
    """Сколько сигналов даёт каждый вариант (до сетки выходов): на оптимизации, на отложенном
    периоде, число монет с сигналами. Нужно, чтобы пороги калибровать по частоте, не по прибыли."""
    oos = pd.Timestamp(setup.oos_start, tz="UTC")
    out = [{"variant": v, "is": 0, "oos": 0, "coins": 0} for v in setup.variants]
    for _, f in uni.frames(setup.tf):
        for vi, var in enumerate(setup.variants):
            sig = setup.signals(f, **var)
            if len(sig.i):
                t = f.df.index[sig.i]
                out[vi]["is"] += int((t < oos).sum())
                out[vi]["oos"] += int((t >= oos).sum())
                out[vi]["coins"] += 1
    return out


def main():
    from .fetch import load_universe
    ap = argparse.ArgumentParser()
    ap.add_argument("--setups", nargs="*")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--universe", default="100",
                    help='монеты: "100" — топ-100, "400" — топ-400, "101-400" — средние и мелкие')
    ap.add_argument("--counts", action="store_true", help="только посчитать сигналы")
    ap.add_argument("--no-control", action="store_true", help="без контрольных проверок")
    ap.add_argument("--pool", choices=["main", "hyp", "all"], default="main",
                    help="main — 11 сетапов tp/setups.py, hyp — 100 гипотез tp/hyp, all — все")
    a = ap.parse_args()
    global BY_NAME, SETUPS
    if a.pool != "main":
        from . import hyp
        SETUPS = (SETUPS if a.pool == "all" else []) + hyp.HYP
        BY_NAME = {**BY_NAME, **hyp.BY_NAME}
    uni = Universe(a.only or load_universe(a.universe))
    if a.counts:
        for s in ([BY_NAME[n] for n in a.setups] if a.setups else SETUPS):
            for c in signal_counts(s, uni):
                print(f"{s.name:17s} {str(c['variant']):36s} IS {c['is']:6d}  "
                      f"OOS {c['oos']:5d}  монет {c['coins']}", flush=True)
        return
    for s in ([BY_NAME[n] for n in a.setups] if a.setups else SETUPS):
        r = run_setup(s, uni, control_on=not a.no_control)
        print(f"  лучшие: {len(r['best'])}, лучшие лонги: {len(r['best_long'])}")
        for c in r["best"]:
            ctl = c.get("control", {})
            ps = " ".join(f"{k}:p={v['oos_p']:.2f}" for k, v in ctl.items())
            print(f"   OOS {c['oos']} | IS {c['is']['net']} | {c['label']} | контроль {ps}")


if __name__ == "__main__":
    main()
