"""«Супертрейдер»: слепой разбор сигналов трейдером-агентом и оцифровка его опыта.

Идея: человек смотрит на график в момент сигнала (тренд старшего ТФ, где цена в диапазоне,
что делает BTC, OI, фандинг, ликвидации, реакция цены) и решает — брать или нет, как войти,
где стоп, как вести. Мы проверяем, добавляет ли такой разбор что-то к «брать все сигналы»,
и если да — вытаскиваем из его причин правила-фильтры и проверяем их системно.

1. sample  — случайные сигналы периода оптимизации; снимок рынка ДО сигнала в текстовом виде:
             монета, даты и цены скрыты (цена = 100 на закрытии сигнального бара), чтобы модель
             не могла вспомнить историю.
2. агент   — по снимкам принимает решения (вход, стоп, тейки, ведение, причины), будущего не видит.
3. evaluate — решения прогоняются через движок на настоящем будущем и сравниваются с базой:
             все сигналы с лучшей простой геометрией первичного отбора (плато карты).

    python -m tp.trader sample                     # results/trader/events.json + снимки
    python -m tp.trader evaluate decisions.json    # results/trader/evaluation.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as ft
from .data import load
from .engine import E_LIMIT, E_MARKET, E_STOP, Costs, Exit, Signals, simulate
from .fetch import load_universe

OUT = Path(__file__).resolve().parent.parent / "results" / "trader"
SCREEN = Path(__file__).resolve().parent.parent / "results" / "screen" / "100"

# сетапы, прошедшие первичный отбор, и вариант сигнала для разбора
PICK = {"liq_flush": 2, "capitulation": 0, "stop_hunter": 1, "absorption": 2}
N_PER_SETUP = 40
BARS = {"15m": 64, "1h": 48, "4h": 40, "5m": 72}


def _cohort(sym: str) -> str:
    try:
        rank = load_universe("400").index(sym) + 1
    except (FileNotFoundError, ValueError):
        return "неизвестно"
    return ("крупная (топ-50)" if rank <= 50 else "крупная (51–100)" if rank <= 100
            else "средняя (101–200)" if rank <= 200 else "мелкая (201–400)")


def snapshot(df: pd.DataFrame, i: int, d: int, tf: str, btc: pd.DataFrame | None,
             setup, sym: str) -> str:
    """Текстовый снимок рынка на закрытии бара i. Ничего после бара i."""
    past = df.iloc[: i + 1]
    c0 = past["close"].iloc[-1]
    n = BARS.get(tf, 48)
    w = past.iloc[-n:]
    norm = lambda x: x / c0 * 100
    base_qv = past["qv"].iloc[-n - 96:-n].median() if len(past) > n + 96 else past["qv"].median()
    rows = []
    for k, (t, r) in enumerate(w.iterrows()):
        oi_chg = (r["oi"] / w["oi"].shift().iloc[k] - 1) * 100 if k and pd.notna(r["oi"]) else np.nan
        liq = ""
        if pd.notna(r.get("liq_long")) and r["qv"] > 0:
            liq = f"{r['liq_long'] / r['qv'] * 100:.2f}/{r['liq_short'] / r['qv'] * 100:.2f}"
        rows.append(f"{k - len(w) + 1:>4} {norm(r['open']):7.2f} {norm(r['high']):7.2f} "
                    f"{norm(r['low']):7.2f} {norm(r['close']):7.2f} {r['qv'] / base_qv:5.1f} "
                    f"{r['delta'] / r['qv'] * 100 if r['qv'] else 0:5.0f} "
                    f"{'' if np.isnan(oi_chg) else f'{oi_chg:6.2f}':>6} {liq}")
    a = ft.atr(past, 14).iloc[-1]
    # тренд 4h по закрытым 4h-барам (ft.htf переносит значение на конец 4h-бара)
    e50 = ft.htf(past, "4h", lambda x: ft.ema(x["close"], 50)).iloc[-1]
    e200 = ft.htf(past, "4h", lambda x: ft.ema(x["close"], 200)).iloc[-1]
    bars_day = int(pd.Timedelta("1D") / (past.index[1] - past.index[0]))
    ch = lambda b: (c0 / past["close"].iloc[-1 - b] - 1) * 100 if len(past) > b else np.nan
    rng = lambda b: ((c0 - past["low"].iloc[-b:].min()) /
                     (past["high"].iloc[-b:].max() - past["low"].iloc[-b:].min()) * 100)
    lines = [
        f"Сигнал сетапа: {'ЛОНГ' if d == 1 else 'ШОРТ'}. Сетап: {setup.title}. {setup.doc}",
        f"ТФ {tf}. Монета: {_cohort(sym)} по обороту. Цена на закрытии сигнального бара = 100.",
        f"ATR(14) = {a / c0 * 100:.2f} % цены. Изменение цены: 1 день {ch(bars_day):+.1f} %, "
        f"7 дней {ch(7 * bars_day):+.1f} %, 30 дней {ch(30 * bars_day):+.1f} %.",
        f"Положение в диапазоне: 7 дней {rng(7 * bars_day):.0f} %, 30 дней {rng(30 * bars_day):.0f} % "
        f"(0 — у минимума, 100 — у максимума).",
        f"Тренд 4h: EMA50 {'выше' if e50 > e200 else 'ниже'} EMA200, цена "
        f"{'выше' if c0 > e50 else 'ниже'} EMA50 ({norm(e50):.1f}), EMA200 = {norm(e200):.1f}.",
    ]
    if "fund_last" in past and pd.notna(past["fund_last"].iloc[-1]):
        fl = past["fund_last"].dropna()
        pct = (fl.iloc[-30 * bars_day:] < fl.iloc[-1]).mean() * 100
        lines.append(f"Фандинг: последний {fl.iloc[-1] * 100:.4f} % за выплату, выше, чем в {pct:.0f} % "
                     f"случаев за 30 дней.")
    if pd.notna(past["ls_global"].iloc[-1]):
        lg, lt = past["ls_global"], past["ls_top"]
        lines.append(f"Доля лонгов: все аккаунты {lg.iloc[-1] * 100:.0f} % (сутки назад "
                     f"{lg.iloc[-1 - bars_day] * 100:.0f} %), топ-трейдеры {lt.iloc[-1] * 100:.0f} % "
                     f"(сутки назад {lt.iloc[-1 - bars_day] * 100:.0f} %).")
    if pd.notna(past["oi"].iloc[-1]):
        lines.append(f"OI: за сутки {(past['oi'].iloc[-1] / past['oi'].iloc[-1 - bars_day] - 1) * 100:+.1f} %, "
                     f"за 7 дней {(past['oi'].iloc[-1] / past['oi'].iloc[-1 - 7 * bars_day] - 1) * 100:+.1f} %.")
    if btc is not None and sym != "BTCUSDT":
        b = btc.loc[: past.index[-1], "close"].dropna()
        if len(b) > 7 * bars_day:
            bc = lambda k: (b.iloc[-1] / b.iloc[-1 - k] - 1) * 100
            lines.append(f"BTC: 4 бара {bc(4):+.1f} %, 1 день {bc(bars_day):+.1f} %, "
                         f"7 дней {bc(7 * bars_day):+.1f} %.")
    head = "   # open    high    low     close   V×  Δ%    OI%   ликв L/S % оборота"
    return "\n".join(lines) + "\n" + head + "\n" + "\n".join(rows)


def sample(n: int = N_PER_SETUP, seed: int = 7) -> list[dict]:
    from .placebo import Pool
    from .research import Universe
    from .setups import BY_NAME
    rng = np.random.default_rng(seed)
    uni = Universe(load_universe("100"))
    events = []
    for name, vi in PICK.items():
        setup = BY_NAME[name]
        pool = Pool(setup, uni)
        t = pool.table(vi)
        is_ = pool.grid[t.g] < pool.oos_ns
        cand = np.flatnonzero(is_)
        rng.shuffle(cand)
        taken, used = [], {}
        for j in cand:
            k, i = int(t.sym[j]), int(pool.pos[t.g[j], int(t.sym[j])])
            if i < 30 * 24 * (4 if setup.tf == "15m" else 1) or i + 100 >= len(pool.bars[k].c):
                continue
            if any(abs(i - q) < 48 for q in used.get(k, [])):      # не два сигнала одного эпизода
                continue
            used.setdefault(k, []).append(i)
            taken.append((k, i, int(t.dir[j])))
            if len(taken) == n:
                break
        btc = load("BTCUSDT", setup.tf)
        for k, i, d in taken:
            sym = pool.syms[k]
            df = uni._c[(sym, setup.tf)].df
            events.append({"id": f"{name[:3]}{len(events):03d}", "setup": name, "variant": vi,
                           "sym": sym, "i": i, "time": str(df.index[i]), "dir": d,
                           "close": float(df["close"].iloc[i]),
                           "snapshot": snapshot(df, i, d, setup.tf, btc, setup, sym)})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "events.json").write_text(json.dumps(events, ensure_ascii=False, indent=1))
    # то, что видит трейдер: только id и снимок
    (OUT / "blind.json").write_text(json.dumps([{"id": e["id"], "snapshot": e["snapshot"]}
                                                for e in events], ensure_ascii=False, indent=1))
    return events


def _plateau_exit(name: str, vi: int) -> tuple[float, Exit]:
    r = json.loads((SCREEN / f"{name}.json").read_text())
    v = r["variants"][vi]
    c = v["real"]["plateau_cell"]
    h = r["grid"]["h_bars"][c["h_idx"]]
    return c["entry_atr"], Exit(sl_atr=c["sl_atr"], tp_r=(c["tp_atr"] / c["sl_atr"], 0, 0),
                                tp_frac=(1, 0, 0), time_exit=h, sl_max_atr=20)


def evaluate(decisions: list[dict], costs: Costs = Costs()) -> dict:
    """Решения трейдера против базы (все сигналы, геометрия плато первичного отбора)."""
    from .setups import F
    events = {e["id"]: e for e in json.loads((OUT / "events.json").read_text())}
    frames = {}
    rows = []
    for dcs in decisions:
        e = events.get(dcs["id"])
        if e is None:
            continue
        from .setups import BY_NAME
        setup = BY_NAME[e["setup"]]
        key = (e["sym"], setup.tf)
        if key not in frames:
            df = load(e["sym"], setup.tf)
            frames[key] = (df, ft.atr(df, 14))
        df, atr = frames[key]
        from .engine import Bars
        bars = Bars.from_frame(df, atr)
        i, d, c0 = e["i"], e["dir"], e["close"]
        a = float(atr.iloc[i])
        # база
        off, ex = _plateau_exit(e["setup"], e["variant"])
        if off == 0:
            sg = Signals([i], [d])
        else:
            kind = E_LIMIT if off > 0 else E_STOP
            sg = Signals([i], [d], [np.nan], [kind], [c0 - d * off * a], [6], [0])
        rb = simulate(bars, sg, ex, costs)
        base = float(rb["pnl"][0]) if len(rb["pnl"]) else 0.0
        # трейдер
        pnl, why = 0.0, "пропуск"
        if dcs.get("take"):
            px = lambda v: None if v is None else v / 100 * c0
            ent = dcs.get("entry") or {}
            kind = {"market": E_MARKET, "limit": E_LIMIT, "stop": E_STOP}.get(ent.get("type"), E_MARKET)
            epx = px(ent.get("price")) if kind != E_MARKET else None
            ref = epx or c0
            stop = px(dcs.get("stop"))
            tg = [t for t in (dcs.get("targets") or []) if t.get("price")][:3]
            if stop is None or d * (ref - stop) <= 0:
                why = "стоп не с той стороны"
            else:
                risk = d * (ref - stop)
                tp_r = [max(d * (px(t["price"]) - ref) / risk, 0.05) for t in tg] or [0.0]
                fr = [float(t.get("frac", 1 / len(tg))) for t in tg] or [0.0]
                s_ = sum(fr) or 1
                fr = [x / s_ for x in fr]
                tp_r += [0.0] * (3 - len(tp_r))
                fr += [0.0] * (3 - len(fr))
                exd = Exit(sl_atr=0, sl_buf=0, sl_max_atr=50, tp_r=tuple(tp_r), tp_frac=tuple(fr),
                           be=bool(dcs.get("breakeven_after_tp1")),
                           trail_atr=float(dcs.get("trail_atr") or 0), time_exit=int(dcs.get("max_bars") or 0))
                sg = Signals([i], [d], [stop], [kind], [np.nan if epx is None else epx],
                             [int(ent.get("ttl_bars") or 6)], [int(ent.get("wait_bars") or 0)])
                rt = simulate(bars, sg, exd, costs)
                if len(rt["pnl"]):
                    pnl, why = float(rt["pnl"][0]), "сделка"
                else:
                    why = "ордер не исполнен"
        rows.append({"id": e["id"], "setup": e["setup"], "take": bool(dcs.get("take")),
                     "trader_pnl": pnl, "base_pnl": base, "status": why,
                     "confidence": dcs.get("confidence"), "reasons": dcs.get("reasons")})
    R = pd.DataFrame(rows)
    out = {"n": len(R)}
    rng = np.random.default_rng(0)
    for name, g in [("все", R)] + list(R.groupby("setup")):
        diff = (g["trader_pnl"] - g["base_pnl"]).to_numpy()
        boot = [rng.choice(diff, len(diff)).mean() for _ in range(2000)] if len(diff) else [0]
        taken = g[g["take"]]
        out[name] = {
            "events": int(len(g)), "taken": int(len(taken)),
            "trader_sum": round(float(g["trader_pnl"].sum()), 2),
            "base_sum_all": round(float(g["base_pnl"].sum()), 2),
            "base_sum_taken": round(float(taken["base_pnl"].sum()), 2),   # фильтр без ведения
            "base_sum_skipped": round(float(g.loc[~g["take"], "base_pnl"].sum()), 2),
            "diff_mean": round(float(diff.mean()), 3) if len(diff) else 0,
            "diff_ci90": [round(float(np.quantile(boot, 0.05)), 3), round(float(np.quantile(boot, 0.95)), 3)],
        }
    (OUT / "evaluation.json").write_text(json.dumps({"summary": out, "rows": rows},
                                                    ensure_ascii=False, indent=1, default=str))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "evaluate"])
    ap.add_argument("decisions", nargs="?")
    a = ap.parse_args()
    if a.cmd == "sample":
        ev = sample()
        print(len(ev), "событий;", OUT / "blind.json")
    else:
        print(json.dumps(evaluate(json.loads(Path(a.decisions).read_text())), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
