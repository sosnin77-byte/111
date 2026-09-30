"""Первичный отбор гипотез без выбора геометрии.

Задача: понять, есть ли в сигнале информация, не зная заранее лучшей геометрии сделки
(вход, стоп, тейк, горизонт), и не принять за преимущество удачно подобранную геометрию.

1. Для каждого сигнала берётся путь цены после него (в ATR от open следующего бара) и по нему
   считается результат сразу всех простых геометрий — «карта»:
     E  вход: стоп-ордер на подтверждение (-0.5 ATR в сторону сделки), рынок (0),
        лимитка на откате (0.25 / 0.5 / 1 ATR), ордер живёт W баров;
     X  тейк 0.5 … 6 ATR от входа;  Y  стоп 0.5 … 4 ATR от входа;
     H  горизонт удержания (три значения по ТФ), дальше выход по close.
   Внутри бара сначала стоп, на баре исполнения лимитки или стоп-ордера тейк не проверяется,
   гэп за стоп — выход по open. Комиссии как в движке: вход рынком и стоп-ордером — тейкер,
   проскальзывание в цене входа (стоп и тейк считаются от неё); лимиткой — мейкер по своей
   цене; тейк — мейкер; стоп и время — тейкер плюс проскальзывание.
2. Статистики карты по сделкам периода оптимизации: лучшая ячейка (t-статистика), лучшее
   плато (средняя t по соседним 3×3 ячейкам тейк × стоп), доля ячеек в плюсе.
3. Те же статистики на подменённых сигналах (tp.placebo: монета, момент, направление).
   p = доля подмен с результатом не хуже. Подмены проходят тот же перебор геометрий, поэтому
   поиск лучшей ячейки не даёт ложного преимущества; все геометрии пробуются, поэтому
   неудачная геометрия не скрывает рабочую гипотезу.
4. Тип преимущества (edge_type): направление должно обыгрывать подмену направления; дальше
   «момент» — обыгрывает ту же монету в другое время, «выбор монеты» — другую монету в тот же
   момент, «момент и монета» — обе. Рыночный тайминг засчитывается.

Сигналы пересекаются во времени и между монетами, t-статистики завышены одинаково для
настоящих сигналов и подмен (подмена монеты сохраняет моменты, подмена момента — монету),
поэтому сравнивать их можно только между собой, не с таблицей Стьюдента.

    python -m tp.screen --pool all              # все сетапы и гипотезы, results/screen/*.json
    python -m tp.screen --setups sweep --runs 200
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from numba import njit

from .engine import Costs
from .placebo import KINDS, Pool, Table

E_GRID = np.array([-0.5, 0.0, 0.25, 0.5, 1.0])
X_GRID = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
Y_GRID = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
H_GRID = {"5m": (12, 48, 144), "15m": (8, 32, 96), "1h": (6, 24, 72), "4h": (3, 12, 36)}
W_BARS = {"5m": 12, "15m": 8, "1h": 6, "4h": 3}
MIN_N = 30            # ячейка учитывается, если в ней не меньше MIN_N сделок

RESULTS = Path(__file__).resolve().parent.parent / "results" / "screen"


# ---------------------------------------------------------------- пути цены после сигналов
def paths(pool: Pool, t: Table, L: int):
    """Матрицы пути длиной L баров после сигнального (бар i+1 … i+L) в ATR сигнального бара,
    в координате «в сторону сделки»: fav — лучший экстремум бара, adv — худший, op, cl —
    open и close. Начало отсчёта — open[i+1]. Возвращает также ATR/цену и признак OOS."""
    parts = []
    for k, s in pool.signals(t).items():
        b = pool.bars[k]
        nb = len(b.c)
        keep = s.i + 1 < nb
        i, d = s.i[keep], s.dir[keep].astype(float)[:, None]
        if not len(i):
            continue
        idx = i[:, None] + 1 + np.arange(L)[None, :]
        inside = idx < nb
        idx = np.minimum(idx, nb - 1)
        o1 = b.o[i + 1][:, None]
        a = b.atr[i][:, None]
        hi, lo = (b.h[idx] - o1) / a, (b.l[idx] - o1) / a
        fav = np.where(d > 0, hi, -lo)
        adv = np.where(d > 0, lo, -hi)
        op = d * (b.o[idx] - o1) / a
        cl = d * (b.c[idx] - o1) / a
        for m in (fav, adv, op, cl):
            m[~inside] = np.nan
        parts.append((fav, adv, op, cl, (a / o1)[:, 0], pool.times[k][i] >= pool.oos_ns))
    if not parts:
        z = np.empty((0, L))
        return z, z, z, z, np.empty(0), np.empty(0, bool)
    return tuple(np.concatenate(x) for x in zip(*parts))


@njit(cache=True)
def _surface(fav, adv, op, cl, scale, E, X, Y, H, W, f_mk, f_tk, slip):
    """Сумма, сумма квадратов и число сделок (в долях номинала) для каждой ячейки E×X×Y×H."""
    nE, nX, nY, nH = len(E), len(X), len(Y), len(H)
    s1 = np.zeros((nE, nX, nY, nH))
    s2 = np.zeros((nE, nX, nY, nH))
    cnt = np.zeros((nE, nX, nY, nH))
    L = fav.shape[1]
    for ev in range(fav.shape[0]):
        sc = scale[ev]
        if not (sc > 0):
            continue
        for ie in range(nE):
            e = E[ie]
            # --- вход: координата P (ATR от open[i+1] в сторону сделки) и бар k0
            k0 = -1
            P = 0.0
            fee_in = f_tk                  # проскальзывание рыночного входа — в цене входа
            if e == 0.0:
                k0 = 0
                P = slip / sc
            elif e > 0.0:                                   # лимитка на откате
                for k in range(min(W, L)):
                    if np.isnan(adv[ev, k]):
                        break
                    if adv[ev, k] <= -e:
                        k0 = k
                        P = -e                              # по своей цене, даже на гэпе
                        fee_in = f_mk
                        break
            else:                                           # стоп-ордер на подтверждение
                for k in range(min(W, L)):
                    if np.isnan(fav[ev, k]):
                        break
                    if fav[ev, k] >= -e:
                        k0 = k
                        P = max(-e, op[ev, k]) + slip / sc
                        break
            if k0 < 0:
                continue
            px_in = 1.0 + P * sc                            # цена входа / open[i+1]
            if not (px_in > 0):
                continue
            for ix in range(nX):
                tp = P + X[ix]
                for iy in range(nY):
                    sl = P - Y[iy]
                    for ih in range(nH):
                        last = k0 + H[ih] - 1
                        if last >= L or np.isnan(cl[ev, last]):
                            continue
                        r = np.nan
                        fee_out = f_tk + slip
                        for k in range(k0, last + 1):
                            if adv[ev, k] <= sl:
                                r = sl - P
                                if k > k0 and op[ev, k] < sl:
                                    r = op[ev, k] - P
                                break
                            if (e == 0.0 or k > k0) and fav[ev, k] >= tp:
                                r = X[ix]
                                fee_out = f_mk
                                break
                        if np.isnan(r):
                            r = cl[ev, last] - P
                        pnl = r * sc / px_in - fee_in - fee_out * (1.0 + r * sc / px_in)
                        s1[ie, ix, iy, ih] += pnl
                        s2[ie, ix, iy, ih] += pnl * pnl
                        cnt[ie, ix, iy, ih] += 1.0
    return s1, s2, cnt


def surface(P, tf: str, costs: Costs = Costs(), mask=None) -> dict:
    fav, adv, op, cl, scale, _ = P
    if mask is not None:
        fav, adv, op, cl, scale = fav[mask], adv[mask], op[mask], cl[mask], scale[mask]
    H = np.array(H_GRID[tf], np.int64)
    s1, s2, n = _surface(fav, adv, op, cl, scale, E_GRID, X_GRID, Y_GRID, H, W_BARS[tf],
                         costs.fee_maker, costs.fee_taker, costs.slip)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n
        sd = np.sqrt(np.maximum(s2 / n - mean ** 2, 0))
        t = mean / sd * np.sqrt(n)
    ok = n >= MIN_N
    mean[~ok] = np.nan
    t[~ok] = np.nan
    return {"mean": mean, "t": t, "n": n}


def stats(S: dict) -> dict:
    """Статистики карты: лучшая ячейка, лучшее плато 3×3 (тейк × стоп), доля ячеек в плюсе."""
    t, mean = S["t"], S["mean"]
    if not np.isfinite(t).any():
        return {"best_t": np.nan, "plateau_t": np.nan, "breadth": np.nan, "best_mean": np.nan,
                "best_cell": None, "plateau_cell": None}
    bi = np.unravel_index(np.nanargmax(t), t.shape)
    tt = np.nan_to_num(t, nan=-5.0)               # пустые ячейки тянут плато вниз
    pad = np.pad(tt, ((0, 0), (1, 1), (1, 1), (0, 0)), constant_values=-5.0)
    plat = np.zeros_like(tt)
    for dx in (0, 1, 2):
        for dy in (0, 1, 2):
            plat += pad[:, dx:dx + tt.shape[1], dy:dy + tt.shape[2], :]
    plat /= 9
    pi = np.unravel_index(np.argmax(plat), plat.shape)
    cell = lambda c: {"entry_atr": float(E_GRID[c[0]]), "tp_atr": float(X_GRID[c[1]]),
                      "sl_atr": float(Y_GRID[c[2]]), "h_idx": int(c[3])}
    return {"best_t": float(t[bi]), "plateau_t": float(plat[pi]),
            "breadth": float(np.sum(mean > 0) / max(np.isfinite(mean).sum(), 1)), "best_mean": float(np.nanmax(mean) * 100),
            "best_cell": cell(bi), "plateau_cell": cell(pi)}


STATS = ("best_t", "plateau_t", "breadth")
SCREEN_P = 0.2        # мягкий порог первичного отбора


def edge_type(p: dict, thr: float) -> str | None:
    """Тип преимущества по p-значениям подмен. Направление обязательно (иначе прибыль даёт
    волатильность, а не прогноз). Момент — лучше той же монеты в другое время; монета — лучше
    другой монеты в тот же момент. Рыночный тайминг (каскад отскакивает у всех монет сразу)
    не проигрывает подмене монеты, но это настоящее преимущество, поэтому тоже засчитывается."""
    if p["dir"] > thr:
        return None
    timing, coin = p["time"] <= thr, p["coin"] <= thr
    if timing and coin:
        return "момент и монета"
    if timing:
        return "момент (рыночный)"
    if coin:
        return "выбор монеты"
    return None


def screen_variant(pool: Pool, vi: int, runs: int = 100, seed: int = 3) -> dict:
    tf = pool.setup.tf
    L = W_BARS[tf] + max(H_GRID[tf])
    t = pool.table(vi)
    P = paths(pool, t, L)
    is_ = ~P[5]
    real = surface(P, tf, pool.costs, is_)
    rs = stats(real)
    oos_S = surface(P, tf, pool.costs, P[5])
    out = {"variant": pool.setup.variants[vi], "signals": int(len(t)),
           "signals_is": int(is_.sum()), "real": rs,
           "real_map_is": {"mean_pct": np.round(real["mean"] * 100, 4).tolist(),
                           "n": real["n"].astype(int).tolist()},
           "oos_at_plateau": None}
    if rs["plateau_cell"]:
        c = rs["plateau_cell"]
        ci = (int(np.where(E_GRID == c["entry_atr"])[0][0]), int(np.where(X_GRID == c["tp_atr"])[0][0]),
              int(np.where(Y_GRID == c["sl_atr"])[0][0]), c["h_idx"])
        out["oos_at_plateau"] = {"mean_pct": float(oos_S["mean"][ci] * 100)
                                 if np.isfinite(oos_S["mean"][ci]) else None,
                                 "n": int(oos_S["n"][ci])}
    rng = np.random.default_rng(seed)
    for kind in KINDS:
        null = {s: [] for s in STATS}
        for _ in range(runs):
            pt = pool.placebo(t, kind, rng)
            Pp = paths(pool, pt, L)
            st = stats(surface(Pp, tf, pool.costs, ~Pp[5]))
            for s in STATS:
                null[s].append(st[s])
        res = {}
        for s in STATS:
            v = np.asarray(null[s], float)
            v = v[np.isfinite(v)]
            r = rs[s]
            res[s] = {"p": float((1 + np.sum(v >= r)) / (1 + len(v))) if np.isfinite(r) and len(v)
                      else 1.0, "null_mean": float(v.mean()) if len(v) else None,
                      "null_p90": float(np.quantile(v, 0.9)) if len(v) else None}
        out[kind] = res
    out["edge_type"] = edge_type({k: out[k]["plateau_t"]["p"] for k in KINDS}, SCREEN_P)
    out["pass"] = out["edge_type"] is not None
    return out


def screen_setup(setup, uni, runs: int = 100, costs: Costs = Costs(), log=print) -> dict:
    t0 = time.time()
    pool = Pool(setup, uni, costs)
    res = {"setup": setup.name, "title": setup.title, "tf": setup.tf, "doc": setup.doc,
           "coins": len(pool.syms), "runs": runs,
           "grid": {"entry_atr": E_GRID.tolist(), "tp_atr": X_GRID.tolist(),
                    "sl_atr": Y_GRID.tolist(), "h_bars": list(H_GRID[setup.tf]),
                    "entry_window": W_BARS[setup.tf]},
           "variants": [screen_variant(pool, vi, runs) for vi in range(len(setup.variants))]}
    res["pass"] = any(v["pass"] for v in res["variants"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{setup.name}.json").write_text(json.dumps(res, ensure_ascii=False, default=str))
    types = sorted({v["edge_type"] for v in res["variants"] if v["edge_type"]})
    log(f"{setup.name:28s} {', '.join(types) if types else '—':20s} "
        + " | ".join(f"{v['variant']}: плато t={v['real']['plateau_t']:.2f} p(монета/момент/напр)="
                     f"{v['coin']['plateau_t']['p']:.2f}/{v['time']['plateau_t']['p']:.2f}/"
                     f"{v['dir']['plateau_t']['p']:.2f}" for v in res["variants"])
        + f"  [{time.time() - t0:.0f}s]")
    return res


def main():
    from .fetch import load_universe
    from .research import Universe
    from .setups import BY_NAME, SETUPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--setups", nargs="*")
    ap.add_argument("--pool", choices=["main", "hyp", "all"], default="main")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--universe", default="100",
                    help='монеты: "100" — топ-100, "400" — топ-400, "101-400" — средние и мелкие')
    ap.add_argument("--runs", type=int, default=100)
    a = ap.parse_args()
    pool = list(SETUPS) if a.pool in ("main", "all") else []
    names = dict(BY_NAME)
    if a.pool in ("hyp", "all") or a.setups:
        try:
            from . import hyp
            names.update(hyp.BY_NAME)
            if a.pool in ("hyp", "all"):
                pool += hyp.HYP
        except Exception as e:           # гипотезы ещё пишутся
            print("гипотезы недоступны:", e)
    todo = [names[n] for n in a.setups] if a.setups else pool
    uni = Universe(a.only or load_universe(a.universe))
    summary = []
    for s in todo:
        try:
            r = screen_setup(s, uni, a.runs)
            summary.append({"setup": s.name, "pass": r["pass"]})
        except Exception as e:
            print(f"{s.name}: ОШИБКА {e!r}", flush=True)
    print(f"прошли первичный отбор: {sum(x['pass'] for x in summary)}/{len(summary)}")


if __name__ == "__main__":
    main()
