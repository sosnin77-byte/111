"""Контрольные проверки сетапов: чем настоящие сигналы лучше случайных.

Для выбранной комбинации (вариант сигнала + выход) настоящие сделки сравниваются с
подменёнными, у которых всё остальное совпадает: те же выходы, число и способ входа,
стоп и цена входа в единицах ATR от close сигнального бара.

Подмены (kind):
    coin  — тот же момент, случайная другая монета, где в этот бар есть данные:
            отделяет выбор монеты от общего движения рынка в этот момент;
    time  — та же монета, другой момент: вся серия сигналов монеты сдвигается по кругу на
            случайное число допустимых баров (внутри оптимизации и внутри отложенного
            периода отдельно), промежутки и кластеры сохраняются: отделяет тайминг
            от дрейфа монеты за период;
    dir   — та же монета и момент, случайное направление (стоп и цена входа отражаются):
            угадываем ли направление, или прибыль даёт волатильность и несимметричный выход.

Дополнительно:
    forward_edge  — средняя доходность сигнала через h баров от входа по open[i+1] без
                    сетки выходов (минус комиссии): есть ли преимущество во входе;
    null_selection — весь отбор (сетка выходов, ранжирование на оптимизации, критерии на
                    отложенном периоде) на подменённых сигналах: сколько «прибыли» даёт
                    выбор лучшей из ~180 комбинаций на шуме (проверка на подгонку).

p-значение: (1 + число подмен с результатом не хуже настоящего) / (1 + число подмен).

    python -m tp.placebo --setup sweep --runs 200         # по лучшим комбинациям из results/
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .engine import Bars, Costs, Exit, Signals, simulate
from .setups import Setup, _ok

KINDS = ("coin", "time", "dir")
HORIZONS = (1, 4, 12, 24)


@dataclass
class Table:
    """Сигналы всех монет одного варианта в виде столбцов (координаты — сетка времени)."""
    sym: np.ndarray       # номер монеты в Pool.syms
    g: np.ndarray         # индекс бара на общей сетке времени
    dir: np.ndarray
    so: np.ndarray        # стоп: (stop - close) / ATR сигнального бара, NaN — нет
    kind: np.ndarray
    po: np.ndarray        # цена входа: (px - close) / ATR, NaN — нет
    ttl: np.ndarray
    wait: np.ndarray

    def __len__(self):
        return len(self.g)

    def take(self, m) -> "Table":
        return Table(*(getattr(self, k)[m] for k in self.__dataclass_fields__))

    def copy(self) -> "Table":
        return Table(*(getattr(self, k).copy() for k in self.__dataclass_fields__))


class Pool:
    """Монеты одного ТФ для сетапа: бары, допустимые для сигнала моменты, настоящие сигналы."""

    def __init__(self, setup: Setup, uni, costs: Costs = Costs(), cuts: list | None = None):
        """cuts — дополнительные границы периодов (метки времени): подмена момента сдвигает
        сигналы только внутри своего периода. Граница отложенного периода есть всегда."""
        self.setup, self.costs = setup, costs
        self.syms, self.bars, self.times, self.valid = [], [], [], []
        self.sig: dict[int, list] = {vi: [] for vi in range(len(setup.variants))}
        for sym, f in uni.frames(setup.tf):
            b = Bars.from_frame(f.df, f.atr)
            ok = _ok(f.df, setup.needs).to_numpy() & (b.atr > 0) & np.isfinite(b.c)
            ok[-1:] = False                        # нужен следующий бар для входа
            self.syms.append(sym)
            self.bars.append(b)
            self.times.append(f.df.index.values.astype("datetime64[ns]").astype(np.int64))
            self.valid.append(ok)
            for vi, var in enumerate(setup.variants):
                self.sig[vi].append(setup.signals(f, **var))
        self.grid = np.unique(np.concatenate(self.times)) if self.times else np.array([], np.int64)
        n, m = len(self.grid), len(self.syms)
        self.pos = np.full((n, m), -1, np.int64)          # индекс бара монеты на сетке
        self.avail = np.zeros((n, m), bool)               # момент допустим для сигнала
        self.gix = []                                     # индекс на сетке для каждого бара монеты
        for k in range(m):
            gi = np.searchsorted(self.grid, self.times[k])
            self.gix.append(gi)
            self.pos[gi, k] = np.arange(len(gi))
            self.avail[gi, k] = self.valid[k]
        self.oos_ns = pd.Timestamp(setup.oos_start, tz="UTC").value
        self.cuts = np.array(sorted({self.oos_ns} | {pd.Timestamp(c, tz="UTC").value
                                                     for c in (cuts or [])}), np.int64)

    # ------------------------------------------------------------ сигналы <-> таблица
    def table(self, vi: int) -> Table:
        parts = []
        for k, sg in enumerate(self.sig[vi]):
            if not len(sg.i):
                continue
            b = self.bars[k]
            c, a = b.c[sg.i], b.atr[sg.i]
            parts.append((np.full(len(sg.i), k), self.gix[k][sg.i], sg.dir, (sg.stop - c) / a,
                          sg.kind, (sg.px - c) / a, sg.ttl, sg.wait))
        if not parts:
            e = np.array([], np.int64)
            return Table(e, e, e, np.array([]), e, np.array([]), e, e)
        cols = [np.concatenate(x) for x in zip(*parts)]
        return Table(*cols)

    def signals(self, t: Table) -> dict[int, Signals]:
        out = {}
        order = np.argsort(t.sym, kind="stable")
        t = t.take(order)
        bounds = np.flatnonzero(np.diff(t.sym)) + 1
        for chunk in np.split(np.arange(len(t)), bounds):
            if not len(chunk):
                continue
            k = int(t.sym[chunk[0]])
            i = self.pos[t.g[chunk], k]
            keep = i >= 0
            chunk, i = chunk[keep], i[keep]
            b = self.bars[k]
            c, a = b.c[i], b.atr[i]
            out[k] = Signals(i, t.dir[chunk], c + t.so[chunk] * a, t.kind[chunk],
                             c + t.po[chunk] * a, t.ttl[chunk], t.wait[chunk])
        return out

    # ------------------------------------------------------------ подмены
    def placebo(self, t: Table, kind: str, rng: np.random.Generator) -> Table:
        t = t.copy()
        if not len(t):
            return t
        if kind == "coin":
            keep = np.ones(len(t), bool)
            for j in range(len(t)):
                cand = np.flatnonzero(self.avail[t.g[j]])
                cand = cand[cand != t.sym[j]]
                if len(cand):
                    t.sym[j] = cand[rng.integers(len(cand))]
                else:
                    keep[j] = False
            return t.take(keep)
        if kind == "time":
            seg = np.searchsorted(self.cuts, self.grid[t.g], side="right")
            keep = np.ones(len(t), bool)
            for k in np.unique(t.sym):
                gi = self.gix[k]
                seg_k = np.searchsorted(self.cuts, self.grid[gi], side="right")
                for per in np.unique(seg[t.sym == k]):
                    rows = np.flatnonzero((t.sym == k) & (seg == per))
                    if not len(rows):
                        continue
                    per_mask = seg_k == per
                    V = np.flatnonzero(self.valid[k] & per_mask)       # допустимые бары монеты
                    if not len(V):
                        keep[rows] = False
                        continue
                    i = self.pos[t.g[rows], k]
                    r = np.searchsorted(V, i)
                    r = np.clip(r, 0, len(V) - 1)
                    u = rng.integers(len(V))
                    t.g[rows] = gi[V[(r + u) % len(V)]]
            return t.take(keep)
        if kind == "dir":
            flip = rng.random(len(t)) < 0.5
            t.dir[flip] *= -1
            t.so[flip] *= -1
            t.po[flip] *= -1
            return t
        raise ValueError(kind)

    # ------------------------------------------------------------ оценка
    def trades(self, t: Table, ex: Exit) -> pd.DataFrame:
        parts = []
        for k, sg in self.signals(t).items():
            r = simulate(self.bars[k], sg, ex, self.costs)
            if len(r["pnl"]):
                parts.append(pd.DataFrame({"entry_time": r["entry_time"],
                                           "exit_time": r["exit_time"], "dir": r["dir"],
                                           "pnl": r["pnl"], "coin": k}))
        if not parts:
            return pd.DataFrame({"entry_time": pd.Series([], dtype="datetime64[ns]"),
                                 "exit_time": pd.Series([], dtype="datetime64[ns]"),
                                 "dir": pd.Series([], dtype=np.int64),
                                 "pnl": pd.Series([], dtype=float),
                                 "coin": pd.Series([], dtype=np.int64)})
        return pd.concat(parts, ignore_index=True)

    def summary(self, t: Table, ex: Exit) -> dict:
        tr = self.trades(t, ex)
        et = pd.to_datetime(tr["entry_time"]).values.astype("datetime64[ns]").astype(np.int64)
        oos = et >= self.oos_ns
        out = {}
        for key, m in (("is", ~oos), ("oos", oos), ("all", np.ones(len(tr), bool))):
            p = tr["pnl"].to_numpy(float)[m]
            gl = -p[p < 0].sum()
            out[key] = {"trades": int(m.sum()), "net": float(p.sum()),
                        "pf": float(p[p > 0].sum() / gl) if gl > 0 else float("inf")}
        lg = tr["dir"].to_numpy() == 1
        out["oos_long"] = {"trades": int((oos & lg).sum()),
                           "net": float(tr["pnl"].to_numpy(float)[oos & lg].sum())}
        return out

    def forward(self, t: Table) -> dict:
        """Средняя доходность сигнала через h баров от входа по open следующего бара, в % и
        в ATR, за вычетом двух тейкерских комиссий и проскальзывания. Без сетки выходов."""
        out = {}
        cost = 2 * (self.costs.fee_taker + self.costs.slip)
        for h in HORIZONS:
            r_pct, r_atr, oos = [], [], []
            for k, sg in self.signals(t).items():
                b = self.bars[k]
                i = sg.i[sg.i + h < len(b.c)]
                d = sg.dir[sg.i + h < len(b.c)]
                ep = b.o[i + 1]
                r_pct.append(d * (b.c[i + h] / ep - 1) - cost)
                r_atr.append(d * (b.c[i + h] - ep) / b.atr[i])
                oos.append(self.times[k][i] >= self.oos_ns)
            if not r_pct:
                out[h] = None
                continue
            rp, ra, oo = (np.concatenate(x) for x in (r_pct, r_atr, oos))
            out[h] = {"n": int(len(rp)), "pct": float(np.mean(rp) * 100),
                      "atr": float(np.mean(ra)), "pct_oos": float(np.mean(rp[oo]) * 100)
                      if oo.any() else None, "hit": float(np.mean(rp > 0))}
        return out


def _p(real: float, null: np.ndarray) -> float:
    return float((1 + np.sum(null >= real)) / (1 + len(null)))


def control(pool: Pool, vi: int, ex: Exit, runs: int = 200, seed: int = 0) -> dict:
    """Настоящий результат комбинации против подмен каждого вида."""
    rng = np.random.default_rng(seed)
    t = pool.table(vi)
    real = pool.summary(t, ex)
    real_fwd = pool.forward(t)
    out = {"real": real, "forward": {"real": real_fwd}}
    for kind in KINDS:
        nets = {k: [] for k in ("is", "oos", "all")}
        fwd = {h: [] for h in HORIZONS}
        for _ in range(runs):
            pt = pool.placebo(t, kind, rng)
            s = pool.summary(pt, ex)
            for k in nets:
                nets[k].append(s[k]["net"])
            if _ < min(runs, 50):          # доходность без выходов считаем на части прогонов
                f = pool.forward(pt)
                for h in HORIZONS:
                    if f[h]:
                        fwd[h].append(f[h]["pct"])
        res = {}
        for k, v in nets.items():
            v = np.asarray(v)
            res[k] = {"mean": float(v.mean()), "p05": float(np.quantile(v, 0.05)),
                      "p95": float(np.quantile(v, 0.95)), "p": _p(real[k]["net"], v)}
        res["forward_p"] = {h: _p(real_fwd[h]["pct"], np.asarray(fwd[h]))
                            for h in HORIZONS if real_fwd[h] and fwd[h]}
        res["forward_mean"] = {h: float(np.mean(fwd[h])) for h in HORIZONS if fwd[h]}
        out[kind] = res
    return out


def entry_control(pool: Pool, vi: int, runs: int = 50, seed: int = 2) -> dict:
    """Дешёвый контроль без сетки выходов: доходность сигнала через h баров против подмен.
    Считается для каждого варианта каждого сетапа, не зависит от подбора выходов."""
    rng = np.random.default_rng(seed)
    t = pool.table(vi)
    real = pool.forward(t)
    out = {"real": real}
    for kind in KINDS:
        null = {h: [] for h in HORIZONS}
        for _ in range(runs):
            f = pool.forward(pool.placebo(t, kind, rng))
            for h in HORIZONS:
                if f[h]:
                    null[h].append(f[h]["pct"])
        out[kind] = {h: {"mean": float(np.mean(null[h])), "p": _p(real[h]["pct"], np.asarray(null[h]))}
                     for h in HORIZONS if real[h] and null[h]}
    return out


def null_selection(pool: Pool, exits: list[Exit], kind: str = "time", runs: int = 20,
                   seed: int = 1, select_fn=None) -> dict:
    """Вся процедура отбора на подменённых сигналах: распределение лучшего результата,
    который отбор находит на шуме. select_fn(combos) -> список отобранных (как в research)."""
    from .research import _score_is, metrics
    rng = np.random.default_rng(seed)
    tabs = [pool.table(vi) for vi in range(len(pool.setup.variants))]
    best_is, best_oos, n_pass = [], [], []
    for _ in range(runs):
        combos = []
        for vi, t in enumerate(tabs):
            pt = pool.placebo(t, kind, rng)
            for ei, ex in enumerate(exits):
                tr = pool.trades(pt, ex)
                et = pd.to_datetime(tr["entry_time"]).values.astype("datetime64[ns]")
                oos = et.astype(np.int64) >= pool.oos_ns
                combos.append({"variant": vi, "exit": ei,
                               "is": metrics(tr[~oos]), "oos": metrics(tr[oos])})
        sc = [_score_is(c["is"]) for c in combos]
        top = combos[int(np.argmax(sc))]
        best_is.append(max(sc) if np.isfinite(max(sc)) else 0.0)
        best_oos.append(top["oos"]["net"])
        n_pass.append(len(select_fn(combos)) if select_fn else 0)
    return {"kind": kind, "runs": runs, "best_is_score": best_is, "best_is_oos_net": best_oos,
            "n_selected": n_pass}


def main():
    from .fetch import load_universe
    from .research import RESULTS, Universe, exit_grid, select
    from .setups import BY_NAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--null-runs", type=int, default=20)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    names = dict(BY_NAME)
    try:
        from . import hyp
        names.update(hyp.BY_NAME)
    except Exception:
        pass
    setup = names[a.setup]
    res = json.loads((RESULTS / f"{setup.name}.json").read_text())
    uni = Universe(a.only or load_universe())
    t0 = time.time()
    pool = Pool(setup, uni)
    out = {"setup": setup.name, "combos": []}
    for c in res["best"] + res["best_long"]:
        ex = Exit(**{k: tuple(v) if isinstance(v, list) else v
                     for k, v in c["exit_params"].items()})
        ctl = control(pool, c["variant"], ex, a.runs)
        out["combos"].append({"label": c["label"], "control": ctl})
        print(c["label"])
        for kind in KINDS:
            r = ctl[kind]
            print(f"  {kind:5s} OOS: реальн {ctl['real']['oos']['net']:8.1f}  подмены "
                  f"{r['oos']['mean']:8.1f} [{r['oos']['p05']:.1f}; {r['oos']['p95']:.1f}]  "
                  f"p={r['oos']['p']:.3f}")
    if a.null_runs:
        out["null"] = null_selection(pool, exit_grid(setup), "time", a.null_runs,
                                     select_fn=lambda cs: select(cs, "is", "oos"))
    (RESULTS / f"{setup.name}_control.json").write_text(json.dumps(out, ensure_ascii=False,
                                                                   default=str, indent=1))
    print(f"готово за {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
