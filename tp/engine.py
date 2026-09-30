"""Бэктест сделок по сигналам на барах одной монеты.

Условности (консервативные):
- сигнал возникает на закрытии бара i. Способ входа задаёт сам сигнал (Signals.kind) или,
  если не задал, настройка выхода (Exit.entry):
  * рынок — open[i+1+wait] (тейкер плюс проскальзывание);
  * лимитка — по цене сигнала px (или close[i] - dir*off*ATR из Exit), ордер выставляется
    после wait баров и живёт ttl баров, исполнение мейкерское по своей цене; если цена
    лимитки уже хуже рынка на момент выставления, ордер исполняется сразу по open как рыночный;
  * стоп-ордер на пробой — срабатывает, когда бар торгует через px, исполнение по худшей из
    px и open (тейкер плюс проскальзывание); если px уже пройдена, вход по open как рыночный;
- внутри бара сначала проверяется стоп, потом тейки: если бар задел и стоп, и тейк,
  считаем, что сработал стоп;
- на баре исполнения лимитки или стоп-ордера тейки не проверяются: неизвестно, что было
  раньше внутри бара; стоп на этом баре проверяется (консервативно);
- перенос в безубыток и трейлинг действуют со следующего бара;
- тейки мейкерские по своей цене, стоп, трейлинг и выход по времени тейкерские с
  проскальзыванием, на гэпе стоп исполняется по open;
- фандинг списывается, если позиция держится в момент выплаты (начало бара k > бара входа);
- если стоп дальше цены ликвидации (изолированная маржа, плечо lev), сделка не открывается;
- по одной монете в одном сетапе одна позиция: сигналы во время открытой сделки пропускаются.

PnL считается в долях номинала; при номинале $100 умножается на 100.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit

REASONS = ("SL", "BE", "TRAIL", "TP", "TIME", "END")
# способ входа сигнала (Signals.kind)
E_DEFAULT, E_MARKET, E_LIMIT, E_STOP = range(4)
R_SL, R_BE, R_TRAIL, R_TP, R_TIME, R_END = range(6)


@dataclass(frozen=True)
class Exit:
    """Параметры управления сделкой."""
    sl_atr: float = 1.5          # стоп в ATR от входа; 0 — структурный стоп сигнала
    sl_buf: float = 0.2          # запас к структурному стопу, ATR
    sl_max_atr: float = 4.0      # предел дистанции стопа, ATR (дальше — сделку не берём)
    tp_r: tuple = (2.0, 0.0, 0.0)      # тейки в R
    tp_frac: tuple = (1.0, 0.0, 0.0)   # доли позиции на тейках
    be: bool = False             # стоп в безубыток после TP1
    trail_atr: float = 0.0       # трейлинг от экстремума, ATR; 0 — нет
    trail_after_tp1: bool = True
    time_exit: int = 24          # баров; 0 — без выхода по времени
    entry: str = "market"        # market | limit
    entry_off: float = 0.3       # отступ лимитки от close сигнального бара, ATR
    entry_ttl: int = 3

    def label(self) -> str:
        tps = "/".join(f"{r:g}R×{f:.2g}" for r, f in zip(self.tp_r, self.tp_frac) if f > 0) or "—"
        sl = f"SL {self.sl_atr:g}ATR" if self.sl_atr > 0 else f"SL струк+{self.sl_buf:g}ATR"
        parts = [sl, f"TP {tps}"]
        if self.be:
            parts.append("BE")
        if self.trail_atr > 0:
            parts.append(f"trail {self.trail_atr:g}ATR")
        if self.time_exit:
            parts.append(f"T {self.time_exit}")
        if self.entry == "limit":
            parts.append(f"limit -{self.entry_off:g}ATR/{self.entry_ttl}")
        return ", ".join(parts)


@dataclass(frozen=True)
class Costs:
    fee_maker: float = 0.00018
    fee_taker: float = 0.00045
    slip: float = 0.0002
    lev: float = 5.0
    mmr: float = 0.01            # поддерживающая маржа (упрощённо, для альтов часто выше)
    notional: float = 100.0


@njit(cache=True)
def _simulate(o, h, l, c, atr, fund, sig_i, sig_dir, sig_stop,
              sig_kind, sig_px, sig_ttl, sig_wait,
              entry_mode, entry_off, entry_ttl,
              sl_atr, sl_buf, sl_max_atr, tp_r, tp_frac, be, trail_atr, trail_after,
              time_exit, fee_maker, fee_taker, slip, lev, mmr):
    n = len(c)
    m = len(sig_i)
    e_i = np.empty(m, np.int64)
    x_i = np.empty(m, np.int64)
    dirs = np.empty(m, np.int64)
    pnl_out = np.empty(m)
    risk_out = np.empty(m)
    reason_out = np.empty(m, np.int64)
    cnt = 0
    busy = -1
    for s in range(m):
        i = sig_i[s]
        d = sig_dir[s]
        if i + 1 >= n or i + 1 <= busy:
            continue
        a = atr[i]
        if not (a > 0):
            continue
        # --- вход
        kind = sig_kind[s]
        px = sig_px[s]
        ttl = sig_ttl[s]
        if kind == E_DEFAULT:
            if entry_mode == 0:
                kind = E_MARKET
            else:
                kind = E_LIMIT
                px = c[i] - d * entry_off * a
                ttl = entry_ttl
        if ttl <= 0:
            ttl = entry_ttl
        t0 = i + 1 + sig_wait[s]          # первый бар, на котором ордер может исполниться
        if t0 >= n or t0 <= busy:
            continue
        if kind != E_MARKET and not (px > 0):
            continue
        # ордер уже «в деньгах» на момент выставления — исполняется по open как рыночный
        if kind == E_LIMIT and d * (px - c[t0 - 1]) >= 0:
            kind = E_MARKET
        if kind == E_STOP and d * (c[t0 - 1] - px) >= 0:
            kind = E_MARKET
        j = -1
        ep = 0.0
        fee_in = fee_taker
        if kind == E_MARKET:
            j = t0
            ep = o[j] * (1.0 + d * slip)
        elif kind == E_LIMIT:
            for t in range(t0, min(n, t0 + ttl)):
                if (d == 1 and l[t] <= px) or (d == -1 and h[t] >= px):
                    j = t
                    ep = px
                    fee_in = fee_maker
                    break
        else:
            for t in range(t0, min(n, t0 + ttl)):
                if (d == 1 and h[t] >= px) or (d == -1 and l[t] <= px):
                    j = t
                    ep = max(px, o[t]) if d == 1 else min(px, o[t])
                    ep = ep * (1.0 + d * slip)
                    break
        if j < 0:
            continue
        market_fill = kind == E_MARKET
        # --- стоп
        if sl_atr > 0:
            dist = sl_atr * a
        else:
            st = sig_stop[s]
            if not (d * (ep - st) > 0):
                continue
            dist = d * (ep - st) + sl_buf * a
        if dist <= 0 or (sl_max_atr > 0 and dist > sl_max_atr * a):
            continue
        stop = ep - d * dist
        liq = ep * (1.0 - d * (1.0 / lev - mmr))
        if d * (stop - liq) <= 0:
            continue
        tp_px = np.empty(3)
        for q in range(3):
            tp_px[q] = ep + d * tp_r[q] * dist
        # --- сопровождение
        rem = 1.0
        pnl = -fee_in
        nxt = 0
        tp1 = False
        cur = stop
        why = R_SL
        ext = ep
        reason = R_END
        k = j
        while k < n:
            if k > j:
                pnl -= d * fund[k] * rem * (o[k] / ep)
            # стоп раньше тейков
            if (d == 1 and l[k] <= cur) or (d == -1 and h[k] >= cur):
                xp = cur
                if k > j:
                    if d == 1 and o[k] < cur:
                        xp = o[k]
                    elif d == -1 and o[k] > cur:
                        xp = o[k]
                xp = xp * (1.0 - d * slip)
                pnl += rem * (d * (xp / ep - 1.0) - fee_taker * xp / ep)
                rem = 0.0
                reason = why
                break
            if market_fill or k > j:
                while nxt < 3 and tp_frac[nxt] > 0:
                    tp = tp_px[nxt]
                    if (d == 1 and h[k] >= tp) or (d == -1 and l[k] <= tp):
                        f = min(tp_frac[nxt], rem)
                        pnl += f * (d * (tp / ep - 1.0) - fee_maker * tp / ep)
                        rem -= f
                        nxt += 1
                        tp1 = True
                    else:
                        break
                if rem <= 1e-9:
                    rem = 0.0
                    reason = R_TP
                    break
            if time_exit > 0 and k - j + 1 >= time_exit:
                xp = c[k] * (1.0 - d * slip)
                pnl += rem * (d * (xp / ep - 1.0) - fee_taker * xp / ep)
                rem = 0.0
                reason = R_TIME
                break
            # перенос стопа на следующий бар
            if be and tp1:
                bep = ep * (1.0 + d * (fee_in + fee_taker + slip))
                if d * (bep - cur) > 0:
                    cur = bep
                    why = R_BE
            if d == 1:
                ext = max(ext, h[k])
            else:
                ext = min(ext, l[k])
            if trail_atr > 0 and (not trail_after or tp1) and atr[k] > 0:
                tsp = ext - d * trail_atr * atr[k]
                if d * (tsp - cur) > 0:
                    cur = tsp
                    why = R_TRAIL
            k += 1
        if rem > 0:
            k = n - 1
            xp = c[k] * (1.0 - d * slip)
            pnl += rem * (d * (xp / ep - 1.0) - fee_taker * xp / ep)
        # потеря не больше маржи (гэп за ликвидацию)
        pnl = max(pnl, -1.0 / lev)
        e_i[cnt] = j
        x_i[cnt] = k
        dirs[cnt] = d
        pnl_out[cnt] = pnl
        risk_out[cnt] = dist / ep
        reason_out[cnt] = reason
        cnt += 1
        busy = k
    return e_i[:cnt], x_i[:cnt], dirs[:cnt], pnl_out[:cnt], risk_out[:cnt], reason_out[:cnt]


@dataclass
class Bars:
    """Массивы одной монеты для быстрого прогона."""
    time: np.ndarray            # datetime64[ns]
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    atr: np.ndarray
    fund: np.ndarray

    @classmethod
    def from_frame(cls, df, atr) -> "Bars":
        f = lambda s: np.ascontiguousarray(np.asarray(s, dtype=np.float64))
        return cls(df.index.values, f(df["open"]), f(df["high"]), f(df["low"]), f(df["close"]),
                   f(atr), f(df["fund"].fillna(0.0)))


@dataclass
class Signals:
    i: np.ndarray                       # индекс сигнального бара
    dir: np.ndarray                     # +1 лонг, -1 шорт
    stop: np.ndarray = field(default=None)  # структурный стоп (цена) или NaN
    kind: np.ndarray = field(default=None)  # способ входа E_*; E_DEFAULT — как в Exit
    px: np.ndarray = field(default=None)    # цена лимитки или стоп-ордера на вход
    ttl: np.ndarray = field(default=None)   # сколько баров живёт ордер; 0 — как в Exit
    wait: np.ndarray = field(default=None)  # сколько баров ждать перед выставлением ордера

    def __post_init__(self):
        n = len(np.asarray(self.i))
        self.i = np.asarray(self.i, dtype=np.int64)
        self.dir = np.asarray(self.dir, dtype=np.int64)
        self.stop = np.full(n, np.nan) if self.stop is None else np.asarray(self.stop, np.float64)
        self.kind = np.zeros(n, np.int64) if self.kind is None else np.asarray(self.kind, np.int64)
        self.px = np.full(n, np.nan) if self.px is None else np.asarray(self.px, np.float64)
        self.ttl = np.zeros(n, np.int64) if self.ttl is None else np.asarray(self.ttl, np.int64)
        self.wait = np.zeros(n, np.int64) if self.wait is None else np.asarray(self.wait, np.int64)
        order = np.argsort(self.i, kind="stable")
        for k in ("i", "dir", "stop", "kind", "px", "ttl", "wait"):
            setattr(self, k, getattr(self, k)[order])

    def take(self, keep) -> "Signals":
        return Signals(self.i[keep], self.dir[keep], self.stop[keep], self.kind[keep],
                       self.px[keep], self.ttl[keep], self.wait[keep])


def simulate(bars: Bars, sig: Signals, ex: Exit, costs: Costs = Costs()) -> dict:
    tp_r = np.asarray(ex.tp_r, dtype=np.float64)
    tp_frac = np.asarray(ex.tp_frac, dtype=np.float64)
    e, x, d, pnl, risk, why = _simulate(
        bars.o, bars.h, bars.l, bars.c, bars.atr, bars.fund, sig.i, sig.dir, sig.stop,
        sig.kind, sig.px, sig.ttl, sig.wait,
        0 if ex.entry == "market" else 1, ex.entry_off, ex.entry_ttl,
        ex.sl_atr, ex.sl_buf, ex.sl_max_atr, tp_r, tp_frac, ex.be, ex.trail_atr,
        ex.trail_after_tp1, ex.time_exit, costs.fee_maker, costs.fee_taker, costs.slip,
        costs.lev, costs.mmr)
    return {
        "entry_time": bars.time[e], "exit_time": bars.time[x], "dir": d,
        "pnl": pnl * costs.notional, "risk": risk, "reason": why,
    }
