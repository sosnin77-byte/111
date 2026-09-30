"""Кандидаты в сетапы. Каждый возвращает сигналы на закрытии бара (Signals).

Сетап описывает:
- tf: рабочий таймфрейм;
- needs: без каких колонок сигналы не считаются (NaN в них — сигнала нет);
- oos_start: начало отложенного периода, который оптимизация не видит;
- variants: 3 варианта параметров самого сигнала (дальше они умножаются на сетку выходов);
- struct_stop: у сигнала есть свой уровень стопа (фитиль, свинг).
Идеи взяты из гайда TRADER.PRO (Liquidity Sweeps, Stop Hunter, OI Change / Net Delta,
Long/Short «толпа против китов», ликбез по OI, ликвидациям и фандингу, торговля по стакану).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import features as ft
from .engine import Signals


class F:
    """Кэш признаков одной монеты на одном ТФ, чтобы варианты сетапов не считали их заново."""

    def __init__(self, df: pd.DataFrame, sym: str = ""):
        self.df = df
        self.sym = sym
        self._c: dict = {}

    def get(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    @property
    def atr(self):
        return self.get("atr", lambda: ft.atr(self.df, 14))

    def ema(self, n):
        return self.get(("ema", n), lambda: ft.ema(self.df["close"], n))

    @property
    def rsi(self):
        return self.get("rsi", lambda: ft.rsi(self.df["close"], 14))

    @property
    def mfi(self):
        return self.get("mfi", lambda: ft.mfi(self.df, 14))

    def bb(self, n=20, k=2.0):
        return self.get(("bb", n, k), lambda: ft.bbands(self.df["close"], n, k))

    def vol_spike(self, n):
        return self.get(("vsp", n), lambda: ft.spike(self.df["qv"], n))

    def hh(self, n):
        """Максимум хаев прошлых n баров (без текущего)."""
        return self.get(("hh", n), lambda: self.df["high"].shift().rolling(n).max())

    def ll(self, n):
        return self.get(("ll", n), lambda: self.df["low"].shift().rolling(n).min())

    def oi_chg(self, n):
        return self.get(("oic", n), lambda: self.df["oi"] / self.df["oi"].shift(n) - 1)

    def delta_sum(self, n):
        return self.get(("ds", n), lambda: self.df["delta"].rolling(n).sum())

    def qv_sum(self, n):
        return self.get(("qs", n), lambda: self.df["qv"].rolling(n).sum())

    def sweeps(self, left, right, sig, depth):
        return self.get(("sw", left, right, sig, depth),
                        lambda: ft.sweeps(self.df, self.atr, left, right, sig, depth))

    @property
    def clv(self):
        """Где закрылся бар внутри своего диапазона: 0 — на лоу, 1 — на хае."""
        d = self.df
        return self.get("clv", lambda: ((d["close"] - d["low"]) /
                                        (d["high"] - d["low"]).replace(0, np.nan)))


def _sig(long_mask: pd.Series, short_mask: pd.Series, long_stop=None, short_stop=None,
         cooldown: int = 0) -> Signals:
    lm = long_mask.fillna(False).to_numpy(bool)
    sm = short_mask.fillna(False).to_numpy(bool)
    both = lm & sm
    lm, sm = lm & ~both, sm & ~both
    i = np.flatnonzero(lm | sm)
    d = np.where(lm[i], 1, -1)
    st = np.full(len(i), np.nan)
    if long_stop is not None:
        st = np.where(d == 1, np.asarray(long_stop, float)[i], st)
    if short_stop is not None:
        st = np.where(d == -1, np.asarray(short_stop, float)[i], st)
    if cooldown and len(i):
        keep = [0]
        for q in range(1, len(i)):
            if i[q] - i[keep[-1]] > cooldown:
                keep.append(q)
        i, d, st = i[keep], d[keep], st[keep]
    return Signals(i, d, st)


def _ok(df, cols):
    m = pd.Series(True, df.index)
    for c in cols:
        m &= df[c].notna()
    return m


# ---------------------------------------------------------------- 1. каскад ликвидаций
def liq_flush(f: F, k: float) -> Signals:
    """Каскад ликвидаций лонгов с выкупом: всплеск ликвидаций в k раз выше обычного,
    цена за 4 бара упала больше 2 ATR, бар закрылся в верхней половине (выкупили фитиль).
    Зеркально — шорт-сквиз с выносом шортов."""
    d, a = f.df, f.atr
    n = 7 * 24 * 4 if len(d) and (d.index[1] - d.index[0]) <= pd.Timedelta("15min") else 7 * 24
    ll_sp = f.get(("lsp", n), lambda: ft.spike(d["liq_long"], n))
    ls_sp = f.get(("ssp", n), lambda: ft.spike(d["liq_short"], n))
    drop = d["close"] - d["high"].rolling(4).max()
    pump = d["close"] - d["low"].rolling(4).min()
    lm = (ll_sp >= k) & (drop <= -2 * a) & (f.clv >= 0.5)
    sm = (ls_sp >= k) & (pump >= 2 * a) & (f.clv <= 0.5)
    return _sig(lm, sm, d["low"], d["high"])


# ---------------------------------------------------------------- 2. свип ликвидности
def sweep(f: F, vmult: float) -> Signals:
    """Ложный пробой свинг-уровня: фитиль снял стопы за уровнем, бар закрылся обратно,
    оборот бара в vmult раз выше обычного (стопы реально исполнились)."""
    bl, bx, sl, sx = f.sweeps(10, 5, 0.5, 0.2)
    vs = f.vol_spike(48)
    return _sig(bl.notna() & (vs >= vmult), sl.notna() & (vs >= vmult), bx, sx)


# ---------------------------------------------------------------- 3. Stop Hunter
def stop_hunter(f: F, sens: int) -> Signals:
    """Stop Hunter из гайда: откат против тренда 4h после снятия стопов.
    Тренд 4h (EMA50/200 и ADX > 18), перепроданность (RSI, MFI, нижняя лента Боллинджера,
    нужно sens из 3), свип или прокол ленты за 5 баров, разворотная свеча."""
    d, a = f.df, f.atr
    alt = 0 if f.sym in ("BTCUSDT", "ETHUSDT") else 5

    def gate(x):
        e50, e200, ad = ft.ema(x["close"], 50), ft.ema(x["close"], 200), ft.adx(x, 14)
        return np.sign(e50 - e200).where(ad > 18, 0.0)

    trend = f.get("htf_gate", lambda: ft.htf(d, "4h", gate))
    lo, _, up = f.bb(20, 2.0)
    r, m = f.rsi, f.mfi
    w = 5
    os_ = ((r < 38 + alt).rolling(w).max().fillna(0) + (m < 33 + alt).rolling(w).max().fillna(0)
           + (d["low"] < lo).rolling(w).max().fillna(0))
    ob = ((r > 62 - alt).rolling(w).max().fillna(0) + (m > 67 - alt).rolling(w).max().fillna(0)
          + (d["high"] > up).rolling(w).max().fillna(0))
    bl, _, sl, _ = f.sweeps(5, 2, 0.0, 0.0)
    liq_l = (bl.notna() | (d["low"] < lo)).astype(float).rolling(w).max() > 0
    liq_s = (sl.notna() | (d["high"] > up)).astype(float).rolling(w).max() > 0
    body = (d["close"] - d["open"]).abs() / (d["high"] - d["low"]).replace(0, np.nan)
    turn_up = (r > r.shift()) | (m > m.shift())
    turn_dn = (r < r.shift()) | (m < m.shift())
    lm = (trend > 0) & (os_ >= sens) & liq_l & (d["close"] > d["open"]) & (body >= 0.3) & turn_up
    sm = (trend < 0) & (ob >= sens) & liq_s & (d["close"] < d["open"]) & (body >= 0.3) & turn_dn
    return _sig(lm, sm, d["low"].rolling(5).min(), d["high"].rolling(5).max(), cooldown=3)


# ---------------------------------------------------------------- 4. пробой на новых деньгах
def oi_breakout(f: F, n: int) -> Signals:
    """Пробой n-барного диапазона, обеспеченный новыми позициями: OI за 6 баров вырос
    больше 1.5 %, дельта агрессора за 6 баров в сторону пробоя (long/short build-up)."""
    d = f.df
    oic, ds = f.oi_chg(6), f.delta_sum(6)
    lm = (d["close"] > f.hh(n)) & (oic >= 0.015) & (ds > 0)
    sm = (d["close"] < f.ll(n)) & (oic >= 0.015) & (ds < 0)
    return _sig(lm, sm)


# ---------------------------------------------------------------- 5. капитуляция
def capitulation(f: F, move_atr: float, oi_drop: float) -> Signals:
    """Принудительная разгрузка лонгов: за 6 баров цена упала больше move_atr ATR,
    OI сократился больше oi_drop (лонги выходят), оборот вдвое выше обычного, бар выкуплен.
    Зеркально — вынос шортов (рост без новых денег, OI падает) — шорт."""
    d, a = f.df, f.atr
    ch = d["close"] - d["close"].shift(6)
    oic, vs = f.oi_chg(6), f.vol_spike(48)
    lm = (ch <= -move_atr * a) & (oic <= -oi_drop) & (vs >= 2) & (f.clv >= 0.5)
    sm = (ch >= move_atr * a) & (oic <= -oi_drop) & (vs >= 2) & (f.clv <= 0.5)
    return _sig(lm, sm, d["low"], d["high"])


# ---------------------------------------------------------------- 6. топливо для сквиза
def squeeze_fuel(f: F, oi_up: float) -> Signals:
    """Шорты набирают позиции: за 24 бара OI вырос больше oi_up, цена не выросла,
    фандинг отрицательный. Вход в лонг на пробое 12-барного хая (начало сквиза).
    Зеркально — лонги набирают на плоской цене при дорогом фандинге, шорт на пробое лоу."""
    d, a = f.df, f.atr
    oic = f.oi_chg(24)
    ch = d["close"].shift() - d["close"].shift(25)
    fl = d["fund_last"]
    lm = (oic >= oi_up) & (ch <= 0.5 * a) & (fl < 0) & (d["close"] > f.hh(12))
    sm = (oic >= oi_up) & (ch >= -0.5 * a) & (fl > 0.0001) & (d["close"] < f.ll(12))
    return _sig(lm, sm)


# ---------------------------------------------------------------- 7. перекос фандинга
def funding_fade(f: F, z: float) -> Signals:
    """Толпа перегружена: фандинг на экстремуме своей истории за 30 дней (z-score),
    цена у 7-дневного экстремума, первый бар против толпы (закрытие за лоу/хай
    предыдущего бара). Дорогой фандинг у хаёв — шорт, отрицательный у лоёв — лонг."""
    d, a = f.df, f.atr
    fz = f.get("fz", lambda: ft.zscore(d["fund_last"], 30 * 24))
    near_hi = d["high"].rolling(168).max() - d["close"] <= 1.5 * a
    near_lo = d["close"] - d["low"].rolling(168).min() <= 1.5 * a
    lm = (fz <= -z) & (d["fund_last"] < 0) & near_lo & (d["close"] > d["high"].shift())
    sm = (fz >= z) & (d["fund_last"] > 0.0001) & near_hi & (d["close"] < d["low"].shift())
    return _sig(lm, sm, d["low"].rolling(3).min(), d["high"].rolling(3).max())


# ---------------------------------------------------------------- 8. киты против толпы
def whales_vs_crowd(f: F, gap: float) -> Signals:
    """Топ-трейдеры по позициям стоят против всех аккаунтов: доля лонгов у топов выше, чем у
    толпы, на gap и больше. Входим за китами, когда цена подтверждает — закрытие выше
    хая 6 баров. Зеркально для шорта."""
    d = f.df
    g = d["ls_top"] - d["ls_global"]
    lm = (g >= gap) & (d["close"] > f.hh(6))
    sm = (g <= -gap) & (d["close"] < f.ll(6))
    return _sig(lm, sm)


# ---------------------------------------------------------------- 9. поглощение
def absorption(f: F, r: float) -> Signals:
    """Зона набора: 12 баров агрессивно продают (дельта ≤ -r от оборота), а цена стоит
    (изменение не хуже -0.3 ATR) в нижней трети 48-барного диапазона — лимитный покупатель
    поглощает продажи. Зеркально — раздача у хаёв."""
    d, a = f.df, f.atr
    dr = f.delta_sum(12) / f.qv_sum(12)
    ch = d["close"] - d["close"].shift(12)
    hi, lo = d["high"].rolling(48).max(), d["low"].rolling(48).min()
    pos = (d["close"] - lo) / (hi - lo).replace(0, np.nan)
    lm = (dr <= -r) & (ch >= -0.3 * a) & (pos <= 0.33)
    sm = (dr >= r) & (ch <= 0.3 * a) & (pos >= 0.67)
    return _sig(lm, sm, lo, hi, cooldown=6)


# ---------------------------------------------------------------- 10. сжатие с набором
def squeeze_breakout(f: F, oi_up: float) -> Signals:
    """Сжатие волатильности (ширина полос Боллинджера в нижних 10 % за 10 дней), за время
    сжатия OI вырос больше oi_up — в диапазоне набирали позиции. Направление набора по дельте
    за 24 бара, вход на закрытии за полосой в ту же сторону."""
    d = f.df
    lo, mid, up = f.bb(20, 2.0)
    width = (up - lo) / mid
    pct = f.get("bbw_pct", lambda: width.rolling(240, min_periods=120).rank(pct=True))
    tight = (pct <= 0.10).astype(float).rolling(3).max() > 0
    oic, ds = f.oi_chg(24), f.delta_sum(24)
    lm = tight & (oic >= oi_up) & (ds > 0) & (d["close"] > up)
    sm = tight & (oic >= oi_up) & (ds < 0) & (d["close"] < lo)
    return _sig(lm, sm, mid, mid)


# ---------------------------------------------------------------- 11. за агрессором
def aggressor(f: F, vm: float) -> Signals:
    """Крупный агрессивный покупатель: оборот бара в vm раз выше обычного, покупки тейкеров
    ≥ 60 % оборота, OI за бар вырос больше 0.5 % (новые позиции, а не закрытие шортов),
    закрытие у хая. Зеркально для продавца."""
    d = f.df
    vs = f.vol_spike(96)
    tb = d["tbq"] / d["qv"].replace(0, np.nan)
    oic = f.oi_chg(1)
    lm = (vs >= vm) & (tb >= 0.6) & (oic >= 0.005) & (f.clv >= 0.7)
    sm = (vs >= vm) & (tb <= 0.4) & (oic >= 0.005) & (f.clv <= 0.3)
    return _sig(lm, sm, d["low"], d["high"])


@dataclass
class Setup:
    name: str
    title: str
    fn: Callable
    tf: str
    needs: tuple
    oos_start: str
    variants: list
    struct_stop: bool = False
    extra_exits: list = field(default_factory=list)
    doc: str = ""

    def signals(self, f: F, **params) -> Signals:
        ok = _ok(f.df, self.needs).to_numpy()
        s = self.fn(f, **params)
        keep = ok[s.i]
        return Signals(s.i[keep], s.dir[keep], s.stop[keep])


# Отложенные периоды: примерно последняя четверть доступной истории каждого набора.
OOS_CANDLES = "2026-07-01"    # свечи и фандинг с 2025-10
OOS_OI = "2026-08-10"         # OI, long/short с 2026-04
OOS_LIQ = "2026-09-01"        # ликвидации с 2026-06

SETUPS = [
    Setup("liq_flush", "Каскад ликвидаций с выкупом", liq_flush, "15m", ("liq_long",), OOS_LIQ,
          [{"k": 5}, {"k": 10}, {"k": 20}], struct_stop=True),
    Setup("sweep", "Свип ликвидности (ложный пробой)", sweep, "1h", (), OOS_CANDLES,
          [{"vmult": 1.0}, {"vmult": 1.5}, {"vmult": 2.5}], struct_stop=True),
    Setup("stop_hunter", "Stop Hunter: откат в тренде 4h после снятия стопов", stop_hunter,
          "1h", (), OOS_CANDLES, [{"sens": 1}, {"sens": 2}, {"sens": 3}], struct_stop=True),
    Setup("oi_breakout", "Пробой на новых деньгах (OI и дельта)", oi_breakout, "1h", ("oi",),
          OOS_OI, [{"n": 24}, {"n": 48}, {"n": 96}]),
    Setup("capitulation", "Капитуляция: разгрузка лонгов / вынос шортов", capitulation, "1h",
          ("oi",), OOS_OI, [{"move_atr": 3, "oi_drop": 0.02}, {"move_atr": 4, "oi_drop": 0.03},
                            {"move_atr": 5, "oi_drop": 0.04}], struct_stop=True),
    Setup("squeeze_fuel", "Топливо для сквиза: набор против цены при перекосе фандинга",
          squeeze_fuel, "1h", ("oi", "fund_last"), OOS_OI,
          [{"oi_up": 0.05}, {"oi_up": 0.10}, {"oi_up": 0.15}]),
    Setup("funding_fade", "Экстремальный фандинг против толпы", funding_fade, "1h",
          ("fund_last",), OOS_CANDLES, [{"z": 1.5}, {"z": 2.0}, {"z": 2.5}], struct_stop=True),
    Setup("whales_vs_crowd", "Киты против толпы (long/short топ-позиций)", whales_vs_crowd,
          "1h", ("ls_top", "ls_global"), OOS_OI,
          [{"gap": 0.05}, {"gap": 0.10}, {"gap": 0.15}]),
    Setup("absorption", "Поглощение: зоны набора и раздачи по дельте", absorption, "1h", (),
          OOS_CANDLES, [{"r": 0.05}, {"r": 0.08}, {"r": 0.12}], struct_stop=True),
    Setup("squeeze_breakout", "Сжатие с набором OI и выход из диапазона", squeeze_breakout,
          "1h", ("oi",), OOS_OI, [{"oi_up": 0.0}, {"oi_up": 0.03}, {"oi_up": 0.06}],
          struct_stop=True),
    Setup("aggressor", "За крупным агрессором (объём, тейкеры, OI)", aggressor, "1h", ("oi",),
          OOS_OI, [{"vm": 3}, {"vm": 5}, {"vm": 8}], struct_stop=True),
]
for _s in SETUPS:
    _s.doc = " ".join((_s.fn.__doc__ or "").split())

BY_NAME = {s.name: s for s in SETUPS}
