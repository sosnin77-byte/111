"""Гипотезы h011–h020: ликвидации и OI на 1h (дивергенция ликвидаций, откат после сквиза,
однобокие ликвидации, смена стороны принуждения, переворот фандинга после каскада, искра
сквиза на перекошенной толпе, продолжение без отскока, вытряхивание плеча в тренде,
против стороны, ведущей OI, провал набора OI).

Обозначения как в b01: Lq(n) = liq_long за n баров / oi.shift(n), Sq(n) — для liq_short,
OIк = oi / close. Структурные стопы требуют «стоп строго по свою сторону от close».
"""
from __future__ import annotations

from numba import njit

from ..engine import Signals
from .common import *  # noqa: F401,F403
from .b01 import _first, _fl8, _lq, _oik, _oik_chg
from .common import F, OOS_LIQ, OOS_OI, Setup, _sig, np, pd


# ---------------------------------------------------------------- h011
def liq_divergence(f: F, r: float) -> Signals:
    """Новый 48-барный лоу с ликвидациями лонгов Lq(1) ≥ 0.15 %, а 4…47 баров назад была волна
    с Lq(1) ≥ max(r·текущего, 0.3 %): цена ниже, принудительных продаж в r раз меньше.
    Стоп-вход на high + 0.05 ATR (живёт 3 бара), стоп за low бара. Зеркально на новом
    48-барном хае со сквизом шортов меньше прошлой волны в r раз."""
    d, a = f.df, f.atr
    c, h, lo = d["close"], d["high"], d["low"]
    lq1, sq1 = _lq(f, 1), _lq(f, 1, "short")
    prev_l = lq1.shift(4).rolling(44).max()
    prev_s = sq1.shift(4).rolling(44).max()
    lm = ((lo <= f.ll(48)) & (lq1 >= 0.0015) & (prev_l >= np.maximum(r * lq1, 0.003))
          & (lo < c))
    sm = ((h >= f.hh(48)) & (sq1 >= 0.0015) & (prev_s >= np.maximum(r * sq1, 0.003))
          & (h > c))
    return _sig(lm, sm, lo, h, entry="stop", long_px=h + 0.05 * a, short_px=lo - 0.05 * a, ttl=3)


# ---------------------------------------------------------------- h012
def squeeze_retrace(f: F, fr: float) -> Signals:
    """Сквиз шортов: Sq(3) ≥ 0.5 % и close выше минимума 6 баров на 3 ATR, первый бар серии.
    Импульс Lo = минимум 6 баров, Hi = максимум 3 баров; лимитка на Hi − fr·(Hi − Lo), живёт
    12 баров, стоп на уровне отката 78.6 %. Зеркально после каскада лонгов: Hi = максимум 6,
    Lo = минимум 3, лимитка на продажу на Lo + fr·(Hi − Lo), стоп на Lo + 0.786·(Hi − Lo)."""
    d, a = f.df, f.atr
    c, h, lo = d["close"], d["high"], d["low"]
    lo6, hi3 = lo.rolling(6).min(), h.rolling(3).max()
    hi6, lo3 = h.rolling(6).max(), lo.rolling(3).min()
    l_px, l_st = hi3 - fr * (hi3 - lo6), hi3 - 0.786 * (hi3 - lo6)
    s_px, s_st = lo3 + fr * (hi6 - lo3), lo3 + 0.786 * (hi6 - lo3)
    lm = _first((_lq(f, 3, "short") >= 0.005) & (c - lo6 >= 3 * a)) & (l_st < c)
    sm = _first((_lq(f, 3) >= 0.005) & (hi6 - c >= 3 * a)) & (s_st > c)
    return _sig(lm, sm, l_st, s_st, entry="limit", long_px=l_px, short_px=s_px, ttl=12)


# ---------------------------------------------------------------- h013
def liq_imb_pullback(f: F, s: float) -> Signals:
    """За 24 бара ликвидации шортов составляют ≥ s всех ликвидаций, а всего ликвидировано
    ≥ 1.5 % OI начала окна; EMA20 > EMA50 и close > EMA20 + 0.5 ATR. Первый такой бар —
    лимитка на EMA20 (живёт 12 баров), стоп ATR из сетки. Зеркально при однобоких
    ликвидациях лонгов в нисходящем тренде — лимитка на продажу на EMA20."""
    d, a = f.df, f.atr
    c = d["close"]
    L = d["liq_long"].rolling(24).sum()
    S = d["liq_short"].rolling(24).sum()
    tot = L + S
    big = tot / d["oi"].shift(24).replace(0, np.nan) >= 0.015
    sh_s = S / tot.replace(0, np.nan)
    e20, e50 = f.ema(20), f.ema(50)
    lm = _first(big & (sh_s >= s) & (e20 > e50) & (c > e20 + 0.5 * a))
    sm = _first(big & (1 - sh_s >= s) & (e20 < e50) & (c < e20 - 0.5 * a))
    return _sig(lm, sm, entry="limit", long_px=e20, short_px=e20, ttl=12)


# ---------------------------------------------------------------- h014
def liq_side_flip(f: F, s: float) -> Signals:
    """В окне t−26…t−3 доминировали ликвидации лонгов (доля ≥ s, всего ≥ 1 % OI), а за
    последние 3 бара доминируют ликвидации шортов (доля ≥ s, Sq(3) ≥ 0.15 %): принуждение
    переключилось на поздних шортов. Первый бар — вход по рынку, стоп за минимум 24 баров.
    Зеркально: сутки выносили шортов, теперь выносят лонгов — шорт, стоп за максимум 24."""
    d = f.df
    c, h, lo = d["close"], d["high"], d["low"]
    Lw = d["liq_long"].shift(3).rolling(24).sum()
    Sw = d["liq_short"].shift(3).rolling(24).sum()
    tw = Lw + Sw
    big = tw / d["oi"].shift(27).replace(0, np.nan) >= 0.01
    shw_l = Lw / tw.replace(0, np.nan)
    L3 = d["liq_long"].rolling(3).sum()
    S3 = d["liq_short"].rolling(3).sum()
    sh3_s = S3 / (L3 + S3).replace(0, np.nan)
    lo24, hi24 = lo.rolling(24).min(), h.rolling(24).max()
    lm = _first(big & (shw_l >= s) & (sh3_s >= s) & (_lq(f, 3, "short") >= 0.0015)) & (lo24 < c)
    sm = _first(big & (1 - shw_l >= s) & (1 - sh3_s >= s) & (_lq(f, 3) >= 0.0015)) & (hi24 > c)
    return _sig(lm, sm, lo24, hi24, entry="market")


# ---------------------------------------------------------------- h015
def fund_flip_cascade(f: F, th: float) -> Signals:
    """Только лонг. За последние 24 бара был каскад: максимум Lq(4) ≥ th. Фандинг только что
    стал отрицательным (fund_last < 0, на прошлом баре ≥ 0), а в окне t−47…t−24 был
    положительным (максимум fund_last в пересчёте на 8 ч > 0.005 %): толпа перевернулась в
    шорт на дне. Лимитка на close − 0.5 ATR (живёт 6 баров), стоп ATR из сетки."""
    d, a = f.df, f.atr
    c, fl = d["close"], d["fund_last"]
    casc = _lq(f, 4).rolling(24).max() >= th
    flip = (fl < 0) & (fl.shift() >= 0)
    was_pos = _fl8(f).shift(24).rolling(24).max() > 0.00005
    lm = casc & flip & was_pos
    sm = pd.Series(False, d.index)
    return _sig(lm, sm, entry="limit", long_px=c - 0.5 * a, ttl=6)


# ---------------------------------------------------------------- h016
def crowd_ignition(f: F, q: float) -> Signals:
    """Толпа в шортах: перцентиль ls_global за 168 баров ≤ q. Первый всплеск ликвидаций
    шортов: Sq(1) ≥ 0.1 % и максимум Sq(1) за прошлые 24 бара < половины текущего, close выше
    максимума 12 прошлых баров. Стоп-вход на high + 0.05 ATR (живёт 3 бара), стоп за low бара.
    Зеркально: толпа в лонгах (перцентиль ≥ 1 − q), первый всплеск ликвидаций лонгов на пробое
    12-барного лоу — шорт."""
    d, a = f.df, f.atr
    c, h, lo = d["close"], d["high"], d["low"]
    pct = f.get("b02_lsg_pct", lambda: d["ls_global"].rolling(168, min_periods=168).rank(pct=True))
    lq1, sq1 = _lq(f, 1), _lq(f, 1, "short")
    lm = ((pct <= q) & (sq1 >= 0.001) & (sq1.shift().rolling(24).max() < 0.5 * sq1)
          & (c >= f.hh(12)) & (lo < c))
    sm = ((pct >= 1 - q) & (lq1 >= 0.001) & (lq1.shift().rolling(24).max() < 0.5 * lq1)
          & (c <= f.ll(12)) & (h > c))
    return _sig(lm, sm, lo, h, entry="stop", long_px=h + 0.05 * a, short_px=lo - 0.05 * a, ttl=3)


# ---------------------------------------------------------------- h017
def no_bounce_cont(f: F, b: float) -> Signals:
    """Ровно 6 баров назад был каскад лонгов (Lq(1) ≥ 0.2 %, close ниже максимума 4 баров на
    2 ATR). H0 — максимум 8 баров до каскада включительно, CL — минимум 7 баров с каскада;
    отскок мелкий: (максимум 6 баров − CL)/(H0 − CL) ≤ b. Стоп-вход в шорт на CL − 0.1 ATR
    (живёт 12 баров), стоп за максимум 6 баров. Зеркально после сквиза шортов: мелкий откат —
    лонг на пробой хая сквиза CH + 0.1 ATR, стоп за минимум 6 баров."""
    d, a = f.df, f.atr
    c, h, lo = d["close"], d["high"], d["low"]
    lq1, sq1 = _lq(f, 1), _lq(f, 1, "short")
    sq_ev = ((sq1 >= 0.002) & (c - lo.rolling(4).min() >= 2 * a)).shift(6, fill_value=False)
    lq_ev = ((lq1 >= 0.002) & (c - h.rolling(4).max() <= -2 * a)).shift(6, fill_value=False)
    lo6, hi6 = lo.rolling(6).min(), h.rolling(6).max()
    # лонг: после сквиза шортов
    L0 = lo.rolling(8).min().shift(6)
    CH = h.rolling(7).max()
    pull = (CH - lo6) / (CH - L0).replace(0, np.nan)
    lm = sq_ev & (pull <= b) & (lo6 < c)
    # шорт: после каскада лонгов
    H0 = h.rolling(8).max().shift(6)
    CL = lo.rolling(7).min()
    bounce = (hi6 - CL) / (H0 - CL).replace(0, np.nan)
    sm = lq_ev & (bounce <= b) & (hi6 > c)
    return _sig(lm, sm, lo6, hi6, entry="stop", long_px=CH + 0.1 * a, short_px=CL - 0.1 * a,
                ttl=12)


# ---------------------------------------------------------------- h018
def trend_shakeout(f: F, v: float) -> Signals:
    """Восходящий тренд (close > EMA200, EMA50 > EMA200), откат от максимума 24 баров на
    1.5–4 ATR, OI в монетах за 3 бара упал на ≥ v (выбили плечевых лонгов), бар зелёный.
    Стоп-вход на high + 0.05 ATR (живёт 3 бара), стоп за минимум 6 баров. Зеркально в
    нисходящем тренде: отскок на 1.5–4 ATR, OI в монетах упал на ≥ v, бар красный — шорт."""
    d, a = f.df, f.atr
    o, c, h, lo = d["open"], d["close"], d["high"], d["low"]
    e50, e200 = f.ema(50), f.ema(200)
    dd = h.rolling(24).max() - c
    du = c - lo.rolling(24).min()
    drop = _oik_chg(f, 3) <= -v
    lo6, hi6 = lo.rolling(6).min(), h.rolling(6).max()
    lm = ((c > e200) & (e50 > e200) & (dd >= 1.5 * a) & (dd <= 4 * a) & drop & (c > o)
          & (lo6 < c))
    sm = ((c < e200) & (e50 < e200) & (du >= 1.5 * a) & (du <= 4 * a) & drop & (c < o)
          & (hi6 > c))
    return _sig(lm, sm, lo6, hi6, entry="stop", long_px=h + 0.05 * a, short_px=lo - 0.05 * a,
                ttl=3)


# ---------------------------------------------------------------- h019
def oi_driver_fade(f: F, rho: float) -> Signals:
    """Режим по корреляции за 48 баров между доходностью цены и изменением OI в монетах.
    ρ ≤ −rho — ведут шорты (открываются на падениях): у 48-барного лоу (low ≤ лоу + 0.3 ATR,
    close выше лоу) лимитка на покупку на low бара. ρ ≥ rho — ведут лонги: у 48-барного хая
    лимитка на продажу на high бара. Живёт 6 баров, пауза 12, стоп ATR из сетки."""
    d, a = f.df, f.atr
    c, h, lo = d["close"], d["high"], d["low"]

    def corr():
        rp = c.pct_change().replace([np.inf, -np.inf], np.nan)
        ro = _oik(f).pct_change().replace([np.inf, -np.inf], np.nan)
        return rp.rolling(48).corr(ro)
    cr = f.get("b02_rho48", corr)
    ll48, hh48 = f.ll(48), f.hh(48)
    lm = (cr <= -rho) & (lo <= ll48 + 0.3 * a) & (c > ll48) & (lo < c)
    sm = (cr >= rho) & (h >= hh48 - 0.3 * a) & (c < hh48) & (h > c)
    return _sig(lm, sm, cooldown=12, entry="limit", long_px=lo, short_px=h, ttl=6)


# ---------------------------------------------------------------- h020
@njit(cache=True)
def _roundtrip(h, l, c, a, k, v, horizon, up):
    """Событие f: OIк за 3 бара вырос ≥ v, а цена за 3 бара ушла на ≥ 1 ATR (up=True — вниз,
    набор шортов; up=False — вверх, набор лонгов). Сигнал — первый бар j из f+1…f+horizon,
    где OIк ≤ 1.005·OIк[f−3] и close по другую сторону от close[f−3]. Стоп — экстремум
    от f до j (лонг: минимум лоёв, шорт: максимум хаёв); если на бар j выпало несколько
    событий, берётся самый дальний стоп."""
    n = len(c)
    sig = np.zeros(n, np.bool_)
    stop = np.full(n, np.nan)
    for fi in range(3, n):
        k0 = k[fi - 3]
        if not (k0 > 0) or not (k[fi] / k0 - 1 >= v):
            continue
        mv = c[fi] - c[fi - 3]
        if up:
            if not (mv <= -a[fi]):
                continue
        else:
            if not (mv >= a[fi]):
                continue
        ext = l[fi] if up else h[fi]
        c0 = c[fi - 3]
        for j in range(fi + 1, min(n, fi + horizon + 1)):
            ext = min(ext, l[j]) if up else max(ext, h[j])
            back = (c[j] > c0) if up else (c[j] < c0)
            if k[j] <= 1.005 * k0 and back:
                if not sig[j]:
                    sig[j] = True
                    stop[j] = ext
                else:
                    stop[j] = min(stop[j], ext) if up else max(stop[j], ext)
                break
    return sig, stop


def oi_roundtrip(f: F, v: float) -> Signals:
    """Событие: OI в монетах за 3 бара вырос ≥ v, цена за 3 бара упала ≥ 1 ATR (набор шортов).
    Первый бар из следующих 24, где OIк вернулся к ≤ 1.005 исходного и close выше старта
    события, — лонг по рынку, стоп за минимум от события до сигнала. Зеркально: набор лонгов на
    росте вышел назад, цена ниже старта — шорт, стоп за максимум от события до сигнала."""
    d, a = f.df, f.atr
    c = d["close"]
    arr = [np.ascontiguousarray(d[x].to_numpy(float)) for x in ("high", "low", "close")]
    av = np.ascontiguousarray(a.to_numpy(float))
    kv = np.ascontiguousarray(_oik(f).to_numpy(float))
    ls_, lst = _roundtrip(*arr, av, kv, v, 24, True)
    ss_, sst = _roundtrip(*arr, av, kv, v, 24, False)
    lst, sst = pd.Series(lst, d.index), pd.Series(sst, d.index)
    lm = pd.Series(ls_, d.index) & (lst < c)
    sm = pd.Series(ss_, d.index) & (sst > c)
    return _sig(lm, sm, lst, sst, entry="market")


_LIQ = ("oi", "liq_long", "liq_short")

SETUPS = [
    Setup("h011_liq_divergence", "Дивергенция ликвидаций на новом лоу", liq_divergence, "1h",
          _LIQ, OOS_LIQ, [{"r": 1.5}, {"r": 2.0}, {"r": 3.0}], struct_stop=True),
    Setup("h012_squeeze_retrace", "Откат после сквиза: продолжение", squeeze_retrace, "1h",
          _LIQ, OOS_LIQ, [{"fr": 0.382}, {"fr": 0.5}, {"fr": 0.618}], struct_stop=True),
    Setup("h013_liq_imb_pullback", "Однобокие ликвидации: покупка отката к EMA20",
          liq_imb_pullback, "1h", _LIQ, OOS_LIQ, [{"s": 0.65}, {"s": 0.75}, {"s": 0.85}]),
    Setup("h014_liq_side_flip", "Смена стороны принуждения", liq_side_flip, "1h", _LIQ, OOS_LIQ,
          [{"s": 0.6}, {"s": 0.7}, {"s": 0.8}], struct_stop=True),
    Setup("h015_fund_flip_cascade", "Каскад перевернул фандинг в минус", fund_flip_cascade,
          "1h", ("oi", "liq_long", "fund_last"), OOS_LIQ,
          [{"th": 0.0015}, {"th": 0.003}, {"th": 0.006}]),
    Setup("h016_crowd_ignition", "Искра сквиза на перекошенной толпе", crowd_ignition, "1h",
          _LIQ + ("ls_global",), OOS_LIQ, [{"q": 0.1}, {"q": 0.2}, {"q": 0.3}], struct_stop=True),
    Setup("h017_no_bounce_cont", "Нет отскока после каскада: продолжение", no_bounce_cont, "1h",
          _LIQ, OOS_LIQ, [{"b": 0.4}, {"b": 0.5}, {"b": 0.6}], struct_stop=True),
    Setup("h018_trend_shakeout", "Вытряхивание плеча в тренде", trend_shakeout, "1h", ("oi",),
          OOS_OI, [{"v": 0.015}, {"v": 0.025}, {"v": 0.04}], struct_stop=True),
    Setup("h019_oi_driver_fade", "Против стороны, которая ведёт OI", oi_driver_fade, "1h",
          ("oi",), OOS_OI, [{"rho": 0.3}, {"rho": 0.5}, {"rho": 0.7}]),
    Setup("h020_oi_roundtrip", "Прирост OI вернулся назад: провал набора", oi_roundtrip, "1h",
          ("oi",), OOS_OI, [{"v": 0.02}, {"v": 0.035}, {"v": 0.06}], struct_stop=True),
]
