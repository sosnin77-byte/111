"""Гипотезы h001–h010: каскады ликвидаций на 15m/5m (вход в нож, затухание, поглощение,
ловушки, ускорение, новые позиции в каскаде, одиночный каскад альта, V-выкуп, киты,
продолжение при перекосе фандинга).

Обозначения: Lq(n) = liq_long за n баров / oi на закрытии бара перед окном (oi.shift(n)),
Sq(n) — то же для liq_short; OIк = oi / close (открытый интерес в монетах);
clv = (close − low)/(high − low). Структурные стопы дополнительно требуют «стоп строго по
свою сторону от close»: бар, где стоп совпал бы с close, сигналом не считается.
"""
from __future__ import annotations

from numba import njit

from .common import *  # noqa: F401,F403
from ..engine import Signals
from .common import F, OOS_LIQ, Setup, _sig, np, pd, ref


# ---------------------------------------------------------------- общие помощники файла
def _lq(f: F, n: int, side: str = "long") -> pd.Series:
    """Ликвидации стороны за n баров в долях OI на закрытии бара перед окном."""
    col = "liq_long" if side == "long" else "liq_short"
    d = f.df
    return f.get(("b01_lq", side, n),
                 lambda: d[col].rolling(n).sum() / d["oi"].shift(n).replace(0, np.nan))


def _oik(f: F) -> pd.Series:
    """Открытый интерес в монетах."""
    d = f.df
    return f.get("b01_oik", lambda: d["oi"] / d["close"].replace(0, np.nan))


def _oik_chg(f: F, n: int) -> pd.Series:
    k = _oik(f)
    return f.get(("b01_oikc", n), lambda: k / k.shift(n).replace(0, np.nan) - 1)


def _first(m: pd.Series) -> pd.Series:
    """Первый бар серии: условие истинно сейчас и ложно на прошлом баре."""
    m = m.fillna(False).astype(bool)
    return m & ~m.shift(1, fill_value=False)


def _norm(f: F, col: str, n: int) -> pd.Series:
    """Среднее колонки за n прошлых баров (без текущего, с нулями пустых баров).
    Нужно полное окно: до n баров истории ликвидаций нормы нет."""
    d = f.df
    return f.get(("b01_norm", col, n),
                 lambda: d[col].shift().rolling(n, min_periods=n).mean().replace(0, np.nan))


def _fl8(f: F) -> pd.Series:
    """fund_last в пересчёте на 8 часов. Интервал выплат оценивается по прошлым выплатам
    (бары с fund != 0): время между двумя последними выплатами, 1–8 ч; до второй выплаты 8 ч."""
    d = f.df

    def calc():
        hours = pd.Series((d.index - d.index[0]).total_seconds() / 3600.0, index=d.index)
        gap = hours[d["fund"] != 0].diff().reindex(d.index).ffill()
        return d["fund_last"] * 8.0 / gap.clip(1, 8).fillna(8.0)
    return f.get("b01_fl8", calc)


# ---------------------------------------------------------------- h001
def liq_knife_limit(f: F, k: float) -> Signals:
    """Каскад ещё идёт: Lq(2) ≥ 0.3 %, close ниже максимума хаёв 8 баров на 2.5 ATR и бар
    закрылся у лоу (clv ≤ 0.35) — лимитка на покупку на close − k·ATR, живёт 4 бара.
    Зеркально: Sq(2) ≥ 0.3 %, close выше минимума лоёв 8 баров на 2.5 ATR, clv ≥ 0.65 —
    лимитка на продажу на close + k·ATR. Кулдаун 4 бара, стоп ATR из сетки."""
    d, a = f.df, f.atr
    c = d["close"]
    lm = (_lq(f, 2) >= 0.003) & (c <= d["high"].rolling(8).max() - 2.5 * a) & (f.clv <= 0.35)
    sm = (_lq(f, 2, "short") >= 0.003) & (c >= d["low"].rolling(8).min() + 2.5 * a) & (f.clv >= 0.65)
    return _sig(lm, sm, cooldown=4, entry="limit", long_px=c - k * a, short_px=c + k * a, ttl=4)


# ---------------------------------------------------------------- h002
def liq_decay_break(f: F, n: int) -> Signals:
    """Каскадный бар (Lq(1) ≥ 0.25 %, close ниже максимума 4 баров на 2 ATR) был n…n+11 баров
    назад; последние n баров не обновили лоу окна каскада, а их максимальный Lq(1) не больше
    25 % пика окна. Первый такой бар — стоп-вход на пробой максимума n баров + 0.05 ATR
    (живёт 4 бара), стоп за лоу окна каскада. Зеркально для сквиза шортов."""
    d, a = f.df, f.atr
    c, h, lo = d["close"], d["high"], d["low"]
    lq1, sq1 = _lq(f, 1), _lq(f, 1, "short")
    casc_l = ((lq1 >= 0.0025) & (c - h.rolling(4).max() <= -2 * a)).astype(float)
    casc_s = ((sq1 >= 0.0025) & (c - lo.rolling(4).min() >= 2 * a)).astype(float)
    had_l = casc_l.rolling(12).max().shift(n) > 0
    had_s = casc_s.rolling(12).max().shift(n) > 0
    win_lo = lo.rolling(12).min().shift(n)
    win_hi = h.rolling(12).max().shift(n)
    pk_l = lq1.rolling(12).max().shift(n)
    pk_s = sq1.rolling(12).max().shift(n)
    lm = had_l & (lo.rolling(n).min() > win_lo) & (lq1.rolling(n).max() <= 0.25 * pk_l)
    sm = had_s & (h.rolling(n).max() < win_hi) & (sq1.rolling(n).max() <= 0.25 * pk_s)
    lm = _first(lm & (win_lo < c))
    sm = _first(sm & (win_hi > c))
    return _sig(lm, sm, win_lo, win_hi, entry="stop",
                long_px=h.rolling(n).max() + 0.05 * a, short_px=lo.rolling(n).min() - 0.05 * a,
                ttl=4)


# ---------------------------------------------------------------- h003
def liq_absorb(f: F, m: float) -> Signals:
    """Крупные ликвидации лонгов (Lq(1) ≥ 0.15 %), а бар узкий: high − low ≤ m·ATR прошлого
    бара и clv ≥ 0.4 — принудительные продажи поглощены. Лимитка на low + 0.25·диапазона,
    живёт 3 бара, стоп за low бара. Зеркально: Sq(1) ≥ 0.15 %, узкий бар, clv ≤ 0.6, лимитка
    на high − 0.25·диапазона, стоп за high."""
    d, a = f.df, f.atr
    h, lo = d["high"], d["low"]
    rng = h - lo
    narrow = (rng <= m * a.shift()) & (rng > 0)
    lm = (_lq(f, 1) >= 0.0015) & narrow & (f.clv >= 0.4)
    sm = (_lq(f, 1, "short") >= 0.0015) & narrow & (f.clv <= 0.6)
    return _sig(lm, sm, lo, h, entry="limit", long_px=lo + 0.25 * rng, short_px=h - 0.25 * rng,
                ttl=3)


# ---------------------------------------------------------------- h004
def bottom_squeeze_trap(f: F, k: float) -> Signals:
    """У дна (минимум 8 баров не выше минимума 96 баров + 0.5 ATR) всплеск ликвидаций ШОРТОВ:
    liq_short ≥ k × среднего за 672 прошлых бара и ≥ 0.03 % OI, бар зелёный, первый бар серии.
    Лимитка на середину тела бара (живёт 4 бара), стоп за минимум 8 баров. Зеркально у хая:
    всплеск ликвидаций лонгов, красный бар — шорт от середины тела, стоп за максимум 8 баров."""
    d, a = f.df, f.atr
    o, c, h, lo = d["open"], d["close"], d["high"], d["low"]
    oi1 = d["oi"].shift().replace(0, np.nan)
    lo8, hi8 = lo.rolling(8).min(), h.rolling(8).max()
    near_lo = lo8 <= lo.rolling(96).min() + 0.5 * a
    near_hi = hi8 >= h.rolling(96).max() - 0.5 * a
    s_sp = (d["liq_short"] >= k * _norm(f, "liq_short", 672)) & (d["liq_short"] / oi1 >= 0.0003)
    l_sp = (d["liq_long"] >= k * _norm(f, "liq_long", 672)) & (d["liq_long"] / oi1 >= 0.0003)
    lm = _first(near_lo & s_sp & (c > o) & (lo8 < c))
    sm = _first(near_hi & l_sp & (c < o) & (hi8 > c))
    mid = (o + c) / 2
    return _sig(lm, sm, lo8, hi8, entry="limit", long_px=mid, short_px=mid, ttl=4)


# ---------------------------------------------------------------- h005
def cascade_accel(f: F, th: float) -> Signals:
    """Сквиз ускоряется: liq_short растёт три бара подряд (> 0), Sq(3) ≥ th, бар закрылся в
    верхних 20 % (clv ≥ 0.8), OI в монетах за 3 бара упал не больше чем на 1 % — топливо ещё
    есть. Стоп-вход на high + 0.05 ATR (живёт 2 бара), стоп за low бара. Зеркально для
    каскада лонгов: шорт на low − 0.05 ATR, стоп за high."""
    d, a = f.df, f.atr
    h, lo = d["high"], d["low"]
    ls, ll_ = d["liq_short"], d["liq_long"]
    oik3 = _oik_chg(f, 3) >= -0.01
    acc_s = (ls > ls.shift(1)) & (ls.shift(1) > ls.shift(2)) & (ls.shift(2) > 0)
    acc_l = (ll_ > ll_.shift(1)) & (ll_.shift(1) > ll_.shift(2)) & (ll_.shift(2) > 0)
    lm = acc_s & (_lq(f, 3, "short") >= th) & (f.clv >= 0.8) & oik3
    sm = acc_l & (_lq(f, 3) >= th) & (f.clv <= 0.2) & oik3
    return _sig(lm, sm, lo, h, entry="stop", long_px=h + 0.05 * a, short_px=lo - 0.05 * a, ttl=2)


# ---------------------------------------------------------------- h006
def oi_up_in_cascade(f: F, x: float) -> Signals:
    """Каскад лонгов (Lq(4) ≥ 0.3 %, close ниже максимума 8 баров на 2.5 ATR), но OI в монетах
    за 4 бара вырос ≥ x, дельта за 4 бара < 0 и close ниже середины 4-барного диапазона:
    в каскад открылись новые шорты. Стоп-вход на возврат выше середины (живёт 8 баров),
    стоп за минимум 4 баров. Зеркально: сквиз с ростом OI и FOMO-лонгами — шорт ниже середины."""
    d, a = f.df, f.atr
    c, h, lo = d["close"], d["high"], d["low"]
    hi4, lo4 = h.rolling(4).max(), lo.rolling(4).min()
    mid4 = (hi4 + lo4) / 2
    ds4 = d["delta"].rolling(4).sum()
    oiu = _oik_chg(f, 4) >= x
    lm = (_lq(f, 4) >= 0.003) & (c <= h.rolling(8).max() - 2.5 * a) & oiu & (ds4 < 0) & (c < mid4)
    sm = ((_lq(f, 4, "short") >= 0.003) & (c >= lo.rolling(8).min() + 2.5 * a) & oiu & (ds4 > 0)
          & (c > mid4))
    lm = _first(lm & (lo4 < c))
    sm = _first(sm & (hi4 > c))
    return _sig(lm, sm, lo4, hi4, entry="stop", long_px=mid4, short_px=mid4, ttl=8)


# ---------------------------------------------------------------- h007
def idio_cascade(f: F, x: float) -> Signals:
    """Одиночный каскад альта: Lq(1) ≥ 0.3 %, close ниже максимума 4 баров на x ATR, а BTC
    спокоен (за 4 бара не хуже −0.3 %, ликвидации лонгов BTC за 4 бара < 0.03 % его OI).
    Стоп-вход на high + 0.05 ATR (живёт 4 бара), стоп за минимум 4 баров. Зеркально для
    сквиза шортов альта при спокойном BTC. Сам BTC не торгуется."""
    d, a = f.df, f.atr
    c, h, lo = d["close"], d["high"], d["low"]
    if f.sym == "BTCUSDT":
        none = pd.Series(False, d.index)
        return _sig(none, none)
    b = ref(f, "BTCUSDT")
    b_ch = b["close"] / b["close"].shift(4) - 1
    b_oi = b["oi"].shift(4).replace(0, np.nan)
    b_ll = b["liq_long"].rolling(4).sum() / b_oi
    b_ls = b["liq_short"].rolling(4).sum() / b_oi
    lo4, hi4 = lo.rolling(4).min(), h.rolling(4).max()
    lm = ((_lq(f, 1) >= 0.003) & (c - hi4 <= -x * a) & (b_ch >= -0.003) & (b_ll < 0.0003)
          & (lo4 < c))
    sm = ((_lq(f, 1, "short") >= 0.003) & (c - lo4 >= x * a) & (b_ch <= 0.003) & (b_ls < 0.0003)
          & (hi4 > c))
    return _sig(lm, sm, lo4, hi4, entry="stop", long_px=h + 0.05 * a, short_px=lo - 0.05 * a,
                ttl=4)


# ---------------------------------------------------------------- h008
@njit(cache=True)
def _v_reclaim(h, l, c, a, lq, thr, r, look, horizon, drop_atr, up):
    """Эпизод каскада и первый бар V-выкупа.

    up=True — каскад лонгов: H0 = максимум хаёв look баров до каскадного бара включительно,
    CL = минимум лоёв с каскада; сигнал, если за horizon баров после последнего каскадного
    бара H0 − CL ≥ drop_atr·ATR (ATR первого каскадного бара эпизода),
    close ≥ CL + r·(H0 − CL) и close < H0. Новые каскадные бары внутри эпизода расширяют
    его (H0 — максимум, CL — минимум). up=False — зеркально для сквиза шортов (уровни в
    зеркальном смысле: H0 — минимум лоёв до сквиза, CL — максимум хаёв со сквиза).
    Возвращает маску сигналов и уровень H0 на сигнальных барах.
    """
    n = len(c)
    sig = np.zeros(n, np.bool_)
    lvl = np.full(n, np.nan)
    active = False
    last = -1
    H0 = 0.0
    CL = 0.0
    ar = 0.0
    for i in range(n):
        if i >= look - 1 and lq[i] >= thr:
            if up:
                h0 = h[i - look + 1:i + 1].max()
            else:
                h0 = l[i - look + 1:i + 1].min()
            if active and i - last <= horizon:
                if up:
                    H0 = max(H0, h0)
                    CL = min(CL, l[i])
                else:
                    H0 = min(H0, h0)
                    CL = max(CL, h[i])
            else:
                active = True
                H0 = h0
                CL = l[i] if up else h[i]
                ar = a[i]
            last = i
            continue
        if not active:
            continue
        if i - last > horizon:
            active = False
            continue
        if up:
            CL = min(CL, l[i])
            rng = H0 - CL
            ok = rng >= drop_atr * ar and c[i] >= CL + r * rng and c[i] < H0
        else:
            CL = max(CL, h[i])
            rng = CL - H0
            ok = rng >= drop_atr * ar and c[i] <= CL - r * rng and c[i] > H0
        if ok:
            sig[i] = True
            lvl[i] = H0
            active = False
    return sig, lvl


def v_reclaim_break(f: F, r: float) -> Signals:
    """Каскад лонгов (Lq(1) ≥ 0.2 %) снёс цену от H0 (максимум 16 баров до каскада) до CL
    (минимум с каскада) не меньше чем на 3 ATR, и в течение 8 баров close отыграл ≥ r падения,
    оставаясь ниже H0. Первый такой бар — стоп-вход на H0 + 0.05 ATR (живёт 8 баров), стоп за
    минимум 3 баров. Зеркально для сквиза шортов: шорт на пробой начала сквиза L0 − 0.05 ATR."""
    d, a = f.df, f.atr
    c, h, lo = d["close"], d["high"], d["low"]
    arr = [np.ascontiguousarray(d[k].to_numpy(float)) for k in ("high", "low", "close")]
    av = np.ascontiguousarray(a.to_numpy(float))
    lq = np.ascontiguousarray(_lq(f, 1).to_numpy(float))
    sq = np.ascontiguousarray(_lq(f, 1, "short").to_numpy(float))
    ls_, lh0 = _v_reclaim(*arr, av, lq, 0.002, r, 16, 8, 3.0, True)
    ss_, sl0 = _v_reclaim(*arr, av, sq, 0.002, r, 16, 8, 3.0, False)
    lo3, hi3 = lo.rolling(3).min(), h.rolling(3).max()
    lm = pd.Series(ls_, d.index) & (lo3 < c)
    sm = pd.Series(ss_, d.index) & (hi3 > c)
    return _sig(lm, sm, lo3, hi3, entry="stop", long_px=pd.Series(lh0, d.index) + 0.05 * a,
                short_px=pd.Series(sl0, d.index) - 0.05 * a, ttl=8)


# ---------------------------------------------------------------- h009
def whale_cascade_buy(f: F, g: float) -> Signals:
    """Каскад лонгов (Lq(4) ≥ 0.3 %, close ниже максимума 8 баров на 2 ATR), а за те же 4 бара
    доля лонгов у топ-трейдеров выросла ≥ g, у всех аккаунтов не выросла: крупные выкупают
    принудительные продажи. Первый бар — вход по рынку, стоп за минимум 4 баров. Зеркально:
    сквиз шортов, топы сокращают лонги на ≥ g, толпа нет — шорт."""
    d, a = f.df, f.atr
    c, h, lo = d["close"], d["high"], d["low"]
    dt = d["ls_top"] - d["ls_top"].shift(4)
    dg = d["ls_global"] - d["ls_global"].shift(4)
    lo4, hi4 = lo.rolling(4).min(), h.rolling(4).max()
    lm = (_lq(f, 4) >= 0.003) & (c <= h.rolling(8).max() - 2 * a) & (dt >= g) & (dg <= 0)
    sm = (_lq(f, 4, "short") >= 0.003) & (c >= lo.rolling(8).min() + 2 * a) & (dt <= -g) & (dg >= 0)
    lm = _first(lm & (lo4 < c))
    sm = _first(sm & (hi4 > c))
    return _sig(lm, sm, lo4, hi4, entry="market")


# ---------------------------------------------------------------- h010
def liq_crowd_go(f: F, k: float) -> Signals:
    """Шорты перегружены (фандинг в пересчёте на 8 ч ≤ −0.5 б.п.), ликвидации шортов бара
    ≥ k × среднего за 672 прошлых бара, бар закрылся в верхних 30 % — стоп-вход на
    high + 0.1 ATR (живёт 2 бара), стоп за low бара. Зеркально: фандинг ≥ 1.5 б.п./8 ч,
    всплеск ликвидаций лонгов, закрытие в нижних 30 % — шорт на low − 0.1 ATR. Пауза 8 баров."""
    d, a = f.df, f.atr
    h, lo = d["high"], d["low"]
    fl8 = _fl8(f)
    lm = ((fl8 <= -0.00005) & (d["liq_short"] >= k * _norm(f, "liq_short", 672))
          & (f.clv >= 0.7) & (d["liq_short"] > 0))
    sm = ((fl8 >= 0.00015) & (d["liq_long"] >= k * _norm(f, "liq_long", 672))
          & (f.clv <= 0.3) & (d["liq_long"] > 0))
    return _sig(lm, sm, lo, h, cooldown=8, entry="stop", long_px=h + 0.1 * a,
                short_px=lo - 0.1 * a, ttl=2)


_LIQ = ("oi", "liq_long", "liq_short")

SETUPS = [
    Setup("h001_liq_knife_limit", "Лимитка в падающий нож каскада", liq_knife_limit, "15m",
          _LIQ, OOS_LIQ, [{"k": 0.25}, {"k": 0.5}, {"k": 1.0}]),
    Setup("h002_liq_decay_break", "Затухание каскада и пробой полки", liq_decay_break, "15m",
          _LIQ, OOS_LIQ, [{"n": 2}, {"n": 4}, {"n": 8}], struct_stop=True),
    Setup("h003_liq_absorb", "Поглощение ликвидаций без движения цены", liq_absorb, "15m",
          _LIQ, OOS_LIQ, [{"m": 1.0}, {"m": 1.5}, {"m": 2.0}], struct_stop=True),
    Setup("h004_bottom_squeeze_trap", "Вынос шортов на дне (ловушка медведей)",
          bottom_squeeze_trap, "15m", _LIQ, OOS_LIQ, [{"k": 3}, {"k": 6}, {"k": 12}],
          struct_stop=True),
    Setup("h005_cascade_accel", "Ускорение каскада: вход по ходу", cascade_accel, "5m",
          _LIQ, OOS_LIQ, [{"th": 0.0005}, {"th": 0.001}, {"th": 0.002}], struct_stop=True),
    Setup("h006_oi_up_in_cascade", "Новые шорты в каскаде: ловушка", oi_up_in_cascade, "15m",
          _LIQ, OOS_LIQ, [{"x": 0.0}, {"x": 0.005}, {"x": 0.015}], struct_stop=True),
    Setup("h007_idio_cascade", "Одиночный каскад альта при спокойном BTC", idio_cascade, "15m",
          _LIQ, OOS_LIQ, [{"x": 1.5}, {"x": 2.0}, {"x": 3.0}], struct_stop=True),
    Setup("h008_v_reclaim_break", "V-выкуп каскада: пробой его начала", v_reclaim_break, "15m",
          _LIQ, OOS_LIQ, [{"r": 0.5}, {"r": 0.65}, {"r": 0.8}], struct_stop=True),
    Setup("h009_whale_cascade_buy", "Топ-трейдеры выкупают каскад", whale_cascade_buy, "15m",
          _LIQ + ("ls_global", "ls_top"), OOS_LIQ, [{"g": 0.002}, {"g": 0.004}, {"g": 0.008}],
          struct_stop=True),
    Setup("h010_liq_crowd_go", "Ликвидации перегруженной стороны: продолжение каскада",
          liq_crowd_go, "15m", ("liq_long", "liq_short", "fund", "fund_last"), OOS_LIQ,
          [{"k": 5}, {"k": 10}, {"k": 20}], struct_stop=True),
]
