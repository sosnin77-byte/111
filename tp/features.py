"""Индикаторы и признаки. Все считаются только по прошлому (без заглядывания вперёд)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    pos = mf.where(tp > tp.shift(), 0.0).rolling(n).sum()
    neg = mf.where(tp < tp.shift(), 0.0).rolling(n).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    ndm = dn.where((dn > up) & (dn > 0), 0.0)
    a = atr(df, n)
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / a
    ndi = 100 * ndm.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def bbands(s: pd.Series, n: int = 20, k: float = 2.0):
    m = s.rolling(n).mean()
    sd = s.rolling(n).std(ddof=0)
    return m - k * sd, m, m + k * sd


def zscore(s: pd.Series, n: int) -> pd.Series:
    m = s.rolling(n, min_periods=n // 2).mean()
    sd = s.rolling(n, min_periods=n // 2).std()
    return (s - m) / sd.replace(0, np.nan)


def spike(s: pd.Series, n: int) -> pd.Series:
    """Во сколько раз значение больше среднего за прошлые n баров (без текущего)."""
    base = s.shift().rolling(n, min_periods=n // 2).mean()
    return s / base.replace(0, np.nan)


def htf(df: pd.DataFrame, rule: str, fn) -> pd.Series:
    """Считает fn на старшем ТФ по закрытым барам и возвращает на исходную сетку.

    Значение старшего бара доступно только после его закрытия: переносим метку на конец бара.
    """
    agg = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["close"])
    v = fn(agg)
    v.index = v.index + pd.Timedelta(rule)            # момент, с которого значение известно
    return v.reindex(df.index.union(v.index)).ffill().reindex(df.index)


@njit(cache=True)
def _sweeps(h, l, c, a, left, right, sig_atr, depth_atr, max_age, lows):
    """Свип подтверждённого свинг-уровня: фитиль за уровень, закрытие обратно.

    lows=True — свипы свинг-лоу (бычьи), иначе свинг-хаев (медвежьи).
    Возвращает уровень свипа (NaN, если свипа нет) и экстремум фитиля.
    """
    n = len(c)
    lvl_out = np.full(n, np.nan)
    ext_out = np.full(n, np.nan)
    levels = np.full(64, np.nan)
    born = np.zeros(64, np.int64)
    cnt = 0
    for i in range(n):
        p = i - right                      # пивот, подтверждаемый на баре i
        if p - left >= 0 and a[p] > 0:
            x = l[p] if lows else h[p]
            ok = True
            for q in range(p - left, i + 1):
                if q == p:
                    continue
                if (lows and l[q] <= x) or ((not lows) and h[q] >= x):
                    ok = False
                    break
            if ok:
                # значимость: насколько пивот выделяется относительно окрестности
                if lows:
                    nb = max(h[p - left:i + 1].max() - x, 0.0)
                else:
                    nb = max(x - l[p - left:i + 1].min(), 0.0)
                if nb >= sig_atr * a[p]:
                    if cnt == 64:
                        levels[:63] = levels[1:]
                        born[:63] = born[1:]
                        cnt = 63
                    levels[cnt] = x
                    born[cnt] = i
                    cnt += 1
        # свип текущим баром: уровень, который бар задел, снимается из пула
        best = np.nan
        k = 0
        for q in range(cnt):
            x = levels[q]
            if i - born[q] > max_age:
                continue
            touched = (l[i] < x) if lows else (h[i] > x)
            if born[q] < i and touched:
                deep = (l[i] < x - depth_atr * a[i]) if lows else (h[i] > x + depth_atr * a[i])
                back = (c[i] > x) if lows else (c[i] < x)
                if deep and back:
                    if np.isnan(best) or (lows and x < best) or ((not lows) and x > best):
                        best = x
                continue
            levels[k] = x
            born[k] = born[q]
            k += 1
        cnt = k
        if not np.isnan(best):
            lvl_out[i] = best
            ext_out[i] = l[i] if lows else h[i]
    return lvl_out, ext_out


def sweeps(df: pd.DataFrame, a: pd.Series, left=10, right=5, sig_atr=0.5, depth_atr=0.1,
           max_age=100):
    """Бычьи и медвежьи свипы: (уровень, экстремум фитиля) для каждой стороны."""
    h, l, c = (np.ascontiguousarray(df[k].to_numpy(float)) for k in ("high", "low", "close"))
    av = np.ascontiguousarray(a.fillna(0).to_numpy(float))
    bl, bx = _sweeps(h, l, c, av, left, right, sig_atr, depth_atr, max_age, True)
    sl, sx = _sweeps(h, l, c, av, left, right, sig_atr, depth_atr, max_age, False)
    return (pd.Series(bl, df.index), pd.Series(bx, df.index),
            pd.Series(sl, df.index), pd.Series(sx, df.index))
