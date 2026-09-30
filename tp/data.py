"""Сборка рядов TRADER.PRO одной монеты в общую таблицу на заданном таймфрейме.

Колонки результата (индекс — время открытия бара, UTC):
    open high low close volume qv tbq      свечи, qv — оборот в USDT,
                                           tbq — покупки тейкеров в USDT
    delta                                  tbq - (qv - tbq), дельта агрессора
    oi                                     открытый интерес в USD на закрытии бара
    fund                                   ставка фандинга, списанная в начале бара, иначе 0
    fund_last                              последняя известная ставка
    ls_global, ls_top                      доля лонгов (0..1): все аккаунты / топ по позициям
    liq_long, liq_short                    ликвидации лонгов / шортов в USD за бар
Отсутствующие данные остаются NaN: сетап сам решает, нужен ли ему ряд.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .fetch import RAW

TF_RULE = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h"}


def _read(name: str, sym: str, raw: Path = RAW) -> pd.DataFrame | None:
    f = raw / name / f"{sym}.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    if df.empty:
        return None
    df.index = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
    return df.sort_index()


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(np.nan, df.index)


def _candles(sym: str, tf: str, raw: Path) -> pd.DataFrame | None:
    src = _read("candles_1h" if tf in ("1h", "4h") else "candles_5m", sym, raw)
    if src is None:
        return None
    c = pd.DataFrame({
        "open": _num(src, "open"), "high": _num(src, "high"), "low": _num(src, "low"),
        "close": _num(src, "close"), "volume": _num(src, "volume"),
        "qv": _num(src, "quote_volume"), "tbq": _num(src, "taker_buy_quote_volume"),
    })
    if tf in ("15m", "4h"):
        c = c.resample(TF_RULE[tf], label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last",
             "volume": "sum", "qv": "sum", "tbq": "sum"}).dropna(subset=["close"])
    c["delta"] = 2 * c["tbq"] - c["qv"]
    return c


def _last_on(series: pd.Series, index: pd.DatetimeIndex, tf: str,
             max_age: pd.Timedelta | None = None) -> pd.Series:
    """Значение ряда на закрытии каждого бара (последняя точка не позже конца бара).

    max_age — сколько после последней точки ряда значение ещё считается актуальным
    (по умолчанию один бар)."""
    end = index + pd.Timedelta(TF_RULE[tf])
    s = series.dropna()
    if s.empty:
        return pd.Series(np.nan, index)
    # точка с меткой t описывает состояние на момент t; для бара [a, a+tf) берём точку <= a+tf
    pos = s.index.searchsorted(end, side="right") - 1
    vals = np.where(pos >= 0, s.to_numpy()[np.clip(pos, 0, None)], np.nan)
    out = pd.Series(vals, index)
    # не протягиваем значение, если ряд ещё не начался или давно кончился
    out[end <= s.index[0]] = np.nan
    out[index > s.index[-1] + (max_age if max_age is not None else pd.Timedelta(TF_RULE[tf]))] = np.nan
    return out


def _sum_on(series: pd.Series, index: pd.DatetimeIndex, tf: str) -> pd.Series:
    s = series.resample(TF_RULE[tf], label="left", closed="left").sum(min_count=1)
    out = s.reindex(index)
    covered = (index >= series.index[0]) & (index <= series.index[-1])
    return out.where(~covered, out.fillna(0.0))


def load(sym: str, tf: str = "1h", raw: Path = RAW) -> pd.DataFrame | None:
    c = _candles(sym, tf, raw)
    if c is None:
        return None
    idx = c.index

    oi = None
    for name in (("oi_1h", "oi_5m") if tf in ("1h", "4h") else ("oi_5m", "oi_1h")):
        d = _read(name, sym, raw)
        if d is not None:
            # OI-бар с меткой t закрывается в t+шаг: сдвигаем на конец бара
            step = pd.Timedelta("1h" if name == "oi_1h" else "5min")
            s = _num(d, "oi_usd_close")
            s.index = s.index + step
            oi = s if oi is None else oi.combine_first(s)
    c["oi"] = _last_on(oi, idx, tf) if oi is not None else np.nan

    f = _read("funding", sym, raw)
    if f is not None:
        rate = _num(f, "rate")
        # метки выплат гуляют на миллисекунды (07:59:59.999): округляем до минуты
        rate.index = rate.index.round("1min")
        rate = rate[~rate.index.duplicated(keep="last")]
        c["fund"] = _sum_on(rate, idx, tf).fillna(0.0)
        c.loc[idx < rate.index[0], "fund"] = 0.0
        # ставка действует до следующей выплаты (интервал до 8 часов)
        c["fund_last"] = _last_on(rate, idx, tf, max_age=pd.Timedelta("9h"))
    else:
        c["fund"] = 0.0
        c["fund_last"] = np.nan

    for name, col in (("ls_global", "ls_global"), ("ls_top", "ls_top")):
        d = _read(name, sym, raw)
        if d is None:
            c[col] = np.nan
            continue
        # снимок публикуется по итогам 5-минутного периода: считаем его известным в конце периода
        s = _num(d, "long_account")
        s.index = s.index + pd.Timedelta("5min")
        c[col] = _last_on(s, idx, tf)

    d = _read("liq_5m", sym, raw)
    if d is not None:
        c["liq_long"] = _sum_on(_num(d, "long_liq_usd"), idx, tf)
        c["liq_short"] = _sum_on(_num(d, "short_liq_usd"), idx, tf)
    else:
        c["liq_long"] = np.nan
        c["liq_short"] = np.nan
    return c
