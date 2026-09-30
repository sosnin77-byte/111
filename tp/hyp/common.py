"""Общее для файлов гипотез: импорты, отложенные периоды, данные другой монеты."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import features as ft
from ..data import load
from ..setups import OOS_CANDLES, OOS_LIQ, OOS_OI, F, Setup, _ok, _sig  # noqa: F401

# Свечи 5m (и собранные из них 15m) есть только с 2026-04-11, поэтому у сетапов на 5m/15m
# без OI и long/short отложенный период тот же, что у OI.
OOS_5M = OOS_OI

__all__ = ["np", "pd", "ft", "F", "Setup", "_sig", "_ok", "OOS_CANDLES", "OOS_OI", "OOS_LIQ",
           "OOS_5M", "tf_of", "ref"]

_TF = {pd.Timedelta("5min"): "5m", pd.Timedelta("15min"): "15m", pd.Timedelta("1h"): "1h",
       pd.Timedelta("4h"): "4h"}
_REF: dict = {}


def tf_of(f: F) -> str:
    """Таймфрейм таблицы по шагу индекса."""
    return _TF[(f.df.index[1:] - f.df.index[:-1]).min()]


def ref(f: F, sym: str = "BTCUSDT") -> pd.DataFrame:
    """Таблица другой монеты (обычно BTC) на сетке f.df — для межрыночных сетапов.

    Бары с одинаковой меткой описывают один и тот же интервал, заглядывания вперёд нет.
    Для самой монеты sym возвращает f.df.
    """
    if f.sym == sym:
        return f.df
    tf = tf_of(f)
    if (sym, tf) not in _REF:
        _REF[(sym, tf)] = load(sym, tf)
    d = _REF[(sym, tf)]
    if d is None:
        return pd.DataFrame(np.nan, index=f.df.index, columns=f.df.columns)
    return d.reindex(f.df.index)
