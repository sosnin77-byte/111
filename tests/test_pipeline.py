"""Прогон всей цепочки на синтетике в формате ответов REST TRADER.PRO."""
import numpy as np
import pandas as pd
import pytest

from tp import research
from tp.data import load
from tp.setups import SETUPS


def _ms(idx):
    return ((idx - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1ms")).to_numpy("int64")


def _write(raw, name, sym, df):
    (raw / name).mkdir(parents=True, exist_ok=True)
    df.to_parquet(raw / name / f"{sym}.parquet", index=False)


def fake_raw(raw, sym, seed):
    rng = np.random.default_rng(seed)
    t5 = pd.date_range("2026-04-01", "2026-09-29", freq="5min", tz="UTC")
    ret = rng.standard_t(3, len(t5)) * 0.002
    close = 100 * np.exp(np.cumsum(ret))
    op = np.r_[close[0], close[:-1]]
    hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.001, len(t5))))
    lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.001, len(t5))))
    qv = rng.lognormal(10, 1, len(t5))
    tbq = qv * rng.uniform(0.3, 0.7, len(t5))
    ms = _ms(t5)
    c5 = pd.DataFrame({"time": ms, "open": op, "high": hi, "low": lo, "close": close,
                       "volume": qv / close, "quote_volume": qv,
                       "taker_buy_quote_volume": tbq})
    _write(raw, "candles_5m", sym, c5)
    c5i = c5.set_index(t5)
    c1 = c5i.resample("1h").agg({"time": "first", "open": "first", "high": "max", "low": "min",
                                 "close": "last", "volume": "sum", "quote_volume": "sum",
                                 "taker_buy_quote_volume": "sum"})
    _write(raw, "candles_1h", sym, c1.reset_index(drop=True))
    oi = 1e7 * np.exp(np.cumsum(rng.normal(0, 0.003, len(t5))))
    _write(raw, "oi_5m", sym, pd.DataFrame({"time": ms, "oi_usd_close": oi}))
    t1 = pd.date_range("2026-04-01", "2026-09-29", freq="1h", tz="UTC")
    _write(raw, "oi_1h", sym, pd.DataFrame({"time": _ms(t1),
                                            "oi_usd_close": oi[::12][:len(t1)]}))
    t8 = pd.date_range("2026-04-01", "2026-09-29", freq="8h", tz="UTC")
    _write(raw, "funding", sym, pd.DataFrame({"time": _ms(t8),
                                              "rate": rng.normal(0.0001, 0.0003, len(t8))}))
    for name, base in (("ls_global", 0.6), ("ls_top", 0.55)):
        _write(raw, name, sym, pd.DataFrame({"time": ms, "long_account":
                                             base + 0.1 * np.sin(np.arange(len(ms)) / 500)}))
    liq = pd.date_range("2026-06-06", "2026-09-29", freq="5min", tz="UTC")
    _write(raw, "liq_5m", sym, pd.DataFrame({
        "time": _ms(liq),
        "long_liq_usd": rng.pareto(1.5, len(liq)) * 1000,
        "short_liq_usd": rng.pareto(1.5, len(liq)) * 1000}))


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    for k, s in enumerate(("AAAUSDT", "BBBUSDT")):
        fake_raw(raw, s, k)
    return raw


def test_load_alignment(raw):
    df = load("AAAUSDT", "1h", raw)
    assert {"oi", "fund", "fund_last", "ls_global", "ls_top", "liq_long"} <= set(df.columns)
    # ликвидаций нет до начала ряда, внутри ряда пустые бары — нули
    assert df.loc[:"2026-06-05", "liq_long"].isna().all()
    assert df.loc["2026-06-07":, "liq_long"].notna().all()
    # фандинг ставится на бар выплаты
    assert (df["fund"] != 0).sum() == pytest.approx(len(df) / 8, rel=0.02)
    df15 = load("AAAUSDT", "15m", raw)
    assert (df15.index[1] - df15.index[0]) == pd.Timedelta("15min")


def test_all_setups_run(raw, tmp_path, monkeypatch):
    monkeypatch.setattr(research, "RESULTS", tmp_path)
    uni = research.Universe(["AAAUSDT", "BBBUSDT"], raw)
    for s in SETUPS:
        r = research.run_setup(s, uni, log=lambda *_: None)
        assert r["combos"] >= 180 - 1e-9 or r["combos"] == len(s.variants) * 60
        assert (tmp_path / f"{s.name}.json").exists()
