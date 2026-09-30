import numpy as np
import pandas as pd
import pytest

from tp.engine import (E_LIMIT, E_MARKET, E_STOP, R_BE, R_SL, R_TIME, R_TP, R_TRAIL, Bars, Costs,
                       Exit, Signals, simulate)

ZERO = Costs(fee_maker=0.0, fee_taker=0.0, slip=0.0)


def bars(rows, atr=1.0, fund=None):
    """rows: список (open, high, low, close); цены около 100, ATR постоянный."""
    a = np.array(rows, dtype=float)
    idx = pd.date_range("2026-01-01", periods=len(a), freq="1h", tz="UTC")
    return Bars(idx.values, a[:, 0].copy(), a[:, 1].copy(), a[:, 2].copy(), a[:, 3].copy(),
                np.full(len(a), atr), np.zeros(len(a)) if fund is None else np.asarray(fund, float))


def run(b, i, d, ex, costs=ZERO, stop=None):
    return simulate(b, Signals([i], [d], None if stop is None else [stop]), ex, costs)


FLAT = (100, 100.2, 99.8, 100)


def test_take_profit_long():
    b = bars([FLAT, FLAT, (100, 102.5, 99.5, 102), FLAT])
    r = run(b, 0, 1, Exit(sl_atr=1, tp_r=(2, 0, 0), tp_frac=(1, 0, 0)))
    assert r["reason"][0] == R_TP
    assert r["pnl"][0] == pytest.approx(2.0)          # +2% от $100


def test_stop_first_when_both_hit():
    b = bars([FLAT, FLAT, (100, 103, 98.5, 101), FLAT])
    r = run(b, 0, 1, Exit(sl_atr=1, tp_r=(2, 0, 0), tp_frac=(1, 0, 0)))
    assert r["reason"][0] == R_SL
    assert r["pnl"][0] == pytest.approx(-1.0)


def test_short_mirror():
    b = bars([FLAT, FLAT, (100, 100.5, 97.5, 98), FLAT])
    r = run(b, 0, -1, Exit(sl_atr=1, tp_r=(2, 0, 0), tp_frac=(1, 0, 0)))
    assert r["reason"][0] == R_TP
    assert r["pnl"][0] == pytest.approx(2.0)


def test_gap_through_stop_fills_at_open():
    b = bars([FLAT, FLAT, (97, 97.5, 96.5, 97), FLAT])
    r = run(b, 0, 1, Exit(sl_atr=1, tp_r=(2, 0, 0), tp_frac=(1, 0, 0)))
    assert r["pnl"][0] == pytest.approx(-3.0)


def test_partial_then_breakeven():
    # TP1 1R половина, затем возврат к входу: вторая половина по безубытку
    b = bars([FLAT, FLAT, (100, 101.2, 99.5, 101), (101, 101.1, 99.9, 100), FLAT])
    ex = Exit(sl_atr=1, tp_r=(1, 3, 0), tp_frac=(0.5, 0.5, 0), be=True)
    r = run(b, 0, 1, ex)
    assert r["reason"][0] == R_BE
    assert r["pnl"][0] == pytest.approx(0.5)


def test_time_exit():
    b = bars([FLAT] * 6)
    r = run(b, 0, 1, Exit(sl_atr=1, tp_r=(2, 0, 0), tp_frac=(1, 0, 0), time_exit=3))
    assert r["reason"][0] == R_TIME
    assert (r["exit_time"][0] - r["entry_time"][0]) == np.timedelta64(2, "h")


def test_trailing_locks_profit():
    rows = [FLAT, FLAT, (100, 101.5, 99.9, 101.4), (101.4, 104, 101.3, 103.8),
            (103.8, 103.9, 101.5, 102), FLAT]
    ex = Exit(sl_atr=1, tp_r=(1, 0, 0), tp_frac=(0.5, 0, 0), trail_atr=1.5)
    r = run(bars(rows), 0, 1, ex)
    assert r["reason"][0] == R_TRAIL
    # половина на +1%, остаток по трейлу 104 - 1.5 = 102.5
    assert r["pnl"][0] == pytest.approx(0.5 + 0.5 * 2.5)


def test_fees_and_slippage():
    b = bars([FLAT, FLAT, (100, 102.5, 99.5, 102), FLAT])
    c = Costs(fee_maker=0.0002, fee_taker=0.0005, slip=0.0)
    r = run(b, 0, 1, Exit(sl_atr=1, tp_r=(2, 0, 0), tp_frac=(1, 0, 0)), c)
    assert r["pnl"][0] == pytest.approx(100 * (0.02 - 0.0005 - 0.0002 * 1.02))


def test_funding_charged_only_while_held():
    fund = [0.001] * 5
    b = bars([FLAT] * 5, fund=fund)
    r = run(b, 0, 1, Exit(sl_atr=1, tp_r=(2, 0, 0), tp_frac=(1, 0, 0), time_exit=3))
    # вход в баре 1, выход на закрытии бара 3: выплаты в начале баров 2 и 3
    assert r["pnl"][0] == pytest.approx(-0.2)


def test_one_position_per_coin():
    b = bars([FLAT] * 10)
    sig = Signals([0, 1, 2, 6], [1, 1, 1, 1])
    r = simulate(b, sig, Exit(sl_atr=1, tp_r=(2, 0, 0), tp_frac=(1, 0, 0), time_exit=3), ZERO)
    assert len(r["pnl"]) == 2


def test_stop_beyond_liquidation_skipped():
    b = bars([FLAT] * 5, atr=10.0)
    r = run(b, 0, 1, Exit(sl_atr=2.5, sl_max_atr=0, tp_r=(2, 0, 0), tp_frac=(1, 0, 0)))
    assert len(r["pnl"]) == 0


def test_structural_stop_and_limit_entry():
    rows = [FLAT, (100, 100.1, 99.5, 99.8), (99.8, 101.5, 99.7, 101), FLAT]
    ex = Exit(sl_atr=0, sl_buf=0.0, tp_r=(1, 0, 0), tp_frac=(1, 0, 0), entry="limit",
              entry_off=0.3, entry_ttl=2)
    r = run(bars(rows), 0, 1, ex, stop=98.7)
    # лимит 99.7 исполнен в баре 1, риск 1.0, тейк 100.7 в баре 2
    assert r["reason"][0] == R_TP
    assert r["pnl"][0] == pytest.approx(100 * (100.7 / 99.7 - 1))


def test_limit_not_filled():
    b = bars([FLAT, (100, 100.5, 99.9, 100.3), FLAT, FLAT])
    ex = Exit(sl_atr=1, entry="limit", entry_off=0.5, entry_ttl=2)
    assert len(run(b, 0, 1, ex)["pnl"]) == 0


def test_stop_on_limit_fill_bar_is_checked():
    rows = [FLAT, (100, 100.1, 98.5, 99), FLAT]
    ex = Exit(sl_atr=1, entry="limit", entry_off=0.3, entry_ttl=2, tp_r=(1, 0, 0),
              tp_frac=(1, 0, 0))
    r = run(bars(rows), 0, 1, ex)
    assert r["reason"][0] == R_SL


# ---------------------------------------------------------------- вход, заданный сигналом
def sig1(i, d, kind, px=np.nan, ttl=0, wait=0, stop=np.nan):
    return Signals([i], [d], [stop], [kind], [px], [ttl], [wait])


EX1 = Exit(sl_atr=1, tp_r=(2, 0, 0), tp_frac=(1, 0, 0), time_exit=0)


def test_signal_limit_at_own_price():
    # лимитка на 99.5, бар 2 до неё не доходит, бар 3 доходит; дальше тейк 101.5
    b = bars([FLAT, FLAT, (100, 100.3, 99.7, 100), (100, 100.1, 99.4, 99.6),
              (99.6, 101.6, 99.6, 101.5), FLAT])
    r = simulate(b, sig1(0, 1, E_LIMIT, px=99.5, ttl=5), EX1, ZERO)
    assert pd.Timestamp(r["entry_time"][0]) == pd.Timestamp("2026-01-01 03:00")
    assert r["reason"][0] == R_TP
    assert r["pnl"][0] == pytest.approx((101.5 / 99.5 - 1) * 100)


def test_signal_limit_expires():
    b = bars([FLAT, FLAT, FLAT, (100, 100.1, 99.4, 99.6), FLAT])
    r = simulate(b, sig1(0, 1, E_LIMIT, px=99.5, ttl=2), EX1, ZERO)
    assert len(r["pnl"]) == 0


def test_signal_limit_already_marketable_fills_at_open():
    b = bars([FLAT, (100.1, 100.2, 99.9, 100), FLAT])
    r = simulate(b, sig1(0, 1, E_LIMIT, px=100.5, ttl=3), EX1, ZERO)
    assert pd.Timestamp(r["entry_time"][0]) == pd.Timestamp("2026-01-01 01:00")
    assert r["pnl"][0] == pytest.approx((100 / 100.1 - 1) * 100)      # по времени/концу: close 100


def test_signal_stop_entry_breakout():
    # стоп-ордер на 100.5: бар 1 не доходит, бар 2 пробивает, вход по 100.5, тейк 102.5
    b = bars([FLAT, FLAT, (100.2, 100.8, 100.1, 100.7), (100.7, 102.6, 100.6, 102.5), FLAT])
    r = simulate(b, sig1(0, 1, E_STOP, px=100.5, ttl=5), EX1, ZERO)
    assert pd.Timestamp(r["entry_time"][0]) == pd.Timestamp("2026-01-01 02:00")
    assert r["reason"][0] == R_TP
    assert r["pnl"][0] == pytest.approx((102.5 / 100.5 - 1) * 100)


def test_signal_stop_entry_gap_fills_at_open():
    b = bars([FLAT, FLAT, (101, 101.2, 100.9, 101), FLAT])
    r = simulate(b, sig1(0, 1, E_STOP, px=100.5, ttl=5), EX1, ZERO)
    assert pd.Timestamp(r["entry_time"][0]) == pd.Timestamp("2026-01-01 02:00")
    assert r["pnl"][0] == pytest.approx((100 / 101 - 1) * 100)


def test_signal_market_after_wait():
    b = bars([FLAT, FLAT, FLAT, (100.3, 100.4, 100.2, 100.3), FLAT])
    r = simulate(b, sig1(0, -1, E_MARKET, wait=2), EX1, ZERO)
    assert pd.Timestamp(r["entry_time"][0]) == pd.Timestamp("2026-01-01 03:00")
    assert r["dir"][0] == -1


def test_sig_helper_entry_fields():
    from tp.setups import _sig
    idx = pd.date_range("2026-01-01", periods=5, freq="1h", tz="UTC")
    lm = pd.Series([False, True, False, False, False], idx)
    sm = pd.Series([False, False, False, True, False], idx)
    px = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], idx)
    s = _sig(lm, sm, entry="limit", long_px=px, short_px=px * 10, ttl=4, wait=1)
    assert list(s.i) == [1, 3] and list(s.dir) == [1, -1]
    assert list(s.px) == [2.0, 40.0] and list(s.kind) == [E_LIMIT, E_LIMIT]
    assert list(s.ttl) == [4, 4] and list(s.wait) == [1, 1]
    assert list(s.take(np.array([False, True])).i) == [3]
