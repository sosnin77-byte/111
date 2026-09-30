"""Контрольные подмены: сохраняют то, что должны, и меняют только своё."""
import numpy as np
import pytest

from tp import research
from tp.engine import Exit
from tp.placebo import Pool, control, entry_control
from tp.setups import BY_NAME

from .test_pipeline import fake_raw


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    for k, s in enumerate(("AAAUSDT", "BBBUSDT", "CCCUSDT")):
        fake_raw(raw, s, 10 + k)
    uni = research.Universe(["AAAUSDT", "BBBUSDT", "CCCUSDT"], raw)
    return Pool(BY_NAME["sweep"], uni)


def test_roundtrip(pool):
    t = pool.table(0)
    assert len(t) > 20
    for k, sg in pool.signals(t).items():
        o = pool.sig[0][k]
        assert np.array_equal(sg.i, o.i) and np.array_equal(sg.dir, o.dir)
        assert np.allclose(sg.stop, o.stop, equal_nan=True)


def test_placebo_kinds(pool):
    rng = np.random.default_rng(0)
    t = pool.table(0)
    c = pool.placebo(t, "coin", rng)
    assert len(c) == len(t)
    assert np.array_equal(c.g, t.g) and not np.any(c.sym == t.sym)        # момент тот же, монета другая
    assert pool.avail[c.g, c.sym].all()
    tm = pool.placebo(t, "time", rng)
    assert np.array_equal(tm.sym, t.sym) and np.array_equal(tm.dir, t.dir)
    assert (tm.g != t.g).mean() > 0.9
    # сдвиг не переносит сигналы между оптимизацией и отложенным периодом
    assert np.array_equal(pool.grid[tm.g] >= pool.oos_ns, pool.grid[t.g] >= pool.oos_ns)
    for k in np.unique(t.sym):
        assert pool.valid[k][pool.pos[tm.g[tm.sym == k], k]].all()
    d = pool.placebo(t, "dir", rng)
    flip = d.dir != t.dir
    assert 0.3 < flip.mean() < 0.7
    assert np.allclose(d.so[flip], -t.so[flip], equal_nan=True)
    # стоп после отражения по правильную сторону
    for k, sg in pool.signals(d).items():
        c = pool.bars[k].c[sg.i]
        ok = ~np.isnan(sg.stop)
        assert np.all(sg.dir[ok] * (c[ok] - sg.stop[ok]) > 0)


def test_control_runs(pool):
    ex = Exit(sl_atr=1.5, tp_r=(2, 0, 0), tp_frac=(1, 0, 0), time_exit=24)
    r = control(pool, 0, ex, runs=5)
    for kind in ("coin", "time", "dir"):
        assert 0 < r[kind]["oos"]["p"] <= 1
    e = entry_control(pool, 0, runs=3)
    assert set(e) == {"real", "coin", "time", "dir"}


def test_screen_kernel_matches_engine(pool):
    """Карта первичного отбора считает сделку так же, как движок (рыночный вход, стоп, тейк, время)."""
    from tp import screen as S
    from tp.engine import Costs, Signals, simulate
    from tp.placebo import Table
    costs = Costs()
    t = pool.table(0)
    L = S.W_BARS["1h"] + max(S.H_GRID["1h"])
    pairs = []
    for j in range(min(len(t), 60)):
        k, g, d = int(t.sym[j]), int(t.g[j]), int(t.dir[j])
        i = int(pool.pos[g, k])
        one = Table(*(np.array([v]) for v in (k, g, d, np.nan, 0, np.nan, 0, 0)))
        P = S.paths(pool, one, L)
        for x, y, h in ((1.0, 1.0, 24), (3.0, 1.5, 6), (0.5, 2.0, 72)):
            r = simulate(pool.bars[k], Signals([i], [d]),
                         Exit(sl_atr=y, tp_r=(x / y, 0, 0), tp_frac=(1, 0, 0), time_exit=h,
                              sl_max_atr=10), costs)
            s1, _, n = S._surface(*P[:5], np.array([0.0]), np.array([x]), np.array([y]),
                                  np.array([h]), 6, costs.fee_maker, costs.fee_taker, costs.slip)
            if len(r["pnl"]) and n[0, 0, 0, 0]:
                pairs.append((r["pnl"][0] / 100, s1[0, 0, 0, 0]))
    a = np.array(pairs)
    assert len(a) > 50
    assert np.abs(a[:, 0] - a[:, 1]).max() < 0.002       # фандинг и проскальзывание выхода — приближённо


def test_screen_variant_runs(pool):
    from tp import screen as S
    r = S.screen_variant(pool, 0, runs=5)
    assert set(("coin", "time", "dir", "real", "pass")) <= set(r)
    assert r["signals_is"] > 0
    for k in ("coin", "time", "dir"):
        assert 0 < r[k]["plateau_t"]["p"] <= 1
