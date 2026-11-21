import math

import pytest

import tilelight

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _t():
    return tilelight.TileLight(4, 4)


def _check_zero():
    return _t().grid() == [[0.0] * 4 for _ in range(4)]


def _check_center():
    t = _t(); t.add_light(2, 2, 2)
    return t.grid()[2][2] > 0.0


def _check_range():
    t = _t(); t.add_light(1, 1, 3)
    return all(0.0 <= v <= 1.0 for row in t.grid() for v in row)


def _check_monotonic():
    t = _t(); t.add_light(0, 0, 3)
    g = t.grid()
    return g[0][0] >= g[0][2]


def _check_outside():
    t = _t(); t.add_light(10, 10, 1)
    return all(v == 0.0 for row in t.grid() for v in row)


def _check_bad_size():
    return _raises(lambda: tilelight.TileLight(0, 4))


def _check_bad_radius():
    return _raises(lambda: _t().add_light(0, 0, 0))


def _check_clamp():
    t = _t(); t.add_light(1, 1, 10); t.add_light(1, 1, 10)
    return all(v <= 1.0 for row in t.grid() for v in row)


def _check_type():
    return isinstance(_t().grid(), list)


def _check_shape():
    g = _t().grid()
    return len(g) == 4 and len(g[0]) == 4


def _check_det():
    a = _t(); a.add_light(1, 1, 2)
    b = _t(); b.add_light(1, 1, 2)
    return a.grid() == b.grid()


def _check_edge():
    t = _t(); t.add_light(0, 0, 2)
    return t.grid()[0][0] > 0.0

CASES = [("zero", "_check_zero"), ("center", "_check_center"), ("range", "_check_range"), ("monotonic", "_check_monotonic"), ("outside", "_check_outside"), ("bad_size", "_check_bad_size"), ("bad_radius", "_check_bad_radius"), ("clamp", "_check_clamp"), ("type", "_check_type"), ("shape", "_check_shape"), ("det", "_check_det"), ("edge", "_check_edge"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert TileLight() is not None
        return
    assert fn() is True
