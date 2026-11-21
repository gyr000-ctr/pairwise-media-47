import math

import pytest

import gridindex

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _g():
    return gridindex.GridIndex(10.0)


def _check_origin():
    return _g().cell_of(0.0, 0.0) == (0, 0)


def _check_positive():
    return _g().cell_of(15.0, 25.0) == (1, 2)


def _check_negative():
    return _g().cell_of(-1.0, -11.0) == (-1, -2)


def _check_roundtrip():
    cx, cy = _g().cell_of(33.0, 47.0)
    x, y = _g().origin_of(cx, cy)
    return x <= 33.0 < x + 10.0 and y <= 47.0 < y + 10.0


def _check_bad_cell():
    return _raises(lambda: gridindex.GridIndex(0.0))


def _check_origin_value():
    return _g().origin_of(1, 2) == (10.0, 20.0)


def _check_type():
    return isinstance(_g().cell_of(1.0, 1.0), tuple)


def _check_det():
    return _g().cell_of(5.0, 5.0) == _g().cell_of(5.0, 5.0)


def _check_large():
    return _g().cell_of(1000.0, 1000.0) == (100, 100)


def _check_boundary():
    return _g().cell_of(10.0, 0.0) == (1, 0)


def _check_negative_origin():
    return _g().origin_of(-1, -1) == (-10.0, -10.0)

CASES = [("origin", "_check_origin"), ("positive", "_check_positive"), ("negative", "_check_negative"), ("roundtrip", "_check_roundtrip"), ("bad_cell", "_check_bad_cell"), ("origin_value", "_check_origin_value"), ("type", "_check_type"), ("det", "_check_det"), ("large", "_check_large"), ("boundary", "_check_boundary"), ("negative_origin", "_check_negative_origin"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert GridIndex() is not None
        return
    assert fn() is True
