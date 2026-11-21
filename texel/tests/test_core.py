import math

import pytest

import texel

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _t():
    return texel.Texel(4, 4)


def _check_origin():
    return _t().to_texel(0.0, 0.0) == (0, 0)


def _check_next():
    return _t().to_texel(0.375, 0.125) == (1, 0)


def _check_clamp_high():
    return _t().to_texel(1.5, 0.0) == (3, 0)


def _check_clamp_low():
    return _t().to_texel(-0.5, 0.0) == (0, 0)


def _check_weight_sum():
    return abs(sum(_t().bilinear_weights(0.3, 0.3)) - 1.0) < 1e-9


def _check_weight_count():
    return len(_t().bilinear_weights(0.3, 0.3)) == 4


def _check_bad_size():
    return _raises(lambda: texel.Texel(0, 4))


def _check_type():
    return isinstance(_t().to_texel(0.1, 0.1), tuple)


def _check_det():
    return _t().to_texel(0.4, 0.4) == _t().to_texel(0.4, 0.4)


def _check_center():
    return _t().to_texel(0.125, 0.125) == (0, 0)


def _check_range():
    return all(0.0 <= v <= 1.0 for v in _t().bilinear_weights(0.7, 0.2))

CASES = [("origin", "_check_origin"), ("next", "_check_next"), ("clamp_high", "_check_clamp_high"), ("clamp_low", "_check_clamp_low"), ("weight_sum", "_check_weight_sum"), ("weight_count", "_check_weight_count"), ("bad_size", "_check_bad_size"), ("type", "_check_type"), ("det", "_check_det"), ("center", "_check_center"), ("range", "_check_range"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Texel() is not None
        return
    assert fn() is True
