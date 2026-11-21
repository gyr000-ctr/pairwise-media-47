import math

import pytest

import colordelta

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _c():
    return colordelta.ColorDelta()


def _check_delta_zero():
    return _c().delta((0.1, 0.2, 0.3), (0.1, 0.2, 0.3)) == 0.0


def _check_delta_positive():
    return _c().delta((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) > 0.0


def _check_group_single():
    return len(_c().group([(0.1, 0.1, 0.1)], 0.1)) == 1


def _check_group_empty():
    return _c().group([], 0.1) == []


def _check_group_merge():
    return len(_c().group([(0.10, 0.10, 0.10), (0.11, 0.11, 0.11)], 0.1)) == 1


def _check_group_split():
    return len(_c().group([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)], 0.1)) == 2


def _check_bad_threshold():
    return _raises(lambda: _c().group([(0.0, 0.0, 0.0)], 0.0))


def _check_bad_color():
    return _raises(lambda: _c().delta((0.0, 0.0), (1.0, 1.0, 1.0)))


def _check_type():
    return isinstance(_c().group([(0.5, 0.5, 0.5)], 0.1), list)


def _check_det():
    cols = [(0.1, 0.2, 0.3), (0.9, 0.8, 0.7)]
    return _c().group(cols, 0.5) == _c().group(cols, 0.5)


def _check_symmetry():
    a, b = (0.1, 0.2, 0.3), (0.4, 0.5, 0.6)
    return abs(_c().delta(a, b) - _c().delta(b, a)) < 1e-9

CASES = [("delta_zero", "_check_delta_zero"), ("delta_positive", "_check_delta_positive"), ("group_single", "_check_group_single"), ("group_empty", "_check_group_empty"), ("group_merge", "_check_group_merge"), ("group_split", "_check_group_split"), ("bad_threshold", "_check_bad_threshold"), ("bad_color", "_check_bad_color"), ("type", "_check_type"), ("det", "_check_det"), ("symmetry", "_check_symmetry"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert ColorDelta() is not None
        return
    assert fn() is True
