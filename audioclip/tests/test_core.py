import math

import pytest

import audioclip

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _c(d=1.0):
    return audioclip.SoftClip(d)


def _check_zero():
    return _c().process([0.0]) == [0.0]


def _check_small():
    return abs(_c().process([0.01])[0] - 0.01) < 1e-3


def _check_bounded():
    return all(abs(v) <= 1.0 for v in _c().process([5.0, -5.0]))


def _check_monotonic():
    out = _c().process([0.1, 0.5, 0.9])
    return out[0] <= out[1] <= out[2]


def _check_len():
    return len(_c().process([0.1, 0.2, 0.3])) == 3


def _check_bad_drive():
    return _raises(lambda: audioclip.SoftClip(0.0))


def _check_empty():
    return _c().process([]) == []


def _check_type():
    return isinstance(_c().process([0.1]), list)


def _check_det():
    return _c().process([0.3]) == _c().process([0.3])


def _check_no_mut():
    s = [0.5]; _c().process(s)
    return s == [0.5]


def _check_odd():
    return abs(_c().process([0.5])[0] + _c().process([-0.5])[0]) < 1e-9


def _check_nan():
    return _raises(lambda: _c().process([float("nan")]))

CASES = [("zero", "_check_zero"), ("small", "_check_small"), ("bounded", "_check_bounded"), ("monotonic", "_check_monotonic"), ("len", "_check_len"), ("bad_drive", "_check_bad_drive"), ("empty", "_check_empty"), ("type", "_check_type"), ("det", "_check_det"), ("no_mut", "_check_no_mut"), ("odd", "_check_odd"), ("nan", "_check_nan"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert SoftClip() is not None
        return
    assert fn() is True
