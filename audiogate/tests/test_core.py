import math

import pytest

import audiogate

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _g(th=0.1):
    return audiogate.NoiseGate(th)


def _check_silence():
    return _g().process([0.0] * 10) == [0.0] * 10


def _check_below():
    return all(abs(v) < 1e-9 for v in _g().process([0.001] * 10))


def _check_above():
    return any(v != 0.0 for v in _g().process([0.5] * 10))


def _check_len():
    return len(_g().process([0.5, 0.0])) == 2


def _check_bad_threshold():
    return _raises(lambda: audiogate.NoiseGate(0.0))


def _check_empty():
    return _g().process([]) == []


def _check_type():
    return isinstance(_g().process([0.5]), list)


def _check_det():
    return _g().process([0.5] * 5) == _g().process([0.5] * 5)


def _check_no_mut():
    s = [0.5]; _g().process(s)
    return s == [0.5]


def _check_range():
    return all(-1.0 <= v <= 1.0 for v in _g().process([0.5, -0.5]))


def _check_nan():
    return _raises(lambda: _g().process([float("nan")]))

CASES = [("silence", "_check_silence"), ("below", "_check_below"), ("above", "_check_above"), ("len", "_check_len"), ("bad_threshold", "_check_bad_threshold"), ("empty", "_check_empty"), ("type", "_check_type"), ("det", "_check_det"), ("no_mut", "_check_no_mut"), ("range", "_check_range"), ("nan", "_check_nan"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert NoiseGate() is not None
        return
    assert fn() is True
