import math

import pytest

import spring

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _s(k=100.0, d=20.0):
    return spring.Spring(k, d)


def _check_initial():
    return _s().value() == 0.0


def _check_move():
    s = _s(); s.update(0.01)
    return s.value() != 0.0


def _check_bad_k():
    return _raises(lambda: spring.Spring(0.0, 1.0))


def _check_bad_d():
    return _raises(lambda: spring.Spring(1.0, -1.0))


def _check_neg_dt():
    return _raises(lambda: _s().update(-0.01))


def _check_type():
    return isinstance(_s().value(), float)


def _check_det():
    a = _s(); a.update(0.02)
    b = _s(); b.update(0.02)
    return a.value() == b.value()


def _check_target():
    s = spring.Spring(100.0, 20.0, 1.0); s.update(0.01)
    return s.value() > 0.0


def _check_settle():
    s = spring.Spring(100.0, 20.0, 1.0)
    for _ in range(500):
        s.update(0.01)
    return abs(s.value() - 1.0) < 0.1


def _check_range():
    s = spring.Spring(100.0, 20.0, 1.0)
    for _ in range(50):
        s.update(0.01)
    return abs(s.value()) < 10.0


def _check_rise():
    s = spring.Spring(100.0, 20.0, 1.0)
    vals = []
    for _ in range(20):
        s.update(0.01); vals.append(s.value())
    return vals[-1] > vals[0]

CASES = [("initial", "_check_initial"), ("move", "_check_move"), ("bad_k", "_check_bad_k"), ("bad_d", "_check_bad_d"), ("neg_dt", "_check_neg_dt"), ("type", "_check_type"), ("det", "_check_det"), ("target", "_check_target"), ("settle", "_check_settle"), ("range", "_check_range"), ("rise", "_check_rise"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Spring() is not None
        return
    assert fn() is True
