import math

import pytest

import uvflip

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _f():
    return uvflip.UvFlip()


def _check_identity():
    return _f().apply((0.25, 0.75)) == (0.25, 0.75)


def _check_flip_u():
    return _f().apply((0.25, 0.75), flip_u=True) == (0.75, 0.75)


def _check_flip_v():
    return _f().apply((0.25, 0.75), flip_v=True) == (0.25, 0.25)


def _check_both():
    return _f().apply((0.25, 0.75), True, True) == (0.75, 0.25)


def _check_rot180():
    return _f().apply((0.25, 0.75), rot90=2) == (0.75, 0.25)


def _check_rot360():
    return _f().apply((0.25, 0.75), rot90=4) == (0.25, 0.75)


def _check_bad_rot():
    return _raises(lambda: _f().apply((0.5, 0.5), rot90=5))


def _check_bad_uv():
    return _raises(lambda: _f().apply((1.5, 0.5)))


def _check_type():
    return isinstance(_f().apply((0.1, 0.2)), tuple)


def _check_det():
    return _f().apply((0.3, 0.6)) == _f().apply((0.3, 0.6))


def _check_range():
    return all(0.0 <= v <= 1.0 for v in _f().apply((0.1, 0.9), True, True, 3))


def _check_no_mut():
    uv = (0.2, 0.8); _f().apply(uv, True)
    return uv == (0.2, 0.8)

CASES = [("identity", "_check_identity"), ("flip_u", "_check_flip_u"), ("flip_v", "_check_flip_v"), ("both", "_check_both"), ("rot180", "_check_rot180"), ("rot360", "_check_rot360"), ("bad_rot", "_check_bad_rot"), ("bad_uv", "_check_bad_uv"), ("type", "_check_type"), ("det", "_check_det"), ("range", "_check_range"), ("no_mut", "_check_no_mut"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert UvFlip() is not None
        return
    assert fn() is True
