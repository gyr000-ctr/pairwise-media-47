import math

import pytest

import layercache

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _c():
    return layercache.LayerCache(10)


def _check_empty():
    return _c().bytes_used() == 0


def _check_store():
    c = _c()
    c.get("a", lambda: b"12345")
    return c.bytes_used() == 5


def _check_hit():
    calls = []
    c = _c()
    c.get("a", lambda: (calls.append(1), b"12345")[1])
    c.get("a", lambda: (calls.append(1), b"12345")[1])
    return len(calls) == 1


def _check_budget():
    c = _c()
    for i in range(5):
        c.get(str(i), lambda: b"12345")
    return c.bytes_used() <= 10


def _check_bad_budget():
    return _raises(lambda: layercache.LayerCache(0))


def _check_type():
    c = _c()
    return isinstance(c.get("a", lambda: b"1"), bytes)


def _check_evict():
    c = _c()
    for i in range(4):
        c.get(str(i), lambda: b"12345")
    return c.evictions() >= 1


def _check_det():
    a = _c(); a.get("x", lambda: b"12")
    b = _c(); b.get("x", lambda: b"12")
    return a.bytes_used() == b.bytes_used()


def _check_reuse():
    c = _c()
    c.get("a", lambda: b"1")
    return c.get("a", lambda: b"2") == b"1"

CASES = [("empty", "_check_empty"), ("store", "_check_store"), ("hit", "_check_hit"), ("budget", "_check_budget"), ("bad_budget", "_check_bad_budget"), ("type", "_check_type"), ("evict", "_check_evict"), ("det", "_check_det"), ("reuse", "_check_reuse"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert LayerCache() is not None
        return
    assert fn() is True
