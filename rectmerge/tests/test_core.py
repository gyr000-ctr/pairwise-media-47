import math

import pytest

import rectmerge

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

R1 = (0, 0, 2, 2)
R2 = (1, 1, 2, 2)


def _m():
    return rectmerge.RectMerge()


def _check_single():
    return _m().area([R1]) == 4


def _check_disjoint():
    return _m().area([R1, (5, 5, 1, 1)]) == 5


def _check_overlap():
    return _m().area([R1, R2]) == 7


def _check_duplicate():
    return _m().area([R1, R1]) == 4


def _check_contained():
    return _m().area([(0, 0, 4, 4), (1, 1, 1, 1)]) == 16


def _check_empty():
    return _m().area([]) == 0


def _check_bad():
    return _raises(lambda: _m().area([(0, 0, -1, 2)]))


def _check_zero():
    return _m().area([(0, 0, 0, 2)]) == 0


def _check_type():
    return isinstance(_m().merge([R1]), list)


def _check_merge_count():
    return len(_m().merge([R1, R2])) <= 2


def _check_det():
    return _m().area([R1, R2]) == _m().area([R1, R2])


def _check_no_mut():
    rs = [R1, R2]
    _m().area(rs)
    return rs == [R1, R2]


def _check_three():
    return _m().area([(0, 0, 1, 1), (1, 0, 1, 1), (2, 0, 1, 1)]) == 3


def _check_large():
    return _m().area([(0, 0, 100, 100)]) == 10000


def _check_float():
    return isinstance(_m().area([R1]) * 1.0, float)

CASES = [("single", "_check_single"), ("disjoint", "_check_disjoint"), ("overlap", "_check_overlap"), ("duplicate", "_check_duplicate"), ("contained", "_check_contained"), ("empty", "_check_empty"), ("bad", "_check_bad"), ("zero", "_check_zero"), ("type", "_check_type"), ("merge_count", "_check_merge_count"), ("det", "_check_det"), ("no_mut", "_check_no_mut"), ("three", "_check_three"), ("large", "_check_large"), ("float", "_check_float")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
