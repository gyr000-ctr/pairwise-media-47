"""uvflip 契约测试：仿射合成、正交性、逆往返、结合律、边界与性能。

文件名与用例名都含 ``uv``，因此 ``pytest -k uv`` 会选中它们。
不修改任何既有测试与既有行为。
"""

import itertools
import random
import threading
import time

import pytest

import uvflip

COMBOS = list(itertools.product((False, True), (False, True), range(5)))
IDENTITY = (0.0, 0.0)
I2 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def test_uv_matrix_is_single_affine_and_orthonormal():
    f = uvflip.UvFlip()
    for flip_u, flip_v, rot90 in COMBOS:
        m = f.matrix(flip_u, flip_v, rot90)
        assert len(m) == 6
        assert all(isinstance(x, float) for x in m)
        assert uvflip.orthonormal_error(m) <= 1e-12
        assert uvflip.is_orthogonal(m)
        # 翻转与旋转合成为单一仿射：M = R @ F（先翻后旋）。
        assert m == uvflip.compose(
            uvflip.rotation_affine(rot90), uvflip.flip_affine(flip_u, flip_v)
        )


def test_uv_inverse_roundtrip_within_tolerance():
    f = uvflip.UvFlip()
    rnd = random.Random(20261123)
    for flip_u, flip_v, rot90 in COMBOS:
        m = f.matrix(flip_u, flip_v, rot90)
        inv = f.inverse_matrix(flip_u, flip_v, rot90)
        for _ in range(200):
            uv = (rnd.random(), rnd.random())
            back = uvflip.apply_affine(inv, uvflip.apply_affine(m, uv))
            err = max(abs(back[0] - uv[0]), abs(back[1] - uv[1]))
            err = min(err, abs(err - 1.0))  # 取模后在圆周上度量
            assert err <= 1e-9


def test_uv_compose_is_associative():
    affines = [uvflip.rotation_affine(k) for k in range(4)] + [
        uvflip.flip_affine(True, False),
        uvflip.flip_affine(False, True),
        uvflip.flip_affine(True, True),
    ]
    for a, b, c in itertools.product(affines, repeat=3):
        assert uvflip.compose(uvflip.compose(a, b), c) == uvflip.compose(
            a, uvflip.compose(b, c)
        )
        assert uvflip.compose_all(a, b, c) == uvflip.compose(uvflip.compose(a, b), c)


def test_uv_flip_then_rotate_equals_combined_affine():
    f = uvflip.UvFlip()
    rnd = random.Random(7)
    for flip_u, flip_v, rot90 in COMBOS:
        for _ in range(100):
            uv = (rnd.random(), rnd.random())
            sequential = f.apply(f.apply(uv, flip_u, flip_v), rot90=rot90)
            assert f.apply(uv, flip_u, flip_v, rot90) == sequential


def test_uv_output_strictly_in_unit_interval():
    f = uvflip.UvFlip()
    corners = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    rnd = random.Random(11)
    points = corners + [(rnd.random(), rnd.random()) for _ in range(500)]
    for uv in points:
        for flip_u, flip_v, rot90 in COMBOS:
            u, v = f.apply(uv, flip_u, flip_v, rot90)
            assert 0.0 <= u < 1.0 and 0.0 <= v < 1.0
            assert type((u, v)) is tuple


def test_uv_errors_are_distinguishable():
    f = uvflip.UvFlip()
    with pytest.raises(uvflip.Rot90Error):
        f.apply((0.5, 0.5), rot90=5)
    with pytest.raises(uvflip.Rot90TypeError):
        f.apply((0.5, 0.5), rot90=1.5)
    for bad in (float("nan"), float("inf"), float("-inf"), (float("nan"), 0.5)):
        with pytest.raises(uvflip.NonFiniteUVError):
            f.apply(bad)
    with pytest.raises(uvflip.OutOfRangeUVError):
        f.apply((1.5, 0.5))
    # 三类互不混淆，且都能按 ValueError 捕获（兼容既有用例）。
    assert not issubclass(uvflip.NonFiniteUVError, uvflip.OutOfRangeUVError)
    assert not issubclass(uvflip.OutOfRangeUVError, uvflip.NonFiniteUVError)
    for exc in (uvflip.Rot90Error, uvflip.NonFiniteUVError, uvflip.OutOfRangeUVError):
        assert issubclass(exc, ValueError)


def test_uv_inputs_not_mutated():
    uv = [0.2, 0.8]
    uvflip.UvFlip().apply(uv, True)
    assert uv == [0.2, 0.8]


def test_uv_concurrent_apply_is_consistent():
    f = uvflip.UvFlip()
    expected = {p: f.apply((0.3, 0.7), *p) for p in COMBOS}
    failures = []

    def worker(seed):
        rnd = random.Random(seed)
        for _ in range(1000):
            p = COMBOS[rnd.randrange(len(COMBOS))]
            m = f.matrix(*p)
            uv = (rnd.random(), rnd.random())
            if f.apply(uv, *p) != uvflip.apply_affine(m, uv):
                failures.append(p)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not failures
    assert expected == {p: f.apply((0.3, 0.7), *p) for p in COMBOS}


def test_uv_100k_applies_under_100ms():
    f = uvflip.UvFlip()
    uv = (0.3, 0.7)
    best = float("inf")
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(100_000):
            f.apply(uv)
        best = min(best, time.perf_counter() - start)
    assert best < 0.100, "100k apply took %.1f ms" % (best * 1000.0)
