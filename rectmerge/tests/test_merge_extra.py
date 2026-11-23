"""rectmerge 补充测试：合并不变量、错误分类、性能、并发安全。

文件名与所有测试函数均含 ``merge``，可被
``python -m pytest tests -q -k merge`` 选中。既有 test_core.py 不改动。
"""

import math
import random
import threading
import time

import pytest

import rectmerge
from rectmerge import (
    DegenerateRectError,
    NegativeSizeError,
    NonFiniteError,
    RectMerge,
)


def _pairwise_disjoint(out):
    for i, a in enumerate(out):
        ax, ay, aw, ah = a
        for b in out[i + 1:]:
            bx, by, bw, bh = b
            if max(ax, bx) < min(ax + aw, bx + bw) and max(ay, by) < min(
                ay + ah, by + bh
            ):
                return False
    return True


def _brute_area(rects):
    cells = set()
    for x, y, w, h in rects:
        for cx in range(x, x + w):
            for cy in range(y, y + h):
                cells.add((cx, cy))
    return len(cells)


def test_merge_union_covers_and_area_conserved():
    m = RectMerge()
    shapes = [
        [(0, 0, 2, 2), (1, 1, 2, 2)],
        [(1, 0, 1, 3), (0, 1, 3, 1)],
        [(0, 0, 4, 1), (0, 3, 4, 1), (0, 1, 1, 2), (3, 1, 1, 2)],
        [(0, 0, 1, 1), (1, 0, 1, 2), (2, 0, 1, 3)],
        [(0, 0, 3, 1), (1, 1, 1, 1), (2, 2, 1, 1)],
    ]
    for rs in shapes:
        out = m.merge(rs)
        # 并集等价：输出自身的扫描线并集 == 输入并集（面积守恒）。
        assert math.isclose(m.area(out), m.area(rs), rel_tol=1e-12, abs_tol=1e-12)


def test_merge_disjoint_inputs_stay_disjoint():
    # 输入互不重叠时，输出必须两两不重叠且块数不增加。
    m = RectMerge()
    rs = [(1, 2, 2, 2), (3, 1, 2, 5), (2, 6, 4, 2)]
    out = m.merge(rs)
    assert _pairwise_disjoint(out)
    assert len(out) <= len(rs)
    assert m.area(out) == m.area(rs)


def test_merge_ring_keeps_hole():
    # 真正的环形（外框挖心），空洞必须保留，面积是外框减内洞。
    m = RectMerge()
    ring = [(0, 0, 4, 1), (0, 3, 4, 1), (0, 1, 1, 2), (3, 1, 1, 2)]
    out = m.merge(ring)
    assert m.area(out) == 12  # 16 - 4
    assert _pairwise_disjoint(out)
    # 空洞中心 (2, 2) 不得落在任何输出矩形内部。
    hx = hy = 2.0
    for x, y, w, h in out:
        assert not (x < hx < x + w and y < hy < y + h)


def test_merge_deterministic():
    m = RectMerge()
    rs = [(0, 0, 2, 2), (1, 1, 2, 2), (3, 0, 1, 1)]
    first = m.merge(rs)
    for _ in range(5):
        assert m.merge(rs) == first
    assert m.area(rs) == m.area(rs)


def test_merge_does_not_mutate_input():
    m = RectMerge()
    rs = [(0, 0, 2, 2), (1, 1, 2, 2)]
    snapshot = list(rs)
    m.area(rs)
    m.merge(rs)
    assert rs == snapshot


def test_merge_empty_and_single():
    m = RectMerge()
    assert m.merge([]) == []
    assert m.merge([(0, 0, 2, 2)]) == [(0, 0, 2, 2)]


def test_merge_returns_list_of_tuples():
    out = RectMerge().merge([(0, 0, 2, 2), (5, 5, 1, 1)])
    assert isinstance(out, list)
    assert all(len(r) == 4 for r in out)


def test_merge_negative_size_error_distinct():
    m = RectMerge()
    with pytest.raises(NegativeSizeError):
        m.merge([(0, 0, -1, 2)])
    with pytest.raises(NegativeSizeError):
        m.area([(0, 0, 2, -3)])
    # 负尺寸与退化是可区分的两个异常类型。
    assert NegativeSizeError is not DegenerateRectError
    assert issubclass(NegativeSizeError, ValueError)


def test_merge_degenerate_error_distinct():
    m = RectMerge()
    with pytest.raises(DegenerateRectError):
        m.merge([(0, 0, 0, 2)])
    with pytest.raises(DegenerateRectError):
        m.merge([(0, 0, 2, 0)])
    # area 对退化矩形宽容：按 0 面积处理。
    assert m.area([(0, 0, 0, 2)]) == 0
    assert issubclass(DegenerateRectError, ValueError)


def test_merge_non_finite_error_distinct():
    m = RectMerge()
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(NonFiniteError):
            m.area([(0, 0, bad, 2)])
        with pytest.raises(NonFiniteError):
            m.merge([(0, 0, 2, bad)])
    assert NonFiniteError is not NegativeSizeError
    assert issubclass(NonFiniteError, ValueError)


def test_merge_bad_shape_raises_type_error():
    m = RectMerge()
    with pytest.raises(TypeError):
        m.area([(1, 2, 3)])
    with pytest.raises(TypeError):
        m.merge([(1, 2, 3, 4, 5)])


def test_merge_randomized_invariants():
    m = RectMerge()
    random.seed(4242)
    for _ in range(200):
        n = random.randint(1, 10)
        rs = [
            (
                random.randint(0, 8),
                random.randint(0, 8),
                random.randint(1, 5),
                random.randint(1, 5),
            )
            for _ in range(n)
        ]
        ref = _brute_area(rs)
        assert m.area(rs) == ref
        out = m.merge(rs)
        assert math.isclose(
            m.area(out), ref, rel_tol=1e-12, abs_tol=1e-12
        )
        # 输入互不重叠 -> 输出必须互不重叠。
        if _pairwise_disjoint(rs):
            assert _pairwise_disjoint(out)


def test_merge_area_performance_100k_under_400ms():
    random.seed(7)
    n = 100_000
    rects = [
        (
            random.randint(0, 500),
            random.randint(0, 500),
            random.randint(1, 50),
            random.randint(1, 50),
        )
        for _ in range(n)
    ]
    m = RectMerge()
    m.area(rects[:1000])  # 预热
    start = time.perf_counter()
    m.area(rects)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 400.0, elapsed_ms


def test_merge_thread_safety():
    random.seed(1)
    threads_n = 8
    per = 5000
    datasets = [
        [
            (
                random.randint(0, 40),
                random.randint(0, 40),
                random.randint(1, 15),
                random.randint(1, 15),
            )
            for _ in range(per)
        ]
        for _ in range(threads_n)
    ]
    m = RectMerge()
    errors = []
    results = [None] * threads_n

    def worker(idx):
        try:
            results[idx] = (m.area(datasets[idx]), m.merge(datasets[idx]))
        except Exception as exc:  # pragma: no cover - 并发不应抛错
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(threads_n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert all(r is not None for r in results)

    # 同一实例、同一数据并发调用，结果必须逐字节一致（无共享可变状态）。
    shared = datasets[0]
    ref_area = m.area(shared)
    ref_merge = m.merge(shared)
    bag = []
    lock = threading.Lock()

    def shared_worker():
        try:
            a = m.area(shared)
            mg = m.merge(shared)
            with lock:
                bag.append((a, mg))
        except Exception as exc:  # pragma: no cover
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=shared_worker) for _ in range(threads_n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert all(a == ref_area and mg == ref_merge for a, mg in bag)
