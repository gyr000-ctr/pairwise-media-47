"""CIEDE2000 / 分组的补充验收测试（文件名与用例名均含 delta，命中 -k delta）。"""

import math
import random
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from colordelta.core import ColorDelta


# Sharma et al. (2005) Table I 官方参考数据，输入为 CIE-Lab，期望 CIEDE2000
_SHARMA = [
    ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
    ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
    ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
    ((50.0000, -1.3877, -84.2730), (50.0000, 0.0000, -82.7485), 1.0022),
    ((50.0000, -1.1848, -84.8006), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, -0.9009, -85.5211), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, 0.0000, 0.0000), (50.0000, -1.0000, 2.0000), 2.3669),
    ((50.0000, -1.0000, 2.0000), (50.0000, 0.0000, 0.0000), 2.3669),
    ((50.0000, 2.5000, 0.0000), (73.0000, 25.0000, -18.0000), 27.1492),
    ((61.2901, 1.4479, -47.5412), (61.4292, 1.7471, -44.7768), 1.1412),
    ((2.5000, 0.0200, 0.0000), (2.5000, 0.0000, 0.0000), 0.0300),
    ((50.0000, 2.5000, 0.0000), (61.0000, -5.0000, 29.0000), 22.8977),
    ((50.0000, 2.5000, 0.0000), (56.0000, -27.0000, -3.0000), 31.9030),
    ((50.0000, 2.5000, 0.0000), (58.0000, 24.0000, 15.0000), 19.4535),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.1736, 0.5854), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.2972, 0.0000), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 1.8634, 0.5757), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.2592, 0.3350), 1.0000),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
    ((61.2901, 3.7196, -5.3901), (61.4292, 2.2480, -4.9620), 1.8731),
    ((35.0831, -44.1164, 3.7933), (35.0232, -40.0716, 1.5901), 1.8645),
    ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
    ((36.4612, 47.8580, 18.3852), (36.2715, 50.5065, 21.2231), 1.4146),
    ((90.8027, -2.0831, 1.4410), (91.1528, -1.6435, 0.0447), 1.4441),
    ((90.9257, -0.5406, -0.9208), (88.6381, -0.8985, -0.7239), 1.5381),
    ((6.7747, -0.2908, -2.4247), (5.8714, -0.0985, -2.2286), 0.6377),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
]


@pytest.mark.parametrize("lab1,lab2,expected", _SHARMA)
def test_delta_ciede2000_matches_sharma_reference(lab1, lab2, expected):
    cd = ColorDelta()
    assert cd.delta(lab1, lab2) == pytest.approx(expected, abs=1e-4)
    assert cd.delta(lab2, lab1) == pytest.approx(expected, abs=1e-4)


def test_delta_is_not_euclidean_distance():
    """CIEDE2000 带色度/色相加权，不能与 Lab 欧氏距离等价。"""
    cd = ColorDelta()
    lab1 = (50.0, 2.6772, -79.7751)
    lab2 = (50.0, 0.0, -82.7485)
    euclidean = math.sqrt(sum((x - y) ** 2 for x, y in zip(lab1, lab2)))
    got = cd.delta(lab1, lab2)
    assert got == pytest.approx(2.0425, abs=1e-4)
    assert abs(got - euclidean) > 0.5


def test_delta_public_symmetry_and_zero():
    cd = ColorDelta()
    a = (0.1, 0.2, 0.3)
    b = (80.0, 4.0, -6.0)
    assert cd.delta(a, a) == 0.0
    assert cd.delta(a, b) == cd.delta(b, a)
    assert cd.delta(a, b) > 0.0


@pytest.mark.parametrize("bad", [0.0, -1.0, -0.5, float("nan"), float("inf")])
def test_delta_threshold_value_errors(bad):
    with pytest.raises(ValueError):
        ColorDelta().group([(0.0, 0.0, 0.0)], bad)


@pytest.mark.parametrize("bad", ["0.5", None, True, [0.5], (0.5,)])
def test_delta_threshold_type_errors(bad):
    with pytest.raises(TypeError):
        ColorDelta().group([(0.0, 0.0, 0.0)], bad)


@pytest.mark.parametrize(
    "bad",
    [
        (0.1, 0.2),
        (0.1, 0.2, 0.3, 0.4),
        "rgb",
        None,
        42,
        (float("nan"), 0.0, 0.0),
        (float("inf"), 0.0, 0.0),
        (0.1, float("-inf"), 0.3),
        ("0.1", 0.2, 0.3),
    ],
)
def test_delta_bad_color_value_errors(bad):
    cd = ColorDelta()
    good = (0.1, 0.2, 0.3)
    with pytest.raises(ValueError):
        cd.delta(bad, good)
    with pytest.raises(ValueError):
        cd.delta(good, bad)
    with pytest.raises(ValueError):
        cd.group([good, bad], 0.5)


def _pairwise_ok(cd, colors, groups, threshold):
    for group in groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if cd.delta(group[i], group[j]) > threshold:
                    return False
    flat = sorted(tuple(color) for group in groups for color in group)
    return flat == sorted(tuple(color) for color in colors)


def _min_group_count(cd, colors, threshold):
    """小输入的精确最小组数：冲突图（色差超阈值即相邻）的最小着色数。"""
    n = len(colors)
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            far = cd.delta(colors[i], colors[j]) > threshold
            adj[i][j] = adj[j][i] = far
    color_of = [-1] * n
    best = [n]

    def backtrack(v, used):
        if used >= best[0]:
            return
        if v == n:
            best[0] = used
            return
        for k in range(used):
            if all(not adj[v][w] for w in range(v) if color_of[w] == k):
                color_of[v] = k
                backtrack(v + 1, used)
                color_of[v] = -1
        if used + 1 < best[0]:
            color_of[v] = used
            backtrack(v + 1, used + 1)
            color_of[v] = -1

    backtrack(0, 0)
    return best[0]


def test_delta_group_pairwise_and_minimum_on_small_inputs():
    cd = ColorDelta()
    rng = random.Random(20261122)
    for _ in range(40):
        n = rng.randint(1, 9)
        colors = [
            (rng.random() * 100, rng.random() * 100 - 50, rng.random() * 100 - 50)
            for _ in range(n)
        ]
        threshold = rng.uniform(2.0, 25.0)
        groups = cd.group(colors, threshold)
        assert _pairwise_ok(cd, colors, groups, threshold)
        assert len(groups) == _min_group_count(cd, colors, threshold)


def test_delta_group_merge_beats_anchor_greedy():
    """锚点贪心可能把可合并的颜色拆开，合并阶段必须把组数降到最小。"""
    cd = ColorDelta()
    # A 与 B 接近、B 与 C 接近，但 A 与 C 超过阈值：最小组数应为 2。
    colors = [(50.0, 0.0, 0.0), (50.0, 1.5, 0.0), (50.0, 3.0, 0.0)]
    threshold = 2.0
    groups = cd.group(colors, threshold)
    assert _pairwise_ok(cd, colors, groups, threshold)
    assert len(groups) == _min_group_count(cd, colors, threshold)

    close = [(50.0, 2.0, -3.0), (50.5, 2.1, -3.1), (51.0, 1.9, -2.9)]
    assert len(cd.group(close, 10.0)) == 1


def test_delta_group_pass_through_and_determinism():
    cd = ColorDelta()
    assert cd.group([], 0.5) == []
    assert cd.group([(0.1, 0.1, 0.1)], 0.1) == [[(0.1, 0.1, 0.1)]]
    colors = [(0.1, 0.2, 0.3), (0.9, 0.8, 0.7), (0.11, 0.21, 0.31)]
    first = cd.group(colors, 5.0)
    assert isinstance(first, list)
    for _ in range(5):
        assert cd.group(colors, 5.0) == first


def test_delta_100k_comparisons_under_400ms():
    cd = ColorDelta()
    rng = random.Random(7)
    colors = [(rng.random(), rng.random(), rng.random()) for _ in range(500)]
    pairs = [
        (colors[rng.randrange(500)], colors[rng.randrange(500)])
        for _ in range(100_000)
    ]
    start = time.perf_counter()
    for a, b in pairs:
        cd.delta(a, b)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert elapsed_ms < 400.0, f"10 万次色差耗时 {elapsed_ms:.1f} ms"


def test_delta_concurrent_safety():
    cd = ColorDelta()
    rng = random.Random(11)
    colors = [
        (rng.random() * 100, rng.random() * 100 - 50, rng.random() * 100 - 50)
        for _ in range(80)
    ]
    expected_groups = cd.group(colors, 8.0)
    expected_delta = cd.delta(colors[0], colors[1])

    def worker(_):
        return cd.delta(colors[0], colors[1]), cd.group(colors, 8.0)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(worker, range(64)))
    for delta, groups in results:
        assert delta == expected_delta
        assert groups == expected_groups


def test_delta_group_medium_inputs_legal():
    """规模超过精确求解上限时，仍需保证组内两两合法且颜色不重不漏。"""
    cd = ColorDelta()
    rng = random.Random(13)
    colors = [
        (rng.random() * 100, rng.random() * 100 - 50, rng.random() * 100 - 50)
        for _ in range(120)
    ]
    for threshold in (3.0, 8.0, 20.0):
        groups = cd.group(colors, threshold)
        assert _pairwise_ok(cd, colors, groups, threshold)
