"""colordelta 核心 API。

- 色差严格采用 CIEDE2000（CIE 159:2004 / Sharma et al. 2005），输入三元组视为
  CIE-Lab 坐标直接代入公式，全程不使用欧氏距离。
- 分组先做锚点贪心，再做强约束修复与分组合并；对规模较小的输入额外用
  DSATUR 分支限界求出精确最小分组数（贪心合并可能停在局部最优）。
- 公共方法只读实例状态，内部缓存由 ``functools.lru_cache`` 保证线程安全，
  因此同一实例可被多线程并发调用。
"""

import functools
import math
import threading

__all__ = ["ColorDelta"]

# CIEDE2000 常量
_K25_7 = 25.0 ** 7
_PI = math.pi
_TWO_PI = 2.0 * math.pi
_DEG30 = math.pi / 6.0
_DEG6 = math.pi / 30.0
_DEG63 = math.radians(63.0)
_DEG275 = math.radians(275.0)
_DEG25 = math.radians(25.0)

# 精确最小组数的输入规模上限与搜索节点预算（确定性，与机器速度无关）
_EXACT_LIMIT = 20
_SEARCH_BUDGET = 50_000


def _validate_color(color):
    """校验 Lab 三元组，统一返回 float 元组；非法输入抛 ValueError。"""
    # 热路径：已是 3 元有限 float 元组时零分配返回（也是缓存键要求的形式）
    if type(color) is tuple and len(color) == 3:
        c0, c1, c2 = color
        if (
            type(c0) is float
            and type(c1) is float
            and type(c2) is float
            and not (c0 - c0 or c1 - c1 or c2 - c2)
        ):
            return color
    try:
        values = tuple(color)
    except TypeError:
        raise ValueError("color must be a sequence of 3 finite numbers") from None
    if len(values) != 3:
        raise ValueError("color must contain exactly 3 channels")
    for value in values:
        if type(value) is bool or not isinstance(value, (int, float)):
            raise ValueError("color channels must be real numbers")
        if not math.isfinite(value):
            raise ValueError("color channels must be finite")
    return (float(values[0]), float(values[1]), float(values[2]))


def _validate_threshold(threshold):
    """阈值校验：类型错误抛 TypeError，取值非法抛 ValueError，两者可区分。"""
    if type(threshold) is bool or not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be a real number")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    return float(threshold)


@functools.lru_cache(maxsize=65536)
def _delta00(lab1, lab2):
    """两个 Lab 颜色的 CIEDE2000 色差（内部使用弧度，避免重复换算）。"""
    if lab1 == lab2:
        return 0.0
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    C_bar = (C1 + C2) * 0.5
    C_bar7 = C_bar ** 7
    G = 0.5 * (1.0 - math.sqrt(C_bar7 / (C_bar7 + _K25_7)))

    k = 1.0 + G
    a1p = k * a1
    a2p = k * a2
    C1p = math.hypot(a1p, b1)
    C2p = math.hypot(a2p, b2)
    h1 = 0.0 if C1p == 0.0 else math.atan2(b1, a1p) % _TWO_PI
    h2 = 0.0 if C2p == 0.0 else math.atan2(b2, a2p) % _TWO_PI

    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p == 0.0 or C2p == 0.0:
        dhp = 0.0
    else:
        dhp = h2 - h1
        if dhp > _PI:
            dhp -= _TWO_PI
        elif dhp < -_PI:
            dhp += _TWO_PI
    dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(dhp * 0.5)

    L_bar = (L1 + L2) * 0.5
    Cp_bar = (C1p + C2p) * 0.5
    if C1p == 0.0 or C2p == 0.0:
        h_bar = h1 + h2
    else:
        h_sum = h1 + h2
        if abs(h1 - h2) > _PI:
            h_bar = (h_sum + _TWO_PI) * 0.5 if h_sum < _TWO_PI else (h_sum - _TWO_PI) * 0.5
        else:
            h_bar = h_sum * 0.5

    T = (
        1.0
        - 0.17 * math.cos(h_bar - _DEG30)
        + 0.24 * math.cos(2.0 * h_bar)
        + 0.32 * math.cos(3.0 * h_bar + _DEG6)
        - 0.20 * math.cos(4.0 * h_bar - _DEG63)
    )
    d_theta = _DEG30 * math.exp(-(((h_bar - _DEG275) / _DEG25) ** 2))
    Cp_bar7 = Cp_bar ** 7
    R_C = 2.0 * math.sqrt(Cp_bar7 / (Cp_bar7 + _K25_7))
    dL50 = L_bar - 50.0
    S_L = 1.0 + 0.015 * dL50 * dL50 / math.sqrt(20.0 + dL50 * dL50)
    S_C = 1.0 + 0.045 * Cp_bar
    S_H = 1.0 + 0.015 * Cp_bar * T
    R_T = -math.sin(2.0 * d_theta) * R_C

    tL = dLp / S_L
    tC = dCp / S_C
    tH = dHp / S_H
    return math.sqrt(tL * tL + tC * tC + tH * tH + R_T * tC * tH)


def _delta(lab1, lab2):
    """对称化入口：同一对颜色无论顺序都命中同一缓存键。"""
    if lab1 > lab2:
        lab1, lab2 = lab2, lab1
    return _delta00(lab1, lab2)


def _greedy_groups(labs, threshold):
    """锚点贪心：只与每组首个成员比较，速度快但可能把可合并的颜色拆开。"""
    groups = []
    anchors = []
    for idx, lab in enumerate(labs):
        for pos, anchor in enumerate(anchors):
            if _delta(lab, anchor) <= threshold:
                groups[pos].append(idx)
                break
        else:
            groups.append([idx])
            anchors.append(lab)
    return groups


def _repair_groups(groups, labs, threshold):
    """强约束修复：保证组内两两色差均不超过阈值。"""
    repaired = []
    for group in groups:
        subgroups = []
        for idx in group:
            for sub in subgroups:
                if all(_delta(labs[idx], labs[j]) <= threshold for j in sub):
                    sub.append(idx)
                    break
            else:
                subgroups.append([idx])
        repaired.extend(subgroups)
    return repaired


def _merge_groups(groups, labs, threshold):
    """合并阶段：可两两合并的组合并，循环到不动点。"""
    groups = [list(group) for group in groups]
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            left = groups[i]
            for j in range(i + 1, len(groups)):
                right = groups[j]
                if all(
                    _delta(labs[x], labs[y]) <= threshold
                    for x in left
                    for y in right
                ):
                    left.extend(right)
                    del groups[j]
                    merged = True
                    break
            if merged:
                break
    return groups


def _conflict_matrix(labs, threshold):
    """冲突位图：两色差超过阈值（不能同组）则相邻。"""
    n = len(labs)
    adj = [0] * n
    for i in range(n):
        li = labs[i]
        for j in range(i + 1, n):
            if _delta(li, labs[j]) > threshold:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    return adj


def _exact_groups(labs, threshold, upper_bound):
    """DSATUR 分支限界求精确最小分组；无改进时返回 None。

    节点预算固定，故结果与运行速度无关、完全确定；预算耗尽时返回已找到的
    最优合法解（若优于上界）。
    """
    n = len(labs)
    adj = _conflict_matrix(labs, threshold)
    degree = [bin(mask).count("1") for mask in adj]

    colors = [-1] * n
    neighbor_colors = [0] * n  # 邻居已用颜色的位图
    best = [upper_bound, None]
    nodes = [0]
    budget = _SEARCH_BUDGET

    def search(colored, used):
        if used >= best[0] or nodes[0] >= budget:
            return
        if colored == n:
            best[0] = used
            best[1] = colors[:]
            return

        # 选择饱和度最高、其次度数最高的未着色顶点（同分取最小下标）
        candidate = -1
        candidate_key = (-1, -1)
        for u in range(n):
            if colors[u] < 0:
                sat = bin(neighbor_colors[u]).count("1")
                key = (sat, degree[u])
                if key > candidate_key:
                    candidate_key = key
                    candidate = u
        vertex = candidate
        nodes[0] += 1

        forbidden = neighbor_colors[vertex]
        mask = adj[vertex]
        for color in range(used):
            if forbidden >> color & 1:
                continue
            colors[vertex] = color
            bit = 1 << color
            added = []
            m = mask
            while m:
                low = m & -m
                other = low.bit_length() - 1
                m ^= low
                if not neighbor_colors[other] >> color & 1:
                    neighbor_colors[other] |= bit
                    added.append(other)
            search(colored + 1, used)
            colors[vertex] = -1
            for other in added:
                neighbor_colors[other] &= ~bit
            if nodes[0] >= budget or used >= best[0]:
                return

        if used + 1 < best[0]:
            colors[vertex] = used
            bit = 1 << used
            added = []
            m = mask
            while m:
                low = m & -m
                other = low.bit_length() - 1
                m ^= low
                if not neighbor_colors[other] >> used & 1:
                    neighbor_colors[other] |= bit
                    added.append(other)
            search(colored + 1, used + 1)
            colors[vertex] = -1
            for other in added:
                neighbor_colors[other] &= ~bit

    # 对称性：第一个被选中的顶点固定用 0 号颜色
    first = max(range(n), key=lambda u: (degree[u], -u))
    colors[first] = 0
    m = adj[first]
    added = []
    while m:
        low = m & -m
        other = low.bit_length() - 1
        m ^= low
        neighbor_colors[other] |= 1
        added.append(other)
    search(1, 1)
    colors[first] = -1
    for other in added:
        neighbor_colors[other] &= ~1

    if best[1] is None:
        return None

    labels = best[1]
    buckets = {}
    for idx, label in enumerate(labels):
        buckets.setdefault(label, []).append(idx)
    return sorted(buckets.values(), key=lambda group: group[0])


class ColorDelta:
    """CIEDE2000 色差与阈值分组。"""

    def __init__(self):
        # 实例不持有可变状态；锁仅用于表明并发语义，可重入以支持扩展
        self._lock = threading.RLock()

    def delta(self, a, b):
        """两个 Lab 颜色之间的 CIEDE2000 色差。"""
        lab_a = _validate_color(a)
        lab_b = _validate_color(b)
        return _delta(lab_a, lab_b)

    def group(self, colors, threshold):
        """按阈值分组，返回 list[list[原始颜色]]，组内两两色差 <= threshold。"""
        threshold = _validate_threshold(threshold)
        try:
            items = list(colors)
        except TypeError:
            raise TypeError("colors must be an iterable of color triples") from None
        if not items:
            return []
        labs = [_validate_color(color) for color in items]
        n = len(labs)
        if n == 1:
            return [list(items)]

        with self._lock:
            groups = _greedy_groups(labs, threshold)
            groups = _repair_groups(groups, labs, threshold)
            groups = _merge_groups(groups, labs, threshold)

            if n <= _EXACT_LIMIT:
                exact = _exact_groups(labs, threshold, len(groups))
                if exact is not None and len(exact) < len(groups):
                    groups = exact

        return [[items[idx] for idx in group] for group in groups]
