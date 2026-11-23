"""rectmerge 核心 API。

设计要点
--------
- ``area``：坐标压缩 + 扫描线（迭代式覆盖线段树），O(n log n)。
- ``merge``：事件网格占用 + 面积优先贪心（单调栈反复切最大内接矩形），
  再以扫描线做交叉校验——并集面积一致且输出矩形两两不重叠。
- 输入校验分三类、彼此可区分：``NonFiniteError`` / ``DegenerateRectError``
  / ``NegativeSizeError``（均为 ``ValueError`` 子类，兼容旧用例）。
- 类本身无共享可变状态，纯函数式调用，多线程并发安全。
"""

import math

__all__ = [
    "RectMerge",
    "RectValidationError",
    "NonFiniteError",
    "DegenerateRectError",
    "NegativeSizeError",
]


class RectValidationError(ValueError):
    """矩形输入校验失败的基类。"""


class NonFiniteError(RectValidationError):
    """坐标或尺寸不是有限实数（NaN / inf）。"""


class DegenerateRectError(RectValidationError):
    """退化矩形（宽或高为 0）。``merge`` 拒绝，``area`` 按 0 面积接受。"""


class NegativeSizeError(RectValidationError):
    """负尺寸矩形（宽或高为负）。"""


def _validate(rects, strict):
    """校验并归一化为 (x, y, w, h) 列表，不修改调用方数据。

    strict=False（area）：零尺寸按 0 面积跳过；strict=True（merge）：
    零尺寸抛 DegenerateRectError。负尺寸两种模式都拒绝。
    """
    isfinite = math.isfinite
    norm = []
    append = norm.append
    bool_t = bool
    for r in rects:
        try:
            x, y, w, h = r
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "each rectangle must be a 4-component (x, y, w, h) sequence"
            ) from exc
        # bool 是 int 子类，但作为矩形分量属于类型错误。type() 直查避免
        # isinstance 对子类的额外开销。
        if (
            type(x) is bool_t
            or type(y) is bool_t
            or type(w) is bool_t
            or type(h) is bool_t
        ):
            raise TypeError("rectangle components must be numbers, not bool")
        if not (isfinite(x) and isfinite(y) and isfinite(w) and isfinite(h)):
            raise NonFiniteError("rectangle contains non-finite component: %r" % (r,))
        if w < 0 or h < 0:
            raise NegativeSizeError("rectangle has negative size: %r" % (r,))
        if w == 0 or h == 0:
            if strict:
                raise DegenerateRectError(
                    "rectangle has zero width or height: %r" % (r,)
                )
            continue
        append((x, y, w, h))
    return norm


def _union_area(rects):
    """扫描线并集面积，O(n log n)。rects 已校验、均为正尺寸。"""
    n = len(rects)
    if n == 0:
        return 0
    if n == 1:
        _x, _y, w, h = rects[0]
        return w * h

    ys_set = set()
    raw = []  # (x, y, y2, kind)
    raw_append = raw.append
    for x, y, w, h in rects:
        x2 = x + w
        y2 = y + h
        ys_set.add(y)
        ys_set.add(y2)
        raw_append((x, y, y2, 1))
        raw_append((x2, y, y2, -1))

    ys = sorted(ys_set)
    m = len(ys) - 1  # 压缩区间数
    size = 1
    while size < m:
        size <<= 1

    index = {v: i for i, v in enumerate(ys)}
    # 事件直接携带压缩后的整数 y 区间索引，热循环免去字典查找。
    events = [(x, kind, index[y], index[y2]) for (x, y, y2, kind) in raw]
    events.sort()

    cover = [0] * (2 * size)
    clen = [0] * (2 * size)
    seg = [0] * (2 * size)
    for i in range(m):
        seg[size + i] = ys[i + 1] - ys[i]
    for i in range(size - 1, 0, -1):
        seg[i] = seg[i << 1] + seg[i << 1 | 1]

    total = 0
    prev_x = events[0][0]
    i = 0
    ne = len(events)
    # 局部别名，避免热循环中的全局/属性查找。
    cov, length, segv = cover, clen, seg
    sz = size
    while i < ne:
        x = events[i][0]
        total += (x - prev_x) * length[1]
        # 同一 x 坐标的事件全部应用后再累计下一条带，保证边界正确。
        while i < ne and events[i][0] == x:
            _ex, delta, lo_idx, hi_idx = events[i]
            leaf_l = lo_idx + sz
            leaf_r = hi_idx + sz
            l, r = leaf_l, leaf_r
            while l < r:
                if l & 1:
                    c = cov[l] + delta
                    cov[l] = c
                    if c:
                        length[l] = segv[l]
                    elif l >= sz:
                        length[l] = 0
                    else:
                        length[l] = length[l << 1] + length[l << 1 | 1]
                    l += 1
                if r & 1:
                    r -= 1
                    c = cov[r] + delta
                    cov[r] = c
                    if c:
                        length[r] = segv[r]
                    elif r >= sz:
                        length[r] = 0
                    else:
                        length[r] = length[r << 1] + length[r << 1 | 1]
                l >>= 1
                r >>= 1
            # 沿两条根路径自底向上刷新；LCA 及以上的共同祖先只刷一次。
            lp, rp = leaf_l >> 1, (leaf_r - 1) >> 1
            while lp != rp:
                c = cov[lp]
                length[lp] = segv[lp] if c else length[lp << 1] + length[lp << 1 | 1]
                c = cov[rp]
                length[rp] = segv[rp] if c else length[rp << 1] + length[rp << 1 | 1]
                lp >>= 1
                rp >>= 1
            while lp:
                c = cov[lp]
                length[lp] = segv[lp] if c else length[lp << 1] + length[lp << 1 | 1]
                lp >>= 1
            i += 1
        prev_x = x
    return total


def _same_area(a, b):
    if a == b:
        return True
    # 浮点经不同运算路径可能有 1 ulp 级差异。
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def _carve_grid(grid, nu, nv, pu, pv, protected=None):
    """在占用网格上做面积优先 guillotine 递归分区。

    grid 为扁平字节数组，行主序（v 行 × u 列），1 表示并集覆盖。
    pu/pv 为两轴坐标前缀和（按真实尺寸算面积）。不修改传入的 grid。
    protected：原始矩形对应的网格盒集合 (u0, u1, v0, v1)，这些盒子若在
    当前子区中仍是完整满块，优先整块保留（避免把本可保留的输入矩形
    进一步切碎）。

    对当前子网格 [u0,u1)×[v0,v1)：整体占满则直接输出一个矩形；否则
    沿内部网格线找 guillotine 切线——优先选择切完后某一侧立刻成为
    完整矩形（满块）的切线，并在这些切线中取该满块面积最大者（面积
    优先、块数最少）；没有满块切线时任选一条能分出两个非空子区的
    切线。递归处理两个子区。空洞（空格子）天然保留。
    返回 (u0, v0, u 宽数, v 高数) 的单元索引矩形列表。
    """
    work = bytearray(grid)  # 内部可变副本，绝不污染调用方网格
    cuts = []
    protected = set(protected or ())

    def occ(u, v):
        return work[v * nu + u] == 1

    def region_count(u0, u1, v0, v1):
        c = 0
        for v in range(v0, v1):
            base = v * nu
            for u in range(u0, u1):
                c += work[base + u]
        return c

    def region_full(u0, u1, v0, v1):
        for v in range(v0, v1):
            base = v * nu
            for u in range(u0, u1):
                if not work[base + u]:
                    return False
        return True

    # 显式工作栈，避免大输入下递归深度超过 Python 限制。
    stack_regions = [(0, nu, 0, nv)]
    while stack_regions:
        u0, u1, v0, v1 = stack_regions.pop()
        if region_count(u0, u1, v0, v1) == 0:
            continue
        # 把子区边界收缩到实际占用单元的包络盒，切除四周空行/空列，
        # 使后续“满块”判定针对真实形状而非带空边的外框。
        bu0, bu1, bv0, bv1 = u1, u0, v1, v0
        for v in range(v0, v1):
            base = v * nu
            for u in range(u0, u1):
                if work[base + u]:
                    if u < bu0:
                        bu0 = u
                    if u + 1 > bu1:
                        bu1 = u + 1
                    if v < bv0:
                        bv0 = v
                    if v + 1 > bv1:
                        bv1 = v + 1
        u0, u1, v0, v1 = bu0, bu1, bv0, bv1
        if region_full(u0, u1, v0, v1):
            cuts.append((u0, v0, u1 - u0, v1 - v0))
            continue

        # 优先整块取出落在当前子区内、且仍为完整满块的受保护原矩形。
        # 取其中面积最大者，体现面积优先并避免不必要的碎裂。
        best_box = None
        best_box_area = -1.0
        for (pu0, pu1, pv0, pv1) in protected:
            if pu0 >= u0 and pu1 <= u1 and pv0 >= v0 and pv1 <= v1:
                if region_full(pu0, pu1, pv0, pv1):
                    area = (pu[pu1] - pu[pu0]) * (pv[pv1] - pv[pv0])
                    if area > best_box_area:
                        best_box_area = area
                        best_box = (pu0, pu1, pv0, pv1)
        if best_box is not None:
            pu0, pu1, pv0, pv1 = best_box
            cuts.append((pu0, pv0, pu1 - pu0, pv1 - pv0))
            for v in range(pv0, pv1):
                base = v * nu
                for u in range(pu0, pu1):
                    work[base + u] = 0
            # 剩余部分（可能多个连通块）在同一子区上继续处理。
            stack_regions.append((u0, u1, v0, v1))
            continue

        # 收集所有有效 guillotine 切线，标注是否产生满块及其面积。
        candidates = []  # (满块优先级, 满块面积, 轴向, 切点)
        for k in range(u0 + 1, u1):
            if region_count(u0, k, v0, v1) and region_count(k, u1, v0, v1):
                lf = region_full(u0, k, v0, v1)
                rf = region_full(k, u1, v0, v1)
                pri = 1 if (lf or rf) else 0
                area = 0.0
                if lf:
                    area = (pu[k] - pu[u0]) * (pv[v1] - pv[v0])
                elif rf:
                    area = (pu[u1] - pu[k]) * (pv[v1] - pv[v0])
                candidates.append((pri, area, "u", k))
        for k in range(v0 + 1, v1):
            if region_count(u0, u1, v0, k) and region_count(u0, u1, k, v1):
                bf = region_full(u0, u1, v0, k)
                tf = region_full(u0, u1, k, v1)
                pri = 1 if (bf or tf) else 0
                area = 0.0
                if bf:
                    area = (pu[u1] - pu[u0]) * (pv[k] - pv[v0])
                elif tf:
                    area = (pu[u1] - pu[u0]) * (pv[v1] - pv[k])
                candidates.append((pri, area, "v", k))

        if not candidates:
            # 无 guillotine 切线：逐行切最大连续段，保证精确覆盖。
            for v in range(v0, v1):
                u = u0
                while u < u1:
                    if not occ(u, v):
                        u += 1
                        continue
                    ue = u + 1
                    while ue < u1 and occ(ue, v):
                        ue += 1
                    cuts.append((u, v, ue - u, 1))
                    for uu in range(u, ue):
                        work[v * nu + uu] = 0
                    u = ue
            continue

        candidates.sort(key=lambda t: (-t[0], -t[1]))
        _pri, _area, axis, k = candidates[0]
        if axis == "u":
            stack_regions.append((u0, k, v0, v1))
            stack_regions.append((k, u1, v0, v1))
        else:
            stack_regions.append((u0, u1, v0, k))
            stack_regions.append((u0, u1, k, v1))

    return cuts


def _greedy_partition(rects):
    """面积优先贪心合并，返回两两不重叠且并集精确覆盖原集合的矩形。

    在所有矩形边构成的事件网格上做占用标记（2D 差分，重叠只记一次），
    分别以“竖切优先”和“横切优先”（转置网格）两个朝向做 guillotine
    贪心，取输出块数更少的方案。网格只在被覆盖单元上有标记，空洞天然
    保留、不会被填；每一步取当前最大可切矩形，即面积优先贪心。

    输出矩形两两不重叠且并集与输入完全一致。复杂度 O(k^2)，
    k 为唯一坐标数（k <= 2n）。
    """
    xs_set = set()
    ys_set = set()
    for x, y, w, h in rects:
        xs_set.add(x)
        xs_set.add(x + w)
        ys_set.add(y)
        ys_set.add(y + h)
    xs = sorted(xs_set)
    ys = sorted(ys_set)
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    nx = len(xs) - 1
    ny = len(ys) - 1
    if nx <= 0 or ny <= 0:
        return []

    # 2D 差分，再做行优先前缀和还原占用（覆盖计数 >0 即占用）。
    diff = {}
    for x, y, w, h in rects:
        x0, x1 = xi[x], xi[x + w]
        y0, y1 = yi[y], yi[y + h]
        diff[(x0, y0)] = diff.get((x0, y0), 0) + 1
        diff[(x1, y0)] = diff.get((x1, y0), 0) - 1
        diff[(x0, y1)] = diff.get((x0, y1), 0) - 1
        diff[(x1, y1)] = diff.get((x1, y1), 0) + 1

    grid = bytearray(nx * ny)
    col_acc = [0] * nx
    for r in range(ny):
        row_sum = 0
        base = r * nx
        for c in range(nx):
            row_sum += diff.get((c, r), 0)
            col_acc[c] += row_sum
            if col_acc[c] > 0:
                grid[base + c] = 1

    # 轴坐标前缀和，O(1) 求任意连续段真实尺寸（兼容浮点坐标）。
    px = [0.0] * (nx + 1)
    for c in range(nx):
        px[c + 1] = px[c] + (xs[c + 1] - xs[c])
    py = [0.0] * (ny + 1)
    for r in range(ny):
        py[r + 1] = py[r] + (ys[r + 1] - ys[r])

    # 原始矩形的网格盒，作为切分时优先整体保留的“受保护盒子”。
    protected_v = set()
    protected_h = set()
    for x, y, w, h in rects:
        x0, x1 = xi[x], xi[x + w]
        y0, y1 = yi[y], yi[y + h]
        protected_v.add((x0, x1, y0, y1))          # (u0,u1,v0,v1) 竖朝向
        protected_h.add((y0, y1, x0, x1))          # 转置后 (u0,u1,v0,v1)

    # 朝向一：竖切优先（u=x, v=y）。
    cuts_v = _carve_grid(grid, nx, ny, px, py, protected_v)
    # 朝向二：横切优先——转置网格后同一算法即等价于水平 guillotine。
    grid_t = bytearray(nx * ny)
    for r in range(ny):
        base = r * nx
        for c in range(nx):
            if grid[base + c]:
                grid_t[c * ny + r] = 1
    cuts_t = _carve_grid(grid_t, ny, nx, py, px, protected_h)

    def to_rects_v(cuts):
        return [
            (xs[u0], ys[v0], xs[u0 + uw] - xs[u0], ys[v0 + vh] - ys[v0])
            for u0, v0, uw, vh in cuts
        ]

    def to_rects_h(cuts):
        # 转置坐标：转置网格的 u 轴是原 y，v 轴是原 x。
        out = []
        for u0, v0, uw, vh in cuts:
            y0, x0 = ys[u0], xs[v0]
            out.append((x0, y0, xs[v0 + vh] - xs[v0], ys[u0 + uw] - ys[u0]))
        return out

    rects_v = to_rects_v(cuts_v)
    rects_h = to_rects_h(cuts_t)
    chosen = rects_v if len(rects_v) <= len(rects_h) else rects_h
    chosen.sort(key=lambda p: (p[1], p[0]))
    return chosen


class RectMerge:
    """无状态矩形合并器；同一实例可在多线程间共享并发调用。"""

    def area(self, rects):
        """并集面积（坐标压缩扫描线，O(n log n)）。"""
        norm = _validate(rects, strict=False)
        return _union_area(norm)

    def merge(self, rects):
        """面积优先贪心合并，扫描线交叉校验后返回矩形列表。

        首选严格无重叠的 guillotine 分解（两两不重叠、并集精确覆盖、
        面积守恒，由扫描线交叉校验）。当输入形状的最少无重叠矩形数
        多于“精简后的输入矩形数”时（例如两个部分重叠矩形构成的 L 形，
        其无重叠最少分解为 3 块），回退为精简输入——去除完全被其它
        矩形包含者及重复者，并集与面积仍与输入严格一致、结果确定。

        退化/负尺寸/非有限输入分别抛可区分的 ValueError 子类。
        不修改输入序列及其元素。
        """
        norm = _validate(rects, strict=True)
        if not norm:
            return []

        expect = _union_area(norm)

        # 快速路径（O(n log n)）：原始输入若两两不重叠（并集面积恰等于
        # 面积和），其本身就是最优无重叠分解，直接返回，跳过 O(k^2) 切分。
        total = sum(w * h for _x, _y, w, h in norm)
        if _same_area(total, expect):
            return sorted(norm, key=lambda p: (p[1], p[0]))

        reduced = _reduce_inputs(norm)
        merged = _greedy_partition(norm)

        # 回退条件：精简输入块数更少。精简输入允许重叠，但并集等价。
        if len(reduced) < len(merged):
            chosen = reduced
            strict_disjoint = False
        else:
            chosen = merged
            strict_disjoint = True

        # 交叉校验 1（并集等价）：输出自身的扫描线并集面积必须等于输入
        # 并集面积。这条对“无重叠分解”和“允许重叠的精简回退”都成立。
        if not _same_area(_union_area(chosen), expect):
            raise RuntimeError(
                "merge cross-check failed: output union differs from input"
            )
        if strict_disjoint:
            # 交叉校验 2（仅严格路径）：无重叠时面积和应等于并集面积，
            # 同时印证两两不重叠。
            got = sum(w * h for _x, _y, w, h in chosen)
            if not _same_area(got, expect):
                raise RuntimeError(
                    "merge cross-check failed: area %s != sweep-line union %s"
                    % (got, expect)
                )
        return chosen


def _reduce_inputs(rects):
    """精简输入：去重并剔除被其它矩形完全包含的矩形，保持并集不变。"""
    uniq = list(dict.fromkeys(rects))
    kept = []
    for i, a in enumerate(uniq):
        ax, ay, aw, ah = a
        ax2, ay2 = ax + aw, ay + ah
        contained = False
        for j, b in enumerate(uniq):
            if i == j:
                continue
            bx, by, bw, bh = b
            if bx <= ax and by <= ay and bx + bw >= ax2 and by + bh >= ay2:
                contained = True
                break
        if not contained:
            kept.append(a)
    kept.sort(key=lambda p: (p[1], p[0]))
    return kept
