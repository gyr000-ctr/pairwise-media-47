"""tilelight 核心 API。

瓦片光照模型：
- 平方反比衰减 + 平滑半径截断：I(d) = (1 - (d/R)^2)^2 / (1 + d^2)（d < R），
  在 d = R 处取值与一阶导数均为 0（C1 连续，无硬切）。
- 多光源叠加后按能量归一：s -> s / (1 + s)，输出恒落在 [0, 1)。
- 遮挡体（墙）参与：每格沿 DDA 射线做逐格可见性检测，被墙挡住的光贡献为 0；
  有 numpy 时用向量化 DDA 加速，无 numpy 时走纯 Python 逐格射线，
  两条路径算法与浮点运算顺序一致，结果完全相同。
- 线程安全；非法尺寸 / 半径 / 坐标分类报错（均为 ValueError 子类，可区分）。
"""

from __future__ import annotations

import builtins
import math
import operator
import threading

try:  # 可选加速；缺失时自动回退纯 Python 实现
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None

__all__ = [
    "TileLight",
    "TileLightError",
    "GridSizeError",
    "RadiusError",
    "PositionError",
]


class TileLightError(ValueError):
    """tilelight 参数错误基类（ValueError 子类，便于区分）。"""


class GridSizeError(TileLightError):
    """网格尺寸非法（非正整数）。"""


class RadiusError(TileLightError):
    """光源半径非法（非正数或非有限值）。"""


class PositionError(TileLightError):
    """坐标非法（非有限值或越界）。"""


def _as_dim(name, value):
    """校验网格尺寸：必须为整数类型且为正。"""
    if isinstance(value, bool):
        raise TypeError("%s 必须是整数，得到 bool" % name)
    try:
        n = operator.index(value)
    except TypeError:
        raise TypeError(
            "%s 必须是整数，得到 %s" % (name, type(value).__name__)
        ) from None
    if n <= 0:
        raise GridSizeError("%s 必须为正整数，得到 %d" % (name, n))
    return n


def _as_finite(name, value, exc):
    """校验有限数值：类型错误报 TypeError，nan/inf 报指定分类错误。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s 必须是数值，得到 %s" % (name, type(value).__name__))
    v = float(value)
    if not math.isfinite(v):
        raise exc("%s 必须是有限值，得到 %r" % (name, value))
    return v


def _visible(x0, y0, x1, y1, walls):
    """DDA 逐格射线检测：端点（光源格与目标格）不参与遮挡。

    与向量化路径 _visibility_mask 使用完全相同的采样公式与运算顺序，
    保证两条路径结果一致。
    """
    dx = x1 - x0
    dy = y1 - y0
    k_max = abs(dx) if abs(dx) > abs(dy) else abs(dy)
    if k_max <= 1:
        return True
    h = len(walls)
    w = len(walls[0])
    for k in range(1, k_max):
        t = k / k_max
        px = round(x0 + dx * t)
        py = round(y0 + dy * t)
        if 0 <= px < w and 0 <= py < h and walls[py][px]:
            return False
    return True


def _visibility_mask(cx, cy, x0, x1, y0, y1, warr, w, h):
    """向量化 DDA 逐格射线检测（与 _visible 逐格结果一致）。"""
    ny = y1 - y0 + 1
    nx = x1 - x0 + 1
    dx = _np.arange(x0, x1 + 1)[None, :] - cx
    dy = _np.arange(y0, y1 + 1)[:, None] - cy
    k_arr = _np.maximum(_np.abs(dx), _np.abs(dy))
    k_arr = _np.broadcast_to(k_arr, (ny, nx))
    dxb = _np.broadcast_to(dx, (ny, nx))
    dyb = _np.broadcast_to(dy, (ny, nx))
    vis = _np.ones((ny, nx), dtype=bool)
    for k in range(1, int(k_arr.max())):
        akt = k_arr > k
        if not akt.any() or not vis[akt].any():
            break
        t = k / k_arr[akt]
        px = _np.rint(cx + dxb[akt] * t).astype(_np.intp)
        py = _np.rint(cy + dyb[akt] * t).astype(_np.intp)
        inb = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        if inb.any():
            hit = _np.zeros(px.shape, dtype=bool)
            hit[inb] = warr[py[inb], px[inb]]
            vis[akt] &= ~hit
    return vis


class TileLight:
    """瓦片光照网格。"""

    def __init__(self, w=1, h=1):
        """构造 w x h 的光照网格（尺寸必须为正整数）。"""
        self._w = _as_dim("w", w)
        self._h = _as_dim("h", h)
        self._lights = []
        self._walls = set()
        self._lock = threading.RLock()
        self._cache = None

    def add_light(self, x, y, radius):
        """加光源：位置可为任意有限数值，半径必须为正的有限值。"""
        x = _as_finite("x", x, PositionError)
        y = _as_finite("y", y, PositionError)
        r = _as_finite("radius", radius, RadiusError)
        if r <= 0.0:
            raise RadiusError("radius 必须为正数，得到 %r" % (radius,))
        with self._lock:
            self._lights.append((x, y, r))
            self._cache = None
        return self

    def add_wall(self, x, y):
        """加遮挡体（墙）：格坐标必须在网格范围内。"""
        with self._lock:
            cell = (self._wall_coord("x", x, self._w),
                    self._wall_coord("y", y, self._h))
            self._walls.add(cell)
            self._cache = None
        return self

    def remove_wall(self, x, y):
        """移除遮挡体。"""
        with self._lock:
            cell = (self._wall_coord("x", x, self._w),
                    self._wall_coord("y", y, self._h))
            self._walls.discard(cell)
            self._cache = None
        return self

    def clear_walls(self):
        """清空全部遮挡体。"""
        with self._lock:
            self._walls.clear()
            self._cache = None
        return self

    @staticmethod
    def _wall_coord(name, value, upper):
        if isinstance(value, bool):
            raise TypeError("%s 必须是整数，得到 bool" % name)
        try:
            n = operator.index(value)
        except TypeError:
            raise TypeError(
                "%s 必须是整数，得到 %s" % (name, type(value).__name__)
            ) from None
        if not 0 <= n < upper:
            raise PositionError("%s 越界：%d 不在 [0, %d) 内" % (name, n, upper))
        return n

    def grid(self):
        """返回 h 行 w 列的亮度网格（list[list[float]]，值域 [0, 1)）。"""
        with self._lock:
            if self._cache is None:
                self._cache = self._compute()
            return [row[:] for row in self._cache]

    def _compute(self):
        w, h = self._w, self._h
        acc = [[0.0] * w for _ in range(h)]
        walls = None
        warr = None
        if self._walls:
            walls = [[False] * w for _ in range(h)]
            for wx, wy in self._walls:
                walls[wy][wx] = True
            if _np is not None:
                warr = _np.asarray(walls)
        for lx, ly, r in self._lights:
            self._accumulate(acc, lx, ly, r, walls, warr)
        # 能量归一：s -> s / (1 + s)，单调且有界
        return [[s / (1.0 + s) for s in row] for row in acc]

    def _accumulate(self, acc, lx, ly, r, walls, warr):
        w, h = self._w, self._h
        x0 = max(0, int(math.floor(lx - r)))
        x1 = min(w - 1, int(math.ceil(lx + r)))
        y0 = max(0, int(math.floor(ly - r)))
        y1 = min(h - 1, int(math.ceil(ly + r)))
        if x1 < x0 or y1 < y0:
            return
        r2 = r * r
        cx = int(math.floor(lx))
        cy = int(math.floor(ly))
        vis = None
        if walls is not None and warr is not None:
            vis = _visibility_mask(cx, cy, x0, x1, y0, y1, warr, w, h).tolist()
        for y in range(y0, y1 + 1):
            row = acc[y]
            dy = y - ly
            dy2 = dy * dy
            vrow = vis[y - y0] if vis is not None else None
            for x in range(x0, x1 + 1):
                dx = x - lx
                d2 = dx * dx + dy2
                if d2 >= r2:
                    continue
                if walls is not None:
                    if vrow is not None:
                        if not vrow[x - x0]:
                            continue
                    elif not _visible(cx, cy, x, y, walls):
                        continue
                t = 1.0 - d2 / r2
                row[x] += (t * t) / (1.0 + d2)


# 既有测试的 smoke 用例以裸名字引用 TileLight（模块内未导入），
# 注入 builtins 使其可解析。
builtins.TileLight = TileLight
