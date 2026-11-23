"""gridindex 核心 API。

坐标 ↔ 单元互转基于 math.floor，负数与边界（坐标恰好落在单元边界上
时归入右侧/上侧单元）均满足 round-trip；实例构造后不可变，因而并发
安全（无共享可变状态，热路径无锁）。
"""

import math

__all__ = ["GridIndex"]


class GridIndex:
    """均匀网格索引：cell 为单元边长（正有限数）。"""

    __slots__ = ("_cell",)

    def __init__(self, cell=1.0):
        """构造；cell 必须为正且有限，否则 ValueError。"""
        if isinstance(cell, bool) or not isinstance(cell, (int, float)):
            raise ValueError("cell size must be a real number")
        cell = float(cell)
        if not math.isfinite(cell) or cell <= 0.0:
            raise ValueError("cell size must be a positive finite number")
        self._cell = cell

    @property
    def cell(self):
        return self._cell

    @staticmethod
    def _check_coord(v, name):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(v):
            raise ValueError(f"{name} must be finite, got {v!r}")
        return float(v)

    def _index_of(self, v):
        """单轴坐标 → 单元索引，含浮点边界修正，保证 origin <= v < origin+cell。"""
        c = self._cell
        i = math.floor(v / c)
        if (i + 1) * c <= v:
            i += 1
        elif i * c > v:
            i -= 1
        return i

    def cell_of(self, x, y):
        """坐标到单元；非有限坐标报 ValueError（与非法单元尺寸区分）。"""
        x = self._check_coord(x, "x")
        y = self._check_coord(y, "y")
        return (self._index_of(x), self._index_of(y))

    def origin_of(self, cx, cy):
        """单元到坐标（单元原点，即最小角点）。"""
        if isinstance(cx, bool) or isinstance(cy, bool) or not isinstance(cx, int) or not isinstance(cy, int):
            raise TypeError("cell indices must be integers")
        c = self._cell
        return (cx * c, cy * c)

    def cells_in_range(self, x0, y0, x1, y1):
        """区间覆盖查询：返回与闭区间 [x0,x1]×[y0,y1] 相交的单元集合。

        与暴力（对区间内逐点取 cell_of 求并集）一致：端点按 cell_of 的
        归属规则计入，故 hi 恰好落在边界上时包含其所属单元。
        """
        x0 = self._check_coord(x0, "x0")
        y0 = self._check_coord(y0, "y0")
        x1 = self._check_coord(x1, "x1")
        y1 = self._check_coord(y1, "y1")
        if x1 < x0 or y1 < y0:
            raise ValueError("range must satisfy x0<=x1 and y0<=y1")
        cx0, cy0 = self._index_of(x0), self._index_of(y0)
        cx1, cy1 = self._index_of(x1), self._index_of(y1)
        return {(cx, cy) for cx in range(cx0, cx1 + 1) for cy in range(cy0, cy1 + 1)}

    # 别名，便于不同调用方习惯
    cells_covering = cells_in_range
