"""texel 核心 API：UV→texel 映射与双线性/三线性权重。

约定：
- +0.5 中心语义：texel (i, j) 的中心位于 UV ((i+0.5)/w, (j+0.5)/h)，
  最近邻 texel 为 floor(u*w), floor(v*h)，在 texel 中心处与采样结果一致。
- 双线性权重顺序对应 texel (x0,y0), (x1,y0), (x0,y1), (x1,y1)，和恒为 1。
- 三线性权重 = 相邻两级 mip 的双线性权重按 lod 小数部分 t 做 (1-t)/t
  线性组合，即逐级双线性组合，8 个权重和恒为 1。
- 边界模式：clamp（夹取）与 wrap（取模回绕）。
- 实例构造后状态不可变、方法无副作用，天然并发安全。
"""

import math

__all__ = [
    "Texel",
    "TexelError",
    "SizeError",
    "NonFiniteError",
]

_CLAMP = "clamp"
_WRAP = "wrap"
_MODES = (_CLAMP, _WRAP)


class TexelError(ValueError):
    """texel 相关错误基类，可按 ValueError 捕获。"""


class SizeError(TexelError):
    """非法纹理尺寸（非正整数、非有限等）。"""


class NonFiniteError(TexelError):
    """非有限（NaN/Inf）UV 或 lod 输入。"""


def _validate_size(w, h):
    for name, v in (("w", w), ("h", h)):
        if isinstance(v, bool) or not isinstance(v, int):
            raise SizeError("texture %s must be an integer, got %r" % (name, v))
        if v <= 0:
            raise SizeError("texture %s must be positive, got %r" % (name, v))


def _validate_mode(mode):
    if mode not in _MODES:
        raise ValueError("unknown boundary mode: %r" % (mode,))
    return mode


def _check_finite(*vals):
    for v in vals:
        # 非数值类型让 math.isfinite 自然抛 TypeError，与“非有限”区分。
        if not math.isfinite(v):
            raise NonFiniteError("non-finite coordinate: %r" % (v,))


def _bilinear_fracs(u, v, w, h):
    fx = u * w - 0.5
    fy = v * h - 0.5
    fx -= math.floor(fx)
    fy -= math.floor(fy)
    return fx, fy


class Texel:
    __slots__ = ("_w", "_h", "_mode", "_levels")

    def __init__(self, w=1, h=1, mode=_CLAMP):
        """构造：w/h 为正整数 texel 尺寸，mode 为 "clamp" 或 "wrap"。"""
        _validate_size(w, h)
        self._w = w
        self._h = h
        self._mode = _validate_mode(mode)
        # 逐级 mip 尺寸，构造时一次算好并保持不可变。
        levels = []
        lw, lh = w, h
        while True:
            levels.append((lw, lh))
            if lw == 1 and lh == 1:
                break
            lw = max(1, lw >> 1)
            lh = max(1, lh >> 1)
        self._levels = tuple(levels)

    @property
    def w(self):
        return self._w

    @property
    def h(self):
        return self._h

    @property
    def mode(self):
        return self._mode

    @property
    def mip_levels(self):
        return self._levels

    def _resolve(self, i, n, mode):
        if mode == _WRAP:
            return i % n
        return 0 if i < 0 else (n - 1 if i >= n else i)

    def to_texel(self, u, v, mode=None):
        """UV 转最近邻 texel 坐标（+0.5 中心语义），返回 (i, j) 元组。"""
        _check_finite(u, v)
        m = self._mode if mode is None else _validate_mode(mode)
        i = math.floor(u * self._w)
        j = math.floor(v * self._h)
        return (self._resolve(i, self._w, m), self._resolve(j, self._h, m))

    def bilinear_weights(self, u, v):
        """双线性 4 权重，顺序 (x0,y0),(x1,y0),(x0,y1),(x1,y1)，和为 1。"""
        _check_finite(u, v)
        fx, fy = _bilinear_fracs(u, v, self._w, self._h)
        gx = 1.0 - fx
        gy = 1.0 - fy
        return (gx * gy, fx * gy, gx * fy, fx * fy)

    def bilinear_indices(self, u, v, mode=None):
        """双线性 4 个 texel 下标，顺序与 bilinear_weights 对应。"""
        _check_finite(u, v)
        m = self._mode if mode is None else _validate_mode(mode)
        x0 = math.floor(u * self._w - 0.5)
        y0 = math.floor(v * self._h - 0.5)
        x1 = self._resolve(x0 + 1, self._w, m)
        x0 = self._resolve(x0, self._w, m)
        y1 = self._resolve(y0 + 1, self._h, m)
        y0 = self._resolve(y0, self._h, m)
        return ((x0, y0), (x1, y0), (x0, y1), (x1, y1))

    def trilinear_weights(self, u, v, lod):
        """三线性 8 权重：两级 mip 双线性权重按 lod 小数部分逐级组合。"""
        _check_finite(u, v, lod)
        max_level = len(self._levels) - 1
        lod = float(lod)
        if lod < 0.0:
            lod = 0.0
        elif lod > max_level:
            lod = float(max_level)
        l0 = int(math.floor(lod))
        if l0 >= max_level:
            l0 = max_level
            l1 = max_level
            t = 0.0
        else:
            l1 = l0 + 1
            t = lod - l0
        lw, lh = self._levels[l0]
        fx, fy = _bilinear_fracs(u, v, lw, lh)
        gx = 1.0 - fx
        gy = 1.0 - fy
        base = (gx * gy, fx * gy, gx * fy, fx * fy)
        if t == 0.0:
            return base + (0.0, 0.0, 0.0, 0.0)
        lw, lh = self._levels[l1]
        fx, fy = _bilinear_fracs(u, v, lw, lh)
        gx = 1.0 - fx
        gy = 1.0 - fy
        s = 1.0 - t
        return (
            base[0] * s, base[1] * s, base[2] * s, base[3] * s,
            gx * gy * t, fx * gy * t, gx * fy * t, fx * fy * t,
        )
