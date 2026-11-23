"""spring 核心 API：解析解弹簧阻尼系统。

模型（单位质量）：x'' = -k * (x - target) - c * x'
update(dt) 使用闭式解析解推进状态，不依赖固定步长，
任意有限 dt >= 0 均无条件稳定（大 dt 直接衰减到稳态）。
"""

import math
import threading

__all__ = ["Spring"]


def _as_finite(name, value):
    """类型错误 -> TypeError；非有限数值 -> ValueError（可区分报错）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            "%s must be a real number, got %s" % (name, type(value).__name__)
        )
    out = float(value)
    if not math.isfinite(out):
        raise ValueError("%s must be finite, got %r" % (name, out))
    return out


class Spring:
    """弹簧阻尼器。

    初始位置 0.0、初速度 1.0，向 target 收敛。
    stiffness > 0，damping >= 0，target 有限；否则按类型/取值分别报错。
    实例状态彼此独立，update 由实例锁保护，可并发调用。
    """

    def __init__(self, stiffness=100.0, damping=20.0, target=0.0):
        k = _as_finite("stiffness", stiffness)
        if k <= 0.0:
            raise ValueError("stiffness must be > 0, got %r" % (k,))
        c = _as_finite("damping", damping)
        if c < 0.0:
            raise ValueError("damping must be >= 0, got %r" % (c,))
        self._k = k
        self._c = c
        self._target = _as_finite("target", target)
        self._x = 0.0
        self._v = 1.0
        self._lock = threading.Lock()
        # 预计算阻尼 regime 常量（特征方程 r^2 + c r + k = 0）
        disc = c * c - 4.0 * k
        tol = 1e-12 * max(c * c, 4.0 * k)
        if disc > tol:
            # 过阻尼：两个负实根
            beta = math.sqrt(disc) * 0.5
            self._mode = 2
            self._r1 = -c * 0.5 + beta
            self._r2 = -c * 0.5 - beta
            self._inv_dr = 1.0 / (self._r1 - self._r2)
        elif disc < -tol:
            # 欠阻尼（含无阻尼 c=0）：衰减振荡
            self._mode = 1
            self._alpha = c * 0.5
            self._wd = math.sqrt(-disc) * 0.5
        else:
            # 临界阻尼：无过冲
            self._mode = 0
            self._w = c * 0.5 if c > 0.0 else math.sqrt(k)

    def update(self, dt):
        """按解析解步进 dt 秒，返回当前值。"""
        if isinstance(dt, bool) or not isinstance(dt, (int, float)):
            raise TypeError("dt must be a real number, got %s" % type(dt).__name__)
        dt = float(dt)
        if not math.isfinite(dt):
            raise ValueError("dt must be finite, got %r" % (dt,))
        if dt < 0.0:
            raise ValueError("dt must be >= 0, got %r" % (dt,))
        if dt == 0.0:
            return self._x
        with self._lock:
            y0 = self._x - self._target
            v0 = self._v
            mode = self._mode
            if mode == 1:
                a = self._alpha
                w = self._wd
                e = math.exp(-a * dt)
                cos = math.cos(w * dt)
                sin = math.sin(w * dt)
                B = (v0 + a * y0) / w
                self._x = self._target + e * (y0 * cos + B * sin)
                self._v = e * ((B * w - y0 * a) * cos - (y0 * w + B * a) * sin)
            elif mode == 2:
                r1 = self._r1
                r2 = self._r2
                A = (v0 - r2 * y0) * self._inv_dr
                B = y0 - A
                e1 = math.exp(r1 * dt)
                e2 = math.exp(r2 * dt)
                self._x = self._target + A * e1 + B * e2
                self._v = A * r1 * e1 + B * r2 * e2
            else:
                w = self._w
                e = math.exp(-w * dt)
                B = v0 + w * y0
                y = y0 + B * dt
                self._x = self._target + y * e
                self._v = (B - w * y) * e
            return self._x

    def value(self):
        """当前位置（float）。"""
        return float(self._x)
