"""audiogate：双阈值迟滞 + attack/release 指数包络的噪声门。

设计要点：
- 迟滞：电平 >= threshold 开门，电平 <= close_threshold 关门，介于两者之间
  维持当前状态，避免在阈值附近抖动。
- 指数包络：增益按 ``g += (target - g) * (1 - exp(-1 / (tau * fs)))`` 逐样本
  逼近 0/1，attack 与 release 使用不同时间常数；绝不瞬时切换。
- 状态连续：增益与开关状态保存在实例上，``process`` 用局部变量推进，全部
  样本处理成功后才写回，报错时实例状态保持不变（原子性）。
- 无爆音：单样本增益变化有上界；输出差 <= 输入差 + 一个包络步长。
- 错误分类：参数非法 / 样本类型非法 / 样本非有限 三类互不混淆。
- 并发安全：``process`` 全程持锁。
"""

import math
import threading
from numbers import Real

__all__ = [
    "NoiseGate",
    "InvalidParameterError",
    "SampleTypeError",
    "NonFiniteSampleError",
]

# 增益距离目标足够近时吸附到精确值，保证收敛误差为 0 而非浮点渐近误差。
_SNAP_EPS = 1e-12


class InvalidParameterError(ValueError, TypeError):
    """构造参数非法（非数值、非有限、越界或阈值组合矛盾）。"""

    def __init__(self, param, reason):
        self.param = param
        self.reason = reason
        super().__init__("invalid parameter %r: %s" % (param, reason))


class SampleTypeError(ValueError, TypeError):
    """样本类型非法（不是实数，例如字符串）。"""

    def __init__(self, index, kind):
        self.index = index
        self.kind = kind
        super().__init__(
            "sample[%d] has non-numeric type %s" % (index, kind)
        )


class NonFiniteSampleError(ValueError):
    """样本非有限（nan / +inf / -inf）。"""

    def __init__(self, index, kind):
        self.index = index
        self.kind = kind
        super().__init__(
            "sample[%d] is not finite: %s" % (index, kind)
        )


def _real(name, value):
    """校验并转换为 float；非实数或布尔值视为参数非法。"""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InvalidParameterError(name, "must be a real number")
    return float(value)


def _finite_positive(name, value):
    value = _real(name, value)
    if not math.isfinite(value):
        raise InvalidParameterError(name, "must be finite")
    if value <= 0.0:
        raise InvalidParameterError(name, "must be > 0")
    return value


class NoiseGate:
    """峰值检测 + 双阈值迟滞噪声门。"""

    def __init__(
        self,
        threshold,
        release=0.05,
        attack=0.01,
        close_threshold=None,
        sample_rate=48000,
    ):
        threshold = _real("threshold", threshold)
        if not math.isfinite(threshold):
            raise InvalidParameterError("threshold", "must be finite")
        if threshold <= 0.0:
            raise InvalidParameterError("threshold", "must be > 0")

        release = _finite_positive("release", release)
        attack = _finite_positive("attack", attack)
        sample_rate = _finite_positive("sample_rate", sample_rate)

        if close_threshold is None:
            close_threshold = threshold * 0.5
        else:
            close_threshold = _real("close_threshold", close_threshold)
            if not math.isfinite(close_threshold):
                raise InvalidParameterError("close_threshold", "must be finite")
        if close_threshold < 0.0:
            raise InvalidParameterError("close_threshold", "must be >= 0")
        if close_threshold >= threshold:
            raise InvalidParameterError(
                "close_threshold", "must be < threshold"
            )

        self.threshold = threshold
        self.close_threshold = close_threshold
        self.attack = attack
        self.release = release
        self.sample_rate = sample_rate

        # 逐样本指数逼近系数；tau*fs 越大，系数越小、包络越缓。
        self._k_attack = 1.0 - math.exp(-1.0 / (attack * sample_rate))
        self._k_release = 1.0 - math.exp(-1.0 / (release * sample_rate))

        self.gain = 0.0
        self.is_open = False
        self._lock = threading.RLock()

    def process(self, samples):
        """逐样本处理，返回新的 list；不改动输入，不修改既有状态除非成功。"""
        if samples is None or isinstance(samples, (str, bytes)):
            raise SampleTypeError(0, type(samples).__name__)
        try:
            iterator = iter(samples)
        except TypeError:
            raise SampleTypeError(0, type(samples).__name__)

        with self._lock:
            gain = self.gain
            is_open = self.is_open
            target_gain = 0.0
            ka = self._k_attack
            kr = self._k_release
            thr = self.threshold
            close = self.close_threshold
            isfinite = math.isfinite
            out = []
            append = out.append

            for i, v in enumerate(iterator):
                if type(v) is not float:
                    if isinstance(v, bool) or not isinstance(v, Real):
                        raise SampleTypeError(i, type(v).__name__)
                    v = float(v)
                if not isfinite(v):
                    if v != v:
                        kind = "nan"
                    elif v > 0.0:
                        kind = "+inf"
                    else:
                        kind = "-inf"
                    raise NonFiniteSampleError(i, kind)

                level = -v if v < 0.0 else v
                if is_open:
                    if level <= close:
                        is_open = False
                        target_gain = 0.0
                    else:
                        target_gain = 1.0
                elif level >= thr:
                    is_open = True
                    target_gain = 1.0
                else:
                    target_gain = 0.0

                if target_gain == 1.0:
                    if gain != 1.0:
                        gain += (1.0 - gain) * ka
                        if 1.0 - gain < _SNAP_EPS:
                            gain = 1.0
                elif gain != 0.0:
                    gain -= gain * kr
                    if gain < _SNAP_EPS:
                        gain = 0.0

                append(v * gain)

            # 全部成功后才提交状态，保证报错时状态原子不变。
            self.gain = gain
            self.is_open = is_open
            return out
