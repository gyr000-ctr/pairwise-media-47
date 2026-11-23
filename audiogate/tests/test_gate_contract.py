"""噪声门契约测试（新增文件，未改动既有测试）。

覆盖题面全部约束：双阈值迟滞、指数 attack/release 包络、跨缓冲区连续、
相邻样本差上界、增益收敛误差、百万样本性能、错误分类与原子性、并发安全。
"""

import math
import threading
import time

import pytest

import audiogate

FS = 48000


def gate(**kw):
    kw.setdefault("threshold", 0.1)
    return audiogate.NoiseGate(**kw)


def test_gate_hysteresis_holds_state_between_thresholds():
    g = gate(threshold=0.5)
    g.process([0.6] * 20)
    assert g.is_open
    assert min(g.process([0.3] * 500)) > 0.0
    assert g.is_open
    g.process([0.25] * 500)
    assert not g.is_open
    g.process([0.3] * 500)
    assert not g.is_open
    g.process([0.5])
    assert g.is_open


def test_gate_attack_is_exponential_and_not_instant():
    out = gate(attack=0.01).process([0.5] * 480)
    assert 0.0 < out[0] < 0.01
    assert out[-1] > out[0]
    assert all(a <= b for a, b in zip(out, out[1:]))
    # 逐样本增量应匹配 exp 包络，而不是一步到位
    k = 1.0 - math.exp(-1.0 / (0.01 * FS))
    assert out[0] == pytest.approx(0.5 * k, rel=0, abs=1e-15)


def test_gate_gain_converges_within_tolerance():
    g = gate()
    g.process([0.5] * FS)
    assert abs(g.gain - 1.0) <= 1e-6
    g.process([0.0] * FS * 5)
    assert abs(g.gain - 0.0) <= 1e-6


def test_gate_state_continuous_across_buffers():
    samples = [math.sin(i / 7.0) for i in range(5000)]
    whole = gate(threshold=0.2).process(samples)
    chunked, g = [], gate(threshold=0.2)
    for i in range(0, len(samples), 137):
        chunked.extend(g.process(samples[i:i + 137]))
    assert chunked == whole


def test_gate_no_pops_sample_delta_bounded():
    attack = release = 0.05
    kg = 1.0 - math.exp(-1.0 / (release * FS))
    samples = [1.0] * 4800 + [0.09] * 48000
    out = gate(attack=attack, release=release).process(samples)
    out_delta = max(abs(b - a) for a, b in zip(out, out[1:]))
    in_delta = max(abs(b - a) for a, b in zip(samples, samples[1:]))
    assert out_delta <= in_delta + kg + 1e-12

    const = gate(attack=attack, release=release).process([1.0] * 2000)
    assert max(abs(b - a) for a, b in zip(const, const[1:])) <= kg + 1e-15


def test_gate_million_samples_under_300ms():
    signal = [((i * 2654435761) % 2001 - 1000) / 1000.0
              for i in range(1_000_000)]
    start = time.perf_counter()
    out = gate().process(signal)
    elapsed = time.perf_counter() - start
    assert len(out) == len(signal)
    assert elapsed < 0.3


def test_gate_parameter_errors_are_classified():
    for bad in (lambda: audiogate.NoiseGate(0.0),
                lambda: audiogate.NoiseGate("x"),
                lambda: audiogate.NoiseGate(float("nan")),
                lambda: audiogate.NoiseGate(float("inf")),
                lambda: audiogate.NoiseGate(0.1, release=-1),
                lambda: audiogate.NoiseGate(0.1, attack=0),
                lambda: audiogate.NoiseGate(0.1, close_threshold=0.2)):
        with pytest.raises(audiogate.InvalidParameterError):
            bad()


@pytest.mark.parametrize("value,kind", [
    (float("nan"), "nan"),
    (float("inf"), "+inf"),
    (float("-inf"), "-inf"),
])
def test_gate_non_finite_samples_are_distinct(value, kind):
    with pytest.raises(audiogate.NonFiniteSampleError) as info:
        gate().process([0.5, value])
    assert info.value.kind == kind
    assert info.value.index == 1


def test_gate_sample_type_error():
    with pytest.raises(audiogate.SampleTypeError):
        gate().process(["a"])


def test_gate_error_taxonomy():
    assert issubclass(audiogate.InvalidParameterError, ValueError)
    assert issubclass(audiogate.InvalidParameterError, TypeError)
    assert issubclass(audiogate.SampleTypeError, ValueError)
    assert issubclass(audiogate.SampleTypeError, TypeError)
    assert issubclass(audiogate.NonFiniteSampleError, ValueError)
    assert not issubclass(audiogate.NonFiniteSampleError, TypeError)


def test_gate_state_atomic_on_error():
    g = gate()
    g.process([0.5] * 10)
    before = (g.gain, g.is_open)
    with pytest.raises(audiogate.NonFiniteSampleError):
        g.process([0.5] * 5 + [float("nan")])
    assert (g.gain, g.is_open) == before


def test_gate_concurrent_process_is_safe():
    g = gate()

    def worker():
        for _ in range(300):
            g.process([0.5] * 100)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert math.isfinite(g.gain) and 0.0 <= g.gain <= 1.0


def test_gate_input_untouched_and_list_returned():
    src = [0.5]
    out = gate().process(src)
    assert src == [0.5]
    assert isinstance(out, list)
    assert gate().process([]) == []
    assert gate().process([0.0] * 10) == [0.0] * 10
