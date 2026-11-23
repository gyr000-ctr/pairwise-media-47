"""layercache 核心 API：带字节预算的精确 LRU 图层缓存。

- 精确 LRU：命中即提升为最近使用，淘汰最久未用项。
- 字节预算：缓存总字节数永不超过 budget，超限即按 LRU 顺序淘汰。
- 单飞（singleflight）：同一 key 的并发 get 只计算一次，等待者共享结果。
- 淘汰后可重算：被淘汰的 key 再次 get 时重新计算，确定性 compute 结果一致。
- 非法预算可区分：类型错误抛 TypeError，非正数抛 ValueError。
"""

from __future__ import annotations

import threading
from collections import OrderedDict

__all__ = ["LayerCache"]

_DEFAULT_BUDGET = 1 << 20  # 1 MiB


class _Flight:
    """一次进行中的计算，等待者通过 event 共享其结果。"""

    __slots__ = ("event", "value", "failed")

    def __init__(self):
        self.event = threading.Event()
        self.value = None
        self.failed = False


class LayerCache:
    """按字节预算淘汰的线程安全 LRU 缓存。"""

    def __init__(self, budget=_DEFAULT_BUDGET):
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise TypeError(
                f"budget must be an int, got {type(budget).__name__}"
            )
        if budget <= 0:
            raise ValueError(f"budget must be a positive int, got {budget}")
        self._budget = budget
        self._items = OrderedDict()  # key -> bytes，首端最久未用
        self._used = 0
        self._evictions = 0
        self._inflight = {}  # key -> _Flight
        self._lock = threading.Lock()

    def get(self, key, compute):
        """返回 key 对应的字节串；未命中时调用 compute() 计算并缓存。"""
        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)  # 命中提升优先级
                return value
            flight = self._inflight.get(key)
            if flight is None:
                flight = _Flight()
                self._inflight[key] = flight
                leader = True
            else:
                leader = False

        if not leader:
            # 单飞等待者：共享领头者的计算结果，绝不重复计算
            flight.event.wait()
            if not flight.failed:
                return flight.value
            return self.get(key, compute)  # 领头者失败，重试

        try:
            value = compute()
        except BaseException:
            with self._lock:
                if self._inflight.get(key) is flight:
                    del self._inflight[key]
                flight.failed = True
                flight.event.set()
            raise

        with self._lock:
            if self._inflight.get(key) is flight:
                del self._inflight[key]
            size = len(value)
            if size <= self._budget:
                self._items[key] = value
                self._used += size
                self._evict_locked()
            flight.value = value
            flight.event.set()
        return value

    def _evict_locked(self):
        """按 LRU 顺序淘汰，直到字节用量回到预算内（调用方须持锁）。"""
        items = self._items
        while self._used > self._budget:
            _, old = items.popitem(last=False)
            self._used -= len(old)
            self._evictions += 1

    def bytes_used(self):
        """当前缓存占用的字节数。"""
        with self._lock:
            return self._used

    def evictions(self):
        """累计淘汰次数。"""
        with self._lock:
            return self._evictions
