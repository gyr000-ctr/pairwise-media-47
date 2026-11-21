"""layercache 核心 API（初始未实现）。"""


class LayerCache:
    def __init__(self, budget):
        """构造"""
        raise NotImplementedError

    def get(self, key, compute):
        """获取"""
        raise NotImplementedError

    def bytes_used(self):
        """已用"""
        raise NotImplementedError
