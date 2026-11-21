"""audiogate 核心 API（初始未实现）。"""


class NoiseGate:
    def __init__(self, threshold, release=0.05):
        """构造"""
        raise NotImplementedError

    def process(self, samples):
        """处理"""
        raise NotImplementedError
