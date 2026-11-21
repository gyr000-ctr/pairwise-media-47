"""colordelta 核心 API（初始未实现）。"""


class ColorDelta:
    def group(self, colors, threshold):
        """分组"""
        raise NotImplementedError

    def delta(self, a, b):
        """色差"""
        raise NotImplementedError
