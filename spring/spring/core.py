"""spring 核心 API（初始未实现）。"""


class Spring:
    def __init__(self, stiffness, damping, target=0.0):
        """构造"""
        raise NotImplementedError

    def update(self, dt):
        """步进"""
        raise NotImplementedError

    def value(self):
        """值"""
        raise NotImplementedError
