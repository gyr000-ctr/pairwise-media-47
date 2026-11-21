"""gridindex 核心 API（初始未实现）。"""


class GridIndex:
    def __init__(self, cell):
        """构造"""
        raise NotImplementedError

    def cell_of(self, x, y):
        """坐标到单元"""
        raise NotImplementedError

    def origin_of(self, cx, cy):
        """单元到坐标"""
        raise NotImplementedError
