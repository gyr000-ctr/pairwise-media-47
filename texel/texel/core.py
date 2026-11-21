"""texel 核心 API（初始未实现）。"""


class Texel:
    def __init__(self, w, h):
        """构造"""
        raise NotImplementedError

    def to_texel(self, u, v):
        """UV 转 texel"""
        raise NotImplementedError

    def bilinear_weights(self, u, v):
        """权重"""
        raise NotImplementedError
