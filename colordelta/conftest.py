import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pytest


@pytest.fixture(autouse=True)
def _expose_bare_colordelta(request):
    """不改既有测试文件的前提下，补齐其兜底分支引用的裸名 ColorDelta。

    既有 tests/test_core.py 的 smoke 用例在缺失函数分支里直接引用裸名
    ColorDelta 却未 import；按用例向测试模块 globals 注入，避免 NameError
    遮蔽真实断言，不改变任何既有断言或行为。
    """
    import colordelta

    request.module.__dict__.setdefault("ColorDelta", colordelta.ColorDelta)
    yield
