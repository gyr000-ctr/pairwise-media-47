import builtins
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pytest  # noqa: E402

import layercache  # noqa: E402

# 既有测试的 smoke 用例直接引用裸名 LayerCache（未显式导入），
# 经 builtins 暴露以保持其可解析，不改动测试文件本身。
builtins.LayerCache = layercache.LayerCache


def pytest_configure(config):
    config.addinivalue_line("markers", "cache: layercache 规格用例")


def pytest_collection_modifyitems(items):
    # 为全部用例打上 cache 标记，使验收命令的 -k cache 能选中它们。
    mark = pytest.mark.cache
    for item in items:
        item.add_marker(mark)
