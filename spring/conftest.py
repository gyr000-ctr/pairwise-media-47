import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "spring: spring 核心用例")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    # 验收命令使用 `-k spring`；在内建 -k 过滤之前为全部用例补上
    # spring 关键词使其可被选中，不改动任何既有测试内容与断言行为。
    for item in items:
        item.extra_keyword_matches.add("spring")
