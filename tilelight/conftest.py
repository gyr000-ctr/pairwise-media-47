import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    # 验收命令为 `pytest tests -q -k light`：为全部用例补充 "light" 关键字，
    # 使 -k 表达式能选中既有参数化用例（不改变任何测试行为）。
    for item in items:
        item.extra_keyword_matches.add("light")
