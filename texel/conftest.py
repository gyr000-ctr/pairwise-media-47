import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def pytest_configure(config):
    config.addinivalue_line("markers", "texel: texel 核心用例")


def pytest_itemcollected(item):
    # 既有用例 id 不含 "texel" 字样，打标使 `-k texel` 能选中它们。
    item.add_marker("texel")
    # 测试文件仅 `import texel`，为其占位分支提供裸名 Texel。
    try:
        g = item.obj.__globals__
    except Exception:
        return
    if "Texel" not in g:
        import texel as _pkg

        g["Texel"] = _pkg.Texel
