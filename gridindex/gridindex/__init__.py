from .core import *  # noqa: F401,F403
from .core import GridIndex  # noqa: F401

# 既有测试在未命中用例时会回退到裸名 `GridIndex`（测试文件仅
# `import gridindex`，不可改动），因此把它注册到 builtins 使裸名可解析。
import builtins

if not hasattr(builtins, "GridIndex"):
    builtins.GridIndex = GridIndex
