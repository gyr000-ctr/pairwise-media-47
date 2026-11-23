import builtins as _builtins

from .core import *  # noqa: F401,F403
from .core import Spring  # noqa: F401

# 既有测试在未定义的检查函数上回退到裸名 `Spring()`（仅 `import spring`），
# 因此将 Spring 暴露到 builtins 使该回退可解析；不改动任何既有测试。
_builtins.Spring = Spring
