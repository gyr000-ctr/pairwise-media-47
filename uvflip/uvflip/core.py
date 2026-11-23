"""uvflip 核心 API：UV 翻转/旋转，合成为单一 2x3 仿射矩阵。

仿射表示 ``(a, b, tx, c, d, ty)``，映射::

    u' = a*u + b*v + tx
    v' = c*u + d*v + ty

约束实现：

* 翻转（``flip_u``/``flip_v``）与 90° 整数倍旋转（``rot90``）合成为
  **单一** 2x3 仿射：``M = R(rot90) @ F(flip)``（先翻后旋，即矩阵乘法）。
* 合成矩阵在模块导入时即验证列向量单位正交（误差 <= ``1e-12``）。
* 正交矩阵的逆为转置（``A^-1 = A^T``），模块导入时验证往返误差 <= ``1e-9``。
* 输出对 1 取模，结果严格落在 ``[0, 1)``。
* 非法 ``rot90``、非有限 UV、有限但越界的 UV、以及非数值 UV 分别抛出
  可区分（可 ``except`` 捕获）的 ``ValueError`` 子类。
* 所有矩阵在导入时一次性构建、之后只读，天然并发安全；无实例可变量。

性能：``apply`` 的一次调用只做字典查表 + 4 次乘法/加法，10 万次远低于
100 ms 的上界（实测见交付证据）。
"""

import math
import operator

__all__ = [
    "UvFlip",
    "UvFlipError",
    "Rot90Error",
    "Rot90TypeError",
    "NonFiniteUVError",
    "OutOfRangeUVError",
    "UvFlipTypeError",
    "compose",
    "compose_all",
    "inverse_affine",
    "apply_affine",
    "rotation_affine",
    "flip_affine",
    "orthonormal_error",
    "is_orthogonal",
    "validate_affine",
]

# 正交性误差上限（题面要求）。
_ORTHO_TOL = 1e-12
# 逆变换往返误差上限（题面要求）。
_ROUNDTRIP_TOL = 1e-9

_IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

# 绕 UV 中心 (0.5, 0.5) 旋转 k*90°（逆时针）的 2x3 仿射。
# 全部系数为精确的 0/±1，复合后仍精确，正交性误差为 0。
_ROT = (
    (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),   # 0°:   (u, v) -> (u, v)
    (0.0, -1.0, 1.0, 1.0, 0.0, 0.0),  # 90°:  (u, v) -> (1 - v, u)
    (-1.0, 0.0, 1.0, 0.0, -1.0, 1.0),  # 180°: (u, v) -> (1 - u, 1 - v)
    (0.0, 1.0, 0.0, -1.0, 0.0, 1.0),  # 270°: (u, v) -> (v, 1 - u)
)


class UvFlipError(ValueError):
    """uvflip 的参数错误基类（``ValueError`` 子类，兼容既有用例）。"""


class Rot90Error(UvFlipError):
    """非法 ``rot90``：布尔值，或不在 ``0..4`` 范围内的整数。"""


class Rot90TypeError(Rot90Error, TypeError):
    """``rot90`` 不是整数类型（既可按 ValueError 也可按 TypeError 捕获）。"""


class UvFlipTypeError(UvFlipError, TypeError):
    """``uv`` 不是由两个实数组成的序列。"""


class NonFiniteUVError(UvFlipError, FloatingPointError):
    """``uv`` 含 NaN 或 ±inf（与有限越界、非法 rot90 可区分）。"""


class OutOfRangeUVError(UvFlipError):
    """``uv`` 为有限值但超出 ``[0, 1]`` 区间。"""


def _compose(second, first):
    """仿射复合（矩阵乘法）：返回 ``second @ first``，即先 ``first`` 后 ``second``。"""
    a2, b2, tx2, c2, d2, ty2 = second
    a1, b1, tx1, c1, d1, ty1 = first
    return (
        a2 * a1 + b2 * c1,
        a2 * b1 + b2 * d1,
        a2 * tx1 + b2 * ty1 + tx2,
        c2 * a1 + d2 * c1,
        c2 * b1 + d2 * d1,
        c2 * tx1 + d2 * ty1 + ty2,
    )


def compose(second, first):
    """公开的仿射矩阵乘法：``compose(A, B) = A @ B``（先应用 B，再应用 A）。

    与数学记号一致，因此“先翻转、后旋转”写作
    ``compose(rotation_affine(rot90), flip_affine(flip_u, flip_v))``，
    且组合满足结合律：``compose(compose(a, b), c) == compose(a, compose(b, c))``。
    两个操作数都是 ``(a, b, tx, c, d, ty)`` 形式的 2x3 仿射。
    """
    return _compose(second, first)


def compose_all(*affines):
    """按数学顺序复合多个仿射：``A @ B @ C ...``（最右边的先应用）。"""
    if not affines:
        return _IDENTITY
    result = affines[0]
    for m in affines[1:]:
        result = _compose(result, m)
    return result


def _inverse(m):
    """正交仿射的逆：线性部分转置，平移取 ``-A^T t``。"""
    a, b, tx, c, d, ty = m
    return (a, c, -(a * tx + c * ty), b, d, -(b * tx + d * ty))


def inverse_affine(m):
    """返回 2x3 仿射的逆（正交前提下即转置 + 平移取反）。"""
    return _inverse(m)


def orthonormal_error(m):
    """返回线性部分列向量偏离单位正交的最大误差。"""
    a, b, _tx, c, d, _ty = m
    return max(
        abs(a * a + c * c - 1.0),
        abs(b * b + d * d - 1.0),
        abs(a * b + c * d),
    )


def is_orthogonal(m, tol=_ORTHO_TOL):
    """列向量是否单位正交（误差 <= ``tol``）。"""
    return orthonormal_error(m) <= tol


def _roundtrip_error(m):
    """正逆复合 ``M^-1 @ M`` 与 ``M @ M^-1`` 相对单位阵的最大误差。"""
    inv = _inverse(m)
    err = 0.0
    for product in (_compose(inv, m), _compose(m, inv)):
        for got, want in zip(product, _IDENTITY):
            diff = abs(got - want)
            if diff > err:
                err = diff
    return err


def validate_affine(m, ortho_tol=_ORTHO_TOL, roundtrip_tol=_ROUNDTRIP_TOL):
    """验证 2x3 仿射：列向量单位正交（<=1e-12）且往返误差（<=1e-9）。

    违例抛 ``ArithmeticError``（不是输入错误，故不混入 ``ValueError`` 家族）。
    """
    err = orthonormal_error(m)
    if err > ortho_tol:
        raise ArithmeticError(
            "仿射矩阵列向量非单位正交：误差 %r > %r，矩阵 %r"
            % (err, ortho_tol, m)
        )
    rt = _roundtrip_error(m)
    if rt > roundtrip_tol:
        raise ArithmeticError(
            "仿射逆变换往返误差 %r > %r，矩阵 %r" % (rt, roundtrip_tol, m)
        )
    return m


def rotation_affine(rot90):
    """绕中心旋转 ``rot90*90°`` 的 2x3 仿射（``rot90`` 取 0..4）。"""
    return _ROT[_check_rot90(rot90) % 4]


def flip_affine(flip_u=False, flip_v=False):
    """翻转的 2x3 仿射（``u' = 1-u`` / ``v' = 1-v``）。"""
    fu, tx = (-1.0, 1.0) if flip_u else (1.0, 0.0)
    fv, ty = (-1.0, 1.0) if flip_v else (1.0, 0.0)
    return (fu, 0.0, tx, 0.0, fv, ty)


def _check_rot90(rot90):
    """校验并归一化 ``rot90``，非法时抛可区分的异常。"""
    if rot90.__class__ is int:
        if 0 <= rot90 <= 4:
            return rot90
        raise Rot90Error("invalid rot90: %r (非法 rot90，须为 0..4 的整数)" % (rot90,))
    if isinstance(rot90, bool):
        raise Rot90Error("invalid rot90: %r (非法 rot90，布尔值不是合法的旋转角度)" % (rot90,))
    try:
        value = operator.index(rot90)
    except TypeError:
        raise Rot90TypeError(
            "invalid rot90: %r (非法 rot90，类型 %s 不是整数)"
            % (rot90, type(rot90).__name__)
        ) from None
    if not 0 <= value <= 4:
        raise Rot90Error("invalid rot90: %r (非法 rot90，须为 0..4 的整数)" % (rot90,))
    return value


def _matrix_slow(flip_u, flip_v, rot90):
    """非内置 int 的 ``rot90``（如 numpy 整数）走慢路径。"""
    k = _check_rot90(rot90) % 4
    return _MATRICES[(k, bool(flip_u), bool(flip_v))]


def _wrap(x):
    """对 1 取模并保证严格落在 ``[0, 1)``。

    浮点取模在极小的负值上可能进位为 ``1.0``（如 ``-1e-17 % 1.0 == 1.0``），
    因此把 ``>= 1.0`` 的结果回绕到 ``0.0``。
    """
    r = x % 1.0
    if r < 1.0:
        return r
    return 0.0


def _classify_bad_uv(uv, u, v):
    """对未通过区间检查的 UV 做可区分分类并抛出。"""
    try:
        finite = math.isfinite(u) and math.isfinite(v)
    except TypeError:
        raise UvFlipTypeError(
            "invalid uv: %r (uv 必须由两个实数组成)" % (uv,)
        ) from None
    if not finite:
        raise NonFiniteUVError("non-finite uv: %r (非有限输入：NaN/inf)" % (uv,))
    raise OutOfRangeUVError("uv out of range: %r (有限但超出 [0, 1] 区间)" % (uv,))


def _is_nonfinite_scalar(uv):
    """裸标量（非序列）是否为非有限实数，用于区分 NaN/inf 与类型错误。"""
    try:
        return not math.isfinite(uv)
    except (TypeError, ValueError):
        return False


def apply_affine(m, uv):
    """把 2x3 仿射作用于 UV，结果取模到 ``[0, 1)``。

    只做变换、不做输入区间检查（供组合/逆向验证使用）。
    """
    a, b, tx, c, d, ty = m
    u, v = uv
    return (_wrap(a * u + b * v + tx), _wrap(c * u + d * v + ty))


def _build_tables():
    """导入时一次性构建 5x2x2 张仿射并逐个验证正交性与往返误差。"""
    tables = {}
    for k in range(5):
        rot = _ROT[k % 4]
        for fu in (False, True):
            for fv in (False, True):
                m = _compose(rot, flip_affine(fu, fv))
                validate_affine(m)
                tables[(k, fu, fv)] = m
    return tables


_MATRICES = _build_tables()


class UvFlip:
    """UV 翻转/旋转；翻转与旋转合成为单一 2x3 仿射。

    实例无可变状态，所有矩阵在导入时构建并验证，可安全地在多线程间共享。
    """

    def matrix(self, flip_u=False, flip_v=False, rot90=0):
        """返回合成并已验证的 2x3 仿射 ``(a, b, tx, c, d, ty)``。

        组合关系为 ``M = R(rot90) @ F(flip_u, flip_v)``（先翻后旋）。
        """
        if rot90.__class__ is int:
            if 0 <= rot90 <= 4:
                return _MATRICES[(rot90, bool(flip_u), bool(flip_v))]
            raise Rot90Error("invalid rot90: %r (非法 rot90，须为 0..4 的整数)" % (rot90,))
        return _matrix_slow(flip_u, flip_v, rot90)

    def inverse_matrix(self, flip_u=False, flip_v=False, rot90=0):
        """返回上述仿射的逆矩阵（正交 => 转置 + 平移取反）。"""
        return _inverse(self.matrix(flip_u, flip_v, rot90))

    def apply(self, uv, flip_u=False, flip_v=False, rot90=0):
        """应用合成仿射，结果取模并严格落在 ``[0, 1)``，返回二元 ``tuple``。

        错误分类：

        * 非法 ``rot90`` -> :class:`Rot90Error` / :class:`Rot90TypeError`
        * NaN / ±inf -> :class:`NonFiniteUVError`
        * 有限但越界 -> :class:`OutOfRangeUVError`
        * 非数值 / 长度不为 2 -> :class:`UvFlipTypeError`

        以上均为 ``ValueError`` 的子类，彼此可通过 ``except`` 精确区分。
        """
        # 快路径：内置 int 的 rot90 直接查表并内联展开，避免多余函数调用。
        if rot90.__class__ is int:
            if 0 <= rot90 <= 4:
                a, b, tx, c, d, ty = _MATRICES[(rot90, bool(flip_u), bool(flip_v))]
            else:
                raise Rot90Error("invalid rot90: %r (非法 rot90，须为 0..4 的整数)" % (rot90,))
        else:
            a, b, tx, c, d, ty = _matrix_slow(flip_u, flip_v, rot90)

        try:
            u, v = uv
            in_range = 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0
        except (TypeError, ValueError):
            if _is_nonfinite_scalar(uv):
                raise NonFiniteUVError(
                    "non-finite uv: %r (非有限输入：NaN/inf)" % (uv,)
                ) from None
            raise UvFlipTypeError(
                "invalid uv: %r (uv 必须由两个实数组成)" % (uv,)
            ) from None

        if in_range:
            x = (a * u + b * v + tx) % 1.0
            y = (c * u + d * v + ty) % 1.0
            # 浮点取模在极小负值上可能进位为 1.0，回绕到 0.0 保证 [0, 1)。
            return (x if x < 1.0 else 0.0, y if y < 1.0 else 0.0)
        _classify_bad_uv(uv, u, v)
