"""
geometry/freecad_export.py
==========================
直接调用 FreeCAD Python API 构建三维模型并导出（可选模块）。

依赖：FreeCAD >= 0.21（可选，不影响核心计算和测试流程）。
所有 FreeCAD 导入均在函数体内（lazy import），模块级只允许 importlib 标准库。

禁止在核心测试流程中导入本模块（CLAUDE.md 原则 6、禁止事项）。
所有几何数据均来自各构型 geometry.py 的输出字典，本模块不调用气动设计函数。
"""

from __future__ import annotations

import importlib.util
import os
from typing import List, Tuple


# ---------------------------------------------------------------------------
# FreeCAD 可用性检测
# ---------------------------------------------------------------------------

def is_freecad_available() -> bool:
    """检测 FreeCAD Python 包是否可用。

    Returns
    -------
    bool
        True 若 ``import FreeCAD`` 可成功执行，否则 False。
        本函数永不抛异常。
    """
    return importlib.util.find_spec("FreeCAD") is not None


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def export_to_freecad(
    inlet_type: str,
    geo_result: dict,
    output_path: str,
) -> None:
    """直接调用 FreeCAD API 构建三维模型并导出 .BREP / .STEP 文件。

    Parameters
    ----------
    inlet_type : str
        ``"pitot"`` | ``"external_2d"`` | ``"axisymmetric"``。
    geo_result : dict
        几何字典（同 freecad_script.py 的输入格式）。
    output_path : str
        输出路径，后缀 ``.brep`` 或 ``.step``（不区分大小写）。
        目录不存在时自动创建。

    Raises
    ------
    ImportError
        FreeCAD 不可用时抛出，含安装提示。
    ValueError
        inlet_type 不合法或 geo_result 缺少必要键，或后缀不支持。
    """
    if not is_freecad_available():
        raise ImportError(
            "FreeCAD Python 包不可用。请安装 FreeCAD（>= 0.21）并确保其 Python "
            "绑定路径已加入 sys.path。\n"
            "参考：https://wiki.freecad.org/Embedding_FreeCAD"
        )

    # lazy import — 仅在 FreeCAD 可用时执行
    import FreeCAD  # noqa: F401 (确认可用后再 import Part)
    import Part
    from FreeCAD import Base

    _BUILDERS = {
        "pitot":        _build_shape_pitot,
        "external_2d":  _build_shape_external_2d,
        "axisymmetric": _build_shape_axisymmetric,
    }
    if inlet_type not in _BUILDERS:
        raise ValueError(
            f"inlet_type 须为 'pitot'、'external_2d' 或 'axisymmetric'，"
            f"当前 '{inlet_type}'。"
        )

    ext = os.path.splitext(output_path)[1].lower()
    if ext not in (".brep", ".step", ".stp"):
        raise ValueError(f"output_path 后缀须为 .brep 或 .step/.stp，当前 '{ext}'。")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    shapes = _BUILDERS[inlet_type](geo_result, Part, Base)

    # 若返回多个 shape，合并为 compound
    if isinstance(shapes, (list, tuple)):
        shape = Part.makeCompound(list(shapes))
    else:
        shape = shapes

    if ext == ".brep":
        shape.exportBrep(output_path)
    else:
        shape.exportStep(output_path)


# ---------------------------------------------------------------------------
# 各构型 Shape 构建（内部函数，仅在 FreeCAD 可用时调用）
# ---------------------------------------------------------------------------

def _vec_list(pts: List[Tuple[float, float]], Base) -> list:
    """(x, r/y) 点列表 → FreeCAD Base.Vector 列表（z=0）。"""
    return [Base.Vector(float(x), float(r), 0.0) for x, r in pts]


def _build_shape_pitot(geo: dict, Part, Base):
    """皮托管：BSpline 母线 + Revolution。"""
    if "profile" not in geo:
        raise ValueError("geo_result 缺少 'profile' 键（pitot 构型）。")

    vecs = _vec_list(geo["profile"], Base)
    bs = Part.BSplineCurve()
    bs.interpolate(vecs)
    edge = bs.toShape()

    axis_pt = Base.Vector(0, 0, 0)
    axis_dir = Base.Vector(1, 0, 0)
    return edge.revolve(axis_pt, axis_dir, 360)


def _build_shape_external_2d(geo: dict, Part, Base):
    """二元外压式：上下壁 BSpline + Face + Extrude。"""
    for key in ("profile_upper", "profile_lower"):
        if key not in geo:
            raise ValueError(f"geo_result 缺少 '{key}' 键（external_2d 构型）。")

    vecs_upper = _vec_list(geo["profile_upper"], Base)
    vecs_lower = _vec_list(geo["profile_lower"], Base)

    bs_up = Part.BSplineCurve()
    bs_up.interpolate(vecs_upper)
    edge_up = bs_up.toShape()

    bs_lo = Part.BSplineCurve()
    bs_lo.interpolate(vecs_lower)
    edge_lo = bs_lo.toShape()

    # 前缘竖线
    x_le, y_le_up = geo["profile_upper"][0]
    _, y_le_lo = geo["profile_lower"][0]
    x_ex, y_ex_up = geo["profile_upper"][-1]
    _, y_ex_lo = geo["profile_lower"][-1]

    edge_le = Part.makeLine(
        Base.Vector(x_le, y_le_up, 0),
        Base.Vector(x_le, y_le_lo, 0),
    )
    edge_ex = Part.makeLine(
        Base.Vector(x_ex, y_ex_up, 0),
        Base.Vector(x_ex, y_ex_lo, 0),
    )

    wire = Part.Wire([edge_up, edge_ex, edge_le, edge_lo])
    face = Part.Face(wire)
    solid = face.extrude(Base.Vector(0, 0, 1.0))  # W = 1.0 m
    return solid


def _build_shape_axisymmetric(geo: dict, Part, Base):
    """轴对称：中心锥 + Cowl 各自 BSpline + Revolution，返回两个 Shape。"""
    for key in ("profile_cb", "profile_cowl"):
        if key not in geo:
            raise ValueError(f"geo_result 缺少 '{key}' 键（axisymmetric 构型）。")

    axis_pt = Base.Vector(0, 0, 0)
    axis_dir = Base.Vector(1, 0, 0)

    vecs_cb = _vec_list(geo["profile_cb"], Base)
    bs_cb = Part.BSplineCurve()
    bs_cb.interpolate(vecs_cb)
    shape_cb = bs_cb.toShape().revolve(axis_pt, axis_dir, 360)

    vecs_cowl = _vec_list(geo["profile_cowl"], Base)
    bs_cowl = Part.BSplineCurve()
    bs_cowl.interpolate(vecs_cowl)
    shape_cowl = bs_cowl.toShape().revolve(axis_pt, axis_dir, 360)

    return [shape_cb, shape_cowl]
