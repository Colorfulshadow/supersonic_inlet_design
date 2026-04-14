"""
tests/test_freecad_script.py
============================
验证 geometry/freecad_script.py 的功能正确性。

所有测试均不依赖 FreeCAD 运行时：只验证生成的宏脚本是合法 Python、
包含正确的坐标数据和 Shape 名称。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geometry.freecad_script import generate_freecad_macro


# ---------------------------------------------------------------------------
# 测试夹具（最小 geo_result，不依赖气动计算）
# ---------------------------------------------------------------------------

@pytest.fixture
def pitot_geo():
    return {
        "profile": [
            (-0.5, 0.5),
            (0.0,  0.5),
            (1.0,  0.6),
            (2.0,  0.7),
        ]
    }


@pytest.fixture
def external_2d_geo():
    return {
        "profile_upper": [
            (-2.0, 1.0),
            (0.0,  1.0),
            (2.0,  1.0),
        ],
        "profile_lower": [
            (-2.0, 0.0),
            (0.0,  0.5),
            (2.0,  0.6),
        ],
    }


@pytest.fixture
def axisymmetric_geo():
    return {
        "profile_cb": [
            (-1.0, 0.0),
            (0.0,  0.4),
        ],
        "profile_cowl": [
            (0.0,  0.8),
            (1.0,  0.9),
        ],
    }


# ---------------------------------------------------------------------------
# 合法 Python 验证（compile() 不抛异常）
# ---------------------------------------------------------------------------

def test_pitot_macro_generates_valid_python(pitot_geo, tmp_path):
    """生成的皮托管宏脚本是合法 Python（可被 compile() 解析），不依赖 FreeCAD 运行时。"""
    out = str(tmp_path / "pitot.py")
    macro = generate_freecad_macro("pitot", pitot_geo, out)
    # compile() 验证语法合法性（不执行）
    compile(macro, "<pitot_macro>", "exec")


def test_external_2d_macro_generates_valid_python(external_2d_geo, tmp_path):
    """生成的二元外压式宏脚本是合法 Python。"""
    out = str(tmp_path / "ext2d.py")
    macro = generate_freecad_macro("external_2d", external_2d_geo, out)
    compile(macro, "<external_2d_macro>", "exec")


def test_axisymmetric_macro_generates_valid_python(axisymmetric_geo, tmp_path):
    """生成的轴对称宏脚本是合法 Python。"""
    out = str(tmp_path / "axi.py")
    macro = generate_freecad_macro("axisymmetric", axisymmetric_geo, out)
    compile(macro, "<axisymmetric_macro>", "exec")


# ---------------------------------------------------------------------------
# 坐标数据包含验证
# ---------------------------------------------------------------------------

def test_external_2d_macro_contains_profile(external_2d_geo, tmp_path):
    """宏脚本中包含来自 profile_upper 的坐标数值。"""
    out = str(tmp_path / "ext2d.py")
    macro = generate_freecad_macro("external_2d", external_2d_geo, out)
    # 检查 profile_upper 的第一个 x 坐标出现在宏中
    x0, y0 = external_2d_geo["profile_upper"][0]
    assert str(x0) in macro or f"{x0:.10g}" in macro, (
        f"profile_upper[0] x={x0} 未出现在宏脚本中。"
    )


def test_pitot_macro_contains_profile_coords(pitot_geo, tmp_path):
    """宏脚本中包含来自 profile 的坐标数值。"""
    out = str(tmp_path / "pitot.py")
    macro = generate_freecad_macro("pitot", pitot_geo, out)
    for x, r in pitot_geo["profile"]:
        assert str(x) in macro or f"{x:.10g}" in macro, (
            f"profile 点 x={x} 未出现在宏脚本中。"
        )


# ---------------------------------------------------------------------------
# 轴对称双 Shape 名称验证
# ---------------------------------------------------------------------------

def test_axisymmetric_macro_has_two_shapes(axisymmetric_geo, tmp_path):
    """宏脚本中同时包含 'CenterBody' 和 'Cowl' 两个 Shape 名称。"""
    out = str(tmp_path / "axi.py")
    macro = generate_freecad_macro("axisymmetric", axisymmetric_geo, out)
    assert "CenterBody" in macro, "'CenterBody' 未出现在宏脚本中。"
    assert "Cowl" in macro, "'Cowl' 未出现在宏脚本中。"


# ---------------------------------------------------------------------------
# 文件写入验证
# ---------------------------------------------------------------------------

def test_macro_written_to_file(tmp_path, pitot_geo):
    """generate_freecad_macro 正确写入文件，文件可读，内容与返回值一致。"""
    out = str(tmp_path / "pitot_out.py")
    macro_str = generate_freecad_macro("pitot", pitot_geo, out)

    assert os.path.isfile(out), "宏脚本文件未创建。"
    with open(out, encoding="utf-8") as f:
        content = f.read()
    assert content == macro_str, "写入文件内容与返回字符串不一致。"


def test_macro_file_nonempty(tmp_path, external_2d_geo):
    """写入的宏脚本文件非空。"""
    out = str(tmp_path / "ext2d_out.py")
    generate_freecad_macro("external_2d", external_2d_geo, out)
    assert os.path.getsize(out) > 0


# ---------------------------------------------------------------------------
# 错误处理
# ---------------------------------------------------------------------------

def test_invalid_inlet_type_raises(tmp_path, pitot_geo):
    """inlet_type 不合法时抛出 ValueError。"""
    out = str(tmp_path / "bad.py")
    with pytest.raises(ValueError, match="inlet_type"):
        generate_freecad_macro("invalid_type", pitot_geo, out)


def test_missing_profile_key_raises(tmp_path):
    """geo_result 缺少必要键时抛出 ValueError。"""
    out = str(tmp_path / "bad.py")
    with pytest.raises(ValueError):
        generate_freecad_macro("pitot", {}, out)


# ---------------------------------------------------------------------------
# 宏脚本头部结构验证
# ---------------------------------------------------------------------------

def test_macro_header_contains_import(pitot_geo, tmp_path):
    """宏脚本包含 'import FreeCAD' 语句（供 FreeCAD 内部执行）。"""
    out = str(tmp_path / "pitot.py")
    macro = generate_freecad_macro("pitot", pitot_geo, out)
    assert "import FreeCAD" in macro


def test_macro_header_contains_doc_new(pitot_geo, tmp_path):
    """宏脚本包含 FreeCAD.newDocument 调用。"""
    out = str(tmp_path / "pitot.py")
    macro = generate_freecad_macro("pitot", pitot_geo, out)
    assert "FreeCAD.newDocument" in macro


def test_macro_footer_contains_recompute(axisymmetric_geo, tmp_path):
    """宏脚本包含 doc.recompute() 调用。"""
    out = str(tmp_path / "axi.py")
    macro = generate_freecad_macro("axisymmetric", axisymmetric_geo, out)
    assert "doc.recompute()" in macro
