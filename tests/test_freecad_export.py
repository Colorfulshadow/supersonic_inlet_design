"""
tests/test_freecad_export.py
============================
验证 geometry/freecad_export.py 的功能正确性。

FreeCAD 相关测试用 skipif 保护：FreeCAD 未安装时自动跳过，不影响 CI 流程。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import geometry.freecad_export as freecad_export


# ---------------------------------------------------------------------------
# 测试夹具
# ---------------------------------------------------------------------------

@pytest.fixture
def pitot_geo():
    return {
        "profile": [
            (0.0, 0.5),
            (1.0, 0.6),
            (2.0, 0.7),
        ]
    }


@pytest.fixture
def external_2d_geo():
    return {
        "profile_upper": [
            (-1.0, 1.0),
            (0.0,  1.0),
            (1.0,  1.0),
        ],
        "profile_lower": [
            (-1.0, 0.0),
            (0.0,  0.4),
            (1.0,  0.5),
        ],
    }


@pytest.fixture
def axisymmetric_geo():
    return {
        "profile_cb": [
            (-0.8, 0.0),
            (0.0,  0.3),
        ],
        "profile_cowl": [
            (0.0,  0.7),
            (0.8,  0.8),
        ],
    }


# ---------------------------------------------------------------------------
# is_freecad_available — 永不抛异常
# ---------------------------------------------------------------------------

def test_freecad_availability_check():
    """is_freecad_available() 返回 bool，不抛任何异常。"""
    result = freecad_export.is_freecad_available()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# FreeCAD 不可用时的行为
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    freecad_export.is_freecad_available(),
    reason="FreeCAD is installed; this test checks the unavailable path"
)
def test_import_error_when_freecad_unavailable(pitot_geo, tmp_path):
    """FreeCAD 不可用时，export_to_freecad 抛出 ImportError（含安装提示）。"""
    out = str(tmp_path / "pitot.step")
    with pytest.raises(ImportError, match="FreeCAD"):
        freecad_export.export_to_freecad("pitot", pitot_geo, out)


# ---------------------------------------------------------------------------
# FreeCAD 可用时的测试（skipif 保护）
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not freecad_export.is_freecad_available(),
    reason="FreeCAD not installed"
)
def test_export_step_file_pitot(pitot_geo, tmp_path):
    """FreeCAD 可用时，export_to_freecad 生成非空 .step 文件（pitot）。"""
    out = str(tmp_path / "pitot.step")
    freecad_export.export_to_freecad("pitot", pitot_geo, out)
    assert os.path.isfile(out)
    assert os.path.getsize(out) > 0


@pytest.mark.skipif(
    not freecad_export.is_freecad_available(),
    reason="FreeCAD not installed"
)
def test_export_brep_file_axisymmetric(axisymmetric_geo, tmp_path):
    """FreeCAD 可用时，export_to_freecad 生成非空 .brep 文件（axisymmetric）。"""
    out = str(tmp_path / "axi.brep")
    freecad_export.export_to_freecad("axisymmetric", axisymmetric_geo, out)
    assert os.path.isfile(out)
    assert os.path.getsize(out) > 0


@pytest.mark.skipif(
    not freecad_export.is_freecad_available(),
    reason="FreeCAD not installed"
)
def test_export_step_file_external_2d(external_2d_geo, tmp_path):
    """FreeCAD 可用时，export_to_freecad 生成非空 .step 文件（external_2d）。"""
    out = str(tmp_path / "ext2d.step")
    freecad_export.export_to_freecad("external_2d", external_2d_geo, out)
    assert os.path.isfile(out)
    assert os.path.getsize(out) > 0


# ---------------------------------------------------------------------------
# 错误处理（不依赖 FreeCAD）
# ---------------------------------------------------------------------------

def test_invalid_inlet_type_raises_value_error(pitot_geo, tmp_path):
    """inlet_type 不合法时抛出 ValueError（在 ImportError 之前被检测到）。

    注意：若 FreeCAD 不可用，会先抛 ImportError；若可用，则抛 ValueError。
    本测试仅在 FreeCAD 可用时才能触达 ValueError 路径。
    """
    if not freecad_export.is_freecad_available():
        pytest.skip("FreeCAD not installed; cannot reach ValueError path")
    out = str(tmp_path / "bad.step")
    with pytest.raises(ValueError, match="inlet_type"):
        freecad_export.export_to_freecad("invalid_type", pitot_geo, out)


def test_unsupported_extension_raises(tmp_path, pitot_geo):
    """不支持的后缀（如 .obj）抛出 ValueError。"""
    if not freecad_export.is_freecad_available():
        pytest.skip("FreeCAD not installed; cannot reach extension check")
    out = str(tmp_path / "pitot.obj")
    with pytest.raises(ValueError, match="后缀"):
        freecad_export.export_to_freecad("pitot", pitot_geo, out)
