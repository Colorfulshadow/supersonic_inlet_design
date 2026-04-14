"""
tests/test_plot3d_export.py
===========================
验证 geometry/plot3d_export.py 的功能正确性。

测试不依赖 FreeCAD，不调用气动设计函数（使用最小手工 geo_result）。
"""

import math
import os
import sys

import numpy as np
import pytest

# 确保项目根目录在路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geometry.plot3d_export import (
    build_surface_2d,
    build_surface_axisymmetric,
    build_surface_pitot,
    export_plot3d,
    export_stl,
    read_plot3d,
)


# ---------------------------------------------------------------------------
# 测试夹具（最小 geo_result，不依赖气动计算）
# ---------------------------------------------------------------------------

@pytest.fixture
def pitot_geo():
    """皮托管最小 geo_result：4 个型线点。"""
    return {
        "profile": [
            (-0.5, 0.5),
            (0.0,  0.5),
            (1.0,  0.6),
            (2.0,  0.7),
        ]
    }


@pytest.fixture
def axisymmetric_geo():
    """轴对称最小 geo_result：中心锥 + cowl 各 4 点。"""
    return {
        "profile_cb": [
            (-1.0, 0.0),
            (-0.5, 0.2),
            (0.0,  0.4),
        ],
        "profile_cowl": [
            (0.0,  0.8),
            (0.5,  0.85),
            (1.0,  0.9),
        ],
    }


@pytest.fixture
def external_2d_geo():
    """二元外压式最小 geo_result。"""
    return {
        "profile_upper": [
            (-2.0, 1.0),
            (-1.0, 1.0),
            (0.0,  1.0),
            (1.0,  1.0),
            (2.0,  1.0),
        ],
        "profile_lower": [
            (-2.0, 0.0),
            (-1.0, 0.3),
            (0.0,  0.5),
            (1.0,  0.55),
            (2.0,  0.6),
        ],
    }


# ---------------------------------------------------------------------------
# build_surface_pitot
# ---------------------------------------------------------------------------

def test_pitot_surface_revolution_shape(pitot_geo):
    """build_surface_pitot 返回 shape = (N_profile_pts, N_phi, 3)。"""
    N_phi = 36
    surf = build_surface_pitot(pitot_geo, N_phi=N_phi)
    N_pts = len(pitot_geo["profile"])
    assert surf.shape == (N_pts, N_phi, 3)


def test_pitot_surface_x_invariant(pitot_geo):
    """旋转后每条母线的 x 坐标沿周向不变。"""
    surf = build_surface_pitot(pitot_geo, N_phi=36)
    for i in range(surf.shape[0]):
        x_row = surf[i, :, 0]
        assert np.allclose(x_row, x_row[0]), f"第 {i} 母线点 x 坐标在旋转方向变化。"


def test_pitot_surface_radius_correct(pitot_geo):
    """旋转后 y² + z² = r² 应精确成立。"""
    surf = build_surface_pitot(pitot_geo, N_phi=36)
    profile = pitot_geo["profile"]
    for i, (_, r) in enumerate(profile):
        r_computed = np.sqrt(surf[i, :, 1]**2 + surf[i, :, 2]**2)
        assert np.allclose(r_computed, abs(r), atol=1e-12)


# ---------------------------------------------------------------------------
# build_surface_axisymmetric
# ---------------------------------------------------------------------------

def test_axisymmetric_surfaces_both_present(axisymmetric_geo):
    """build_surface_axisymmetric 返回 dict 含 'cowl' 和 'center_body' 两个键。"""
    result = build_surface_axisymmetric(axisymmetric_geo, N_phi=36)
    assert "cowl" in result
    assert "center_body" in result


def test_axisymmetric_surfaces_correct_shape(axisymmetric_geo):
    """两个旋转体的 shape 分别对应各自母线点数。"""
    N_phi = 36
    result = build_surface_axisymmetric(axisymmetric_geo, N_phi=N_phi)
    n_cb = len(axisymmetric_geo["profile_cb"])
    n_cw = len(axisymmetric_geo["profile_cowl"])
    assert result["center_body"].shape == (n_cb, N_phi, 3)
    assert result["cowl"].shape == (n_cw, N_phi, 3)


# ---------------------------------------------------------------------------
# build_surface_2d
# ---------------------------------------------------------------------------

def test_2d_surfaces_both_present(external_2d_geo):
    """build_surface_2d 返回 dict 含 'upper' 和 'lower' 两个键。"""
    result = build_surface_2d(external_2d_geo, W=1.0, N_z=2)
    assert "upper" in result
    assert "lower" in result


def test_2d_surface_shape(external_2d_geo):
    """拉伸面 shape = (N_x, N_z, 3)。"""
    N_z = 3
    result = build_surface_2d(external_2d_geo, W=2.0, N_z=N_z)
    N_x = len(external_2d_geo["profile_upper"])
    assert result["upper"].shape == (N_x, N_z, 3)
    assert result["lower"].shape == (N_x, N_z, 3)


def test_2d_surface_z_range(external_2d_geo):
    """拉伸方向 z 从 0 到 W。"""
    W = 1.5
    result = build_surface_2d(external_2d_geo, W=W, N_z=5)
    z_upper = result["upper"][:, :, 2]
    assert pytest.approx(z_upper.min(), abs=1e-12) == 0.0
    assert pytest.approx(z_upper.max(), abs=1e-12) == W


# ---------------------------------------------------------------------------
# export_plot3d 格式验证
# ---------------------------------------------------------------------------

def test_plot3d_file_format(tmp_path, pitot_geo):
    """输出文件第一行为 '1'（单块），第二行含正确的 N_i N_j 1。"""
    surf = build_surface_pitot(pitot_geo, N_phi=36)
    out = str(tmp_path / "test.xyz")
    export_plot3d(surf, out)

    with open(out) as f:
        lines = [l.strip() for l in f if l.strip()]

    assert lines[0] == "1"
    parts = lines[1].split()
    N_i, N_j, N_k = int(parts[0]), int(parts[1]), int(parts[2])
    assert N_k == 1
    assert N_i == surf.shape[0]
    assert N_j == surf.shape[1]


def test_plot3d_roundtrip(tmp_path, pitot_geo):
    """写入后重新读取，坐标值与原始数组误差 < 1e-6。"""
    surf = build_surface_pitot(pitot_geo, N_phi=36)
    out = str(tmp_path / "roundtrip.xyz")
    export_plot3d(surf, out)
    recovered = read_plot3d(out)
    assert recovered.shape == surf.shape
    assert np.allclose(recovered, surf, atol=1e-6)


# ---------------------------------------------------------------------------
# export_stl 三角形数量验证
# ---------------------------------------------------------------------------

def test_stl_triangle_count(tmp_path, pitot_geo):
    """对于 (N_i, N_j) 网格，三角形数量 = 2*(N_i-1)*(N_j-1)。"""
    N_phi = 36
    surf = build_surface_pitot(pitot_geo, N_phi=N_phi)
    out = str(tmp_path / "test.stl")
    export_stl(surf, out)

    N_i, N_j = surf.shape[:2]
    expected_triangles = 2 * (N_i - 1) * (N_j - 1)

    with open(out) as f:
        content = f.read()

    count = content.count("facet normal")
    assert count == expected_triangles


def test_stl_file_syntax(tmp_path, axisymmetric_geo):
    """STL 文件以 'solid' 开头和 'endsolid' 结尾。"""
    surf = build_surface_axisymmetric(axisymmetric_geo, N_phi=18)["cowl"]
    out = str(tmp_path / "axi.stl")
    export_stl(surf, out)

    with open(out) as f:
        content = f.read()

    assert content.startswith("solid ")
    assert content.strip().endswith("endsolid " + os.path.splitext(os.path.basename(out))[0])
