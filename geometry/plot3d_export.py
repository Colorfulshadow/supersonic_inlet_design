"""
geometry/plot3d_export.py
=========================
Plot3D 结构化网格（.xyz）和 ASCII STL（.stl）导出工具。

依赖：
- numpy
- geometry/surface_3d.py（旋转/拉伸工具函数）

禁止 import FreeCAD（CLAUDE.md 禁止事项）。
禁止在本模块中调用气动设计函数，所有输入来自各构型 geometry.py 的输出字典。
"""

from __future__ import annotations

import os
import numpy as np
from typing import Dict

from geometry.surface_3d import build_revolution_surface, build_extrusion_surface


# ---------------------------------------------------------------------------
# Plot3D ASCII 单块格式
# ---------------------------------------------------------------------------

def export_plot3d(surface_xyz: np.ndarray, output_path: str) -> None:
    """输出 Plot3D 单块结构化网格（ASCII 格式）。

    Parameters
    ----------
    surface_xyz : np.ndarray
        shape = (N_i, N_j, 3)，结构化面网格坐标 [x, y, z]。
    output_path : str
        .xyz 文件路径。

    Plot3D ASCII 格式（单块 2D 切片，k=1）：
        第 1 行：块数（1）
        第 2 行：N_i  N_j  1
        之后：x 坐标（Fortran 列优先顺序，先 i 后 j）
              y 坐标
              z 坐标
    """
    if surface_xyz.ndim != 3 or surface_xyz.shape[2] != 3:
        raise ValueError(
            f"surface_xyz 须为 shape (N_i, N_j, 3)，当前 shape={surface_xyz.shape}。"
        )

    N_i, N_j, _ = surface_xyz.shape
    x = surface_xyz[:, :, 0]
    y = surface_xyz[:, :, 1]
    z = surface_xyz[:, :, 2]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("1\n")                              # 块数
        f.write(f"{N_i} {N_j} 1\n")               # 维度

        # Fortran 列优先顺序：先遍历 i（行），再遍历 j（列）
        # 即 x[0,0], x[1,0], ..., x[N_i-1,0], x[0,1], ...
        for coord in (x, y, z):
            vals = coord.T.ravel()                  # Fortran order: j-major
            # 每行写 6 个值（Plot3D 惯例）
            for start in range(0, len(vals), 6):
                line = "  ".join(
                    f"{v: .10E}" for v in vals[start:start + 6]
                )
                f.write(line + "\n")


def read_plot3d(input_path: str) -> np.ndarray:
    """读取 Plot3D ASCII 单块文件，返回 (N_i, N_j, 3) 数组。

    仅用于 roundtrip 测试验证，不作为主接口。
    """
    with open(input_path, "r") as f:
        lines = f.read().split()

    idx = 0
    n_blocks = int(lines[idx]); idx += 1
    if n_blocks != 1:
        raise ValueError("仅支持单块 Plot3D 文件。")

    N_i = int(lines[idx]); idx += 1
    N_j = int(lines[idx]); idx += 1
    N_k = int(lines[idx]); idx += 1

    n_pts = N_i * N_j * N_k
    x_flat = np.array([float(lines[idx + k]) for k in range(n_pts)]); idx += n_pts
    y_flat = np.array([float(lines[idx + k]) for k in range(n_pts)]); idx += n_pts
    z_flat = np.array([float(lines[idx + k]) for k in range(n_pts)]); idx += n_pts

    # Fortran order 还原：j-major → (N_i, N_j)
    x = x_flat.reshape((N_j, N_i)).T
    y = y_flat.reshape((N_j, N_i)).T
    z = z_flat.reshape((N_j, N_i)).T

    surf = np.stack([x, y, z], axis=2)
    return surf


# ---------------------------------------------------------------------------
# ASCII STL
# ---------------------------------------------------------------------------

def export_stl(surface_xyz: np.ndarray, output_path: str) -> None:
    """输出 ASCII STL 文件（三角面片）。

    Parameters
    ----------
    surface_xyz : np.ndarray
        shape = (N_i, N_j, 3)。
    output_path : str
        .stl 文件路径。

    三角剖分规则：每个四边形格子拆为 2 个三角形。
    法向量由右手定则（叉积 (v1-v0) × (v2-v0)）确定并归一化。
    退化三角形（面积≈0）写入零法向量，不跳过（STL 规范允许）。
    """
    if surface_xyz.ndim != 3 or surface_xyz.shape[2] != 3:
        raise ValueError(
            f"surface_xyz 须为 shape (N_i, N_j, 3)，当前 shape={surface_xyz.shape}。"
        )

    N_i, N_j, _ = surface_xyz.shape

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    solid_name = os.path.splitext(os.path.basename(output_path))[0]

    with open(output_path, "w") as f:
        f.write(f"solid {solid_name}\n")

        for i in range(N_i - 1):
            for j in range(N_j - 1):
                # 四顶点（逆时针为正法向）
                p00 = surface_xyz[i,     j    ]
                p10 = surface_xyz[i + 1, j    ]
                p01 = surface_xyz[i,     j + 1]
                p11 = surface_xyz[i + 1, j + 1]

                # 三角形 1: (p00, p10, p11)
                _write_stl_triangle(f, p00, p10, p11)
                # 三角形 2: (p00, p11, p01)
                _write_stl_triangle(f, p00, p11, p01)

        f.write(f"endsolid {solid_name}\n")


def _write_stl_triangle(f, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> None:
    """向已打开的文件写入一个 STL 三角面片。"""
    e1 = v1 - v0
    e2 = v2 - v0
    n = np.cross(e1, e2)
    norm = np.linalg.norm(n)
    if norm > 1e-30:
        n = n / norm
    f.write(f"  facet normal {n[0]:.6E} {n[1]:.6E} {n[2]:.6E}\n")
    f.write("    outer loop\n")
    f.write(f"      vertex {v0[0]:.10E} {v0[1]:.10E} {v0[2]:.10E}\n")
    f.write(f"      vertex {v1[0]:.10E} {v1[1]:.10E} {v1[2]:.10E}\n")
    f.write(f"      vertex {v2[0]:.10E} {v2[1]:.10E} {v2[2]:.10E}\n")
    f.write("    endloop\n")
    f.write("  endfacet\n")


# ---------------------------------------------------------------------------
# 三构型表面构建便利函数
# ---------------------------------------------------------------------------

def build_surface_pitot(geo_result: dict, N_phi: int = 72) -> np.ndarray:
    """将皮托管 profile 母线绕 x 轴旋转，生成 (N_r, N_phi, 3) 网格。

    Parameters
    ----------
    geo_result : dict
        ``pitot_geometry()`` 的返回值，须含 ``'profile'`` 键
        （list of (x, r)）。
    N_phi : int
        周向分点数，默认 72。

    Returns
    -------
    np.ndarray
        shape = (N_profile_pts, N_phi, 3)。
    """
    profile = geo_result["profile"]
    return build_revolution_surface(profile, N_phi=N_phi)


def build_surface_axisymmetric(geo_result: dict, N_phi: int = 72) -> Dict[str, np.ndarray]:
    """将轴对称构型母线绕 x 轴旋转，生成两个旋转体表面。

    Parameters
    ----------
    geo_result : dict
        ``axisymmetric_geometry()`` 的返回值，须含
        ``'profile_cb'``（中心锥）和 ``'profile_cowl'``（cowl 内壁）。
    N_phi : int
        周向分点数，默认 72。

    Returns
    -------
    dict
        ``{'center_body': ndarray, 'cowl': ndarray}``，
        各为 shape = (N_pts, N_phi, 3)。
    """
    cb_surf = build_revolution_surface(geo_result["profile_cb"], N_phi=N_phi)
    cowl_surf = build_revolution_surface(geo_result["profile_cowl"], N_phi=N_phi)
    return {"center_body": cb_surf, "cowl": cowl_surf}


def build_surface_2d(
    geo_result: dict,
    W: float = 1.0,
    N_z: int = 2,
) -> Dict[str, np.ndarray]:
    """将二元外压式上下壁型线沿 z 轴拉伸，生成平板面网格。

    Parameters
    ----------
    geo_result : dict
        ``external_2d_geometry()`` 的返回值，须含
        ``'profile_upper'`` 和 ``'profile_lower'`` 键。
    W : float
        展向宽度（m），默认 1.0 m。
    N_z : int
        展向分点数，默认 2。

    Returns
    -------
    dict
        ``{'upper': ndarray, 'lower': ndarray}``，
        各为 shape = (N_x, N_z, 3)。
    """
    upper = build_extrusion_surface(geo_result["profile_upper"], W=W, N_z=N_z)
    lower = build_extrusion_surface(geo_result["profile_lower"], W=W, N_z=N_z)
    return {"upper": upper, "lower": lower}
