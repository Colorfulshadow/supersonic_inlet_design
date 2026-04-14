"""
geometry/surface_3d.py
======================
2D 母线 → 3D 结构化面网格工具函数。

两种建面方式：
- :func:`build_revolution_surface`：绕 x 轴旋转（轴对称构型）
- :func:`build_extrusion_surface`：沿 z 轴拉伸（二元矩形截面构型）

输出数组 shape = (N_i, N_j, 3)，坐标顺序 [x, y, z]，
与 :mod:`geometry.plot3d_export` 的 Plot3D 格式直接对接。

不依赖 FreeCAD，不调用气动设计函数。
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple


def build_revolution_surface(
    profile: List[Tuple[float, float]],
    N_phi: int = 72,
) -> np.ndarray:
    """将 (x, r) 母线绕 x 轴旋转一周，生成结构化面网格。

    Parameters
    ----------
    profile : list of (x, r)
        母线离散点，r >= 0，x 单调递增（不强制检查）。
    N_phi : int
        周向分点数（包含 0° 和 360°），默认 72（每 5°一个截面）。

    Returns
    -------
    np.ndarray
        shape = (N_r, N_phi, 3)，坐标顺序 [x, y, z]。
        N_r = len(profile)。

        旋转公式（φ 从 0 到 2π）：
            x[i,j] = profile[i][0]
            y[i,j] = profile[i][1] * cos(phi[j])
            z[i,j] = profile[i][1] * sin(phi[j])
    """
    if N_phi < 2:
        raise ValueError(f"N_phi 须 >= 2，当前 N_phi={N_phi}。")
    if not profile:
        raise ValueError("profile 不能为空。")

    xs = np.array([p[0] for p in profile], dtype=float)
    rs = np.array([p[1] for p in profile], dtype=float)
    N_r = len(xs)

    phi = np.linspace(0.0, 2.0 * np.pi, N_phi)

    surf = np.empty((N_r, N_phi, 3), dtype=float)
    surf[:, :, 0] = xs[:, np.newaxis]                          # x 不变
    surf[:, :, 1] = rs[:, np.newaxis] * np.cos(phi)[np.newaxis, :]  # y
    surf[:, :, 2] = rs[:, np.newaxis] * np.sin(phi)[np.newaxis, :]  # z

    return surf


def build_extrusion_surface(
    profile: List[Tuple[float, float]],
    W: float = 1.0,
    N_z: int = 2,
) -> np.ndarray:
    """将 (x, y) 型线沿 z 轴拉伸，生成平板状结构化面网格。

    Parameters
    ----------
    profile : list of (x, y)
        型线离散点（上壁或下壁），x 单调递增。
    W : float
        展向拉伸宽度（m），须 > 0，默认 1.0 m（单位展宽）。
    N_z : int
        展向分点数（包含两端），须 >= 2，默认 2（仅两端面）。

    Returns
    -------
    np.ndarray
        shape = (N_x, N_z, 3)，坐标顺序 [x, y, z]。
        N_x = len(profile)。

        拉伸公式（z 从 0 到 W）：
            x[i,j] = profile[i][0]
            y[i,j] = profile[i][1]
            z[i,j] = z_vals[j]
    """
    if W <= 0:
        raise ValueError(f"W 须 > 0，当前 W={W}。")
    if N_z < 2:
        raise ValueError(f"N_z 须 >= 2，当前 N_z={N_z}。")
    if not profile:
        raise ValueError("profile 不能为空。")

    xs = np.array([p[0] for p in profile], dtype=float)
    ys = np.array([p[1] for p in profile], dtype=float)
    N_x = len(xs)

    z_vals = np.linspace(0.0, W, N_z)

    surf = np.empty((N_x, N_z, 3), dtype=float)
    surf[:, :, 0] = xs[:, np.newaxis]
    surf[:, :, 1] = ys[:, np.newaxis]
    surf[:, :, 2] = z_vals[np.newaxis, :]

    return surf
