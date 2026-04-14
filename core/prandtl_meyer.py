"""
core/prandtl_meyer.py
=====================
Prandtl-Meyer 等熵膨胀/压缩函数，用于等熵压缩段（等熵 ramp）设计。

公开接口
--------
- ``prandtl_meyer_angle(M, gamma=1.4) -> float``  单位：弧度，M ≤ 1 时返回 0.0
- ``M_from_pm_angle(nu, gamma=1.4) -> float``  brentq 反算，[1+1e-9, 20.0]
- ``isentropic_ramp_coords(M_start, theta_start_deg, delta_theta_deg, n_steps, gamma)``
  → ``(xs, ys, M_final, theta_final_deg)``

物理说明
--------
等熵压缩时流动转向角增大（ramp 斜面更陡），Prandtl-Meyer 角 ν 减小：

    ν(M_final) = ν(M_start) − delta_theta  (弧度)

角度约定
--------
- 函数内部：弧度
- 公开接口：theta_start_deg、delta_theta_deg 以度（°）传入
- 返回值 theta_final_deg 以度（°）输出

禁止在本模块中导入 geometry/、gui/ 或 inlets/ 的任何内容。
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Prandtl-Meyer 角
# ---------------------------------------------------------------------------

def prandtl_meyer_angle(M: float, gamma: float = 1.4) -> float:
    """Prandtl-Meyer 角（弧度）。

    Parameters
    ----------
    M : float
        马赫数。M ≤ 1 时返回 0.0。
    gamma : float
        比热比，默认 1.4。

    Returns
    -------
    float
        ν(M)（弧度）。M ≤ 1 时返回 0.0。

    Notes
    -----
    公式（Anderson《现代可压缩流》式 4.43）：

    .. code-block::

        ν(M) = √((γ+1)/(γ-1)) · arctan(√((γ-1)/(γ+1) · (M²-1))) - arctan(√(M²-1))

    数值验证（γ=1.4）：
    - M=1.0 → ν=0.0 rad
    - M=2.0 → ν≈0.4606 rad（26.38°）
    - M=1.4 → ν≈0.1569 rad（8.99°）
    """
    if M <= 1.0:
        return 0.0
    g  = float(gamma)
    g1 = g - 1.0
    gp = g + 1.0
    k  = math.sqrt(gp / g1)
    t1 = math.sqrt(g1 / gp * (M * M - 1.0))
    t2 = math.sqrt(M * M - 1.0)
    return float(k * math.atan(t1) - math.atan(t2))


# ---------------------------------------------------------------------------
# 反算：由 ν 求 M
# ---------------------------------------------------------------------------

def M_from_pm_angle(nu: float, gamma: float = 1.4) -> float:
    """由 Prandtl-Meyer 角（弧度）反算马赫数。

    Parameters
    ----------
    nu : float
        Prandtl-Meyer 角（弧度），须 ≥ 0。
    gamma : float
        比热比，默认 1.4。

    Returns
    -------
    float
        对应的马赫数（≥ 1.0）。

    Raises
    ------
    ValueError
        - 若 ``nu < 0``。
        - 若 ``nu`` 超过给定 γ 下的 PM 角上限（M→∞）。

    Notes
    -----
    上限：ν_max = π/2 · (√((γ+1)/(γ-1)) − 1)，γ=1.4 时约 130.45°（2.277 rad）。
    """
    if nu < 0.0:
        raise ValueError(f"Prandtl-Meyer 角不能为负：nu={nu:.6f} rad。")
    if nu == 0.0:
        return 1.0

    nu_max = (math.pi / 2.0) * (math.sqrt((gamma + 1.0) / (gamma - 1.0)) - 1.0)
    if nu > nu_max:
        raise ValueError(
            f"nu={math.degrees(nu):.4f}° 超过 γ={gamma} 时的 PM 角上限 "
            f"{math.degrees(nu_max):.4f}°（M→∞）。"
        )

    def f(M_try: float) -> float:
        return prandtl_meyer_angle(M_try, gamma) - nu

    return float(brentq(f, 1.0 + 1e-9, 20.0, xtol=1e-10, maxiter=200))


# ---------------------------------------------------------------------------
# 等熵压缩 ramp 坐标
# ---------------------------------------------------------------------------

def isentropic_ramp_coords(
    M_start: float,
    theta_start_deg: float,
    delta_theta_deg: float,
    n_steps: int = 50,
    gamma: float = 1.4,
) -> tuple:
    """等熵压缩 ramp 归一化坐标。

    将 Prandtl-Meyer 等熵压缩过程离散化为 ``n_steps`` 个小步，每步弧长 ds=1
    （归一化），沿当前 ramp 斜面方向（倾角 θᵢ）前进。

    Parameters
    ----------
    M_start : float
        等熵段起始马赫数（须 > 1）。
    theta_start_deg : float
        起始 ramp 面相对水平方向的倾斜角（度，上倾为正）。
        通常等于前序斜激波的累积偏转角 Σθⱼ。
    delta_theta_deg : float
        等熵压缩段额外偏转角（度，须 > 0）。马赫数按 PM 关系降低：
        ν(M_final) = ν(M_start) − delta_theta_deg（单位转换后）。
    n_steps : int
        离散步数，默认 50。步数越多，曲线越平滑。
    gamma : float
        比热比，默认 1.4。

    Returns
    -------
    xs : np.ndarray, shape (n_steps + 1,)
        归一化 x 坐标，``xs[0] = 0``，单调递增（每步 cos(θᵢ)）。
    ys : np.ndarray, shape (n_steps + 1,)
        归一化 y 坐标，``ys[0] = 0``，单调递增（每步 sin(θᵢ)，θ > 0 时正）。
    M_final : float
        等熵压缩段末尾马赫数（< M_start）。
    theta_final_deg : float
        等熵压缩段末尾 ramp 倾斜角（度）= ``theta_start_deg + delta_theta_deg``。

    Raises
    ------
    ValueError
        - ``M_start ≤ 1``
        - ``delta_theta_deg ≤ 0``
        - ``n_steps < 1``

    Notes
    -----
    **归一化**：返回坐标弧长单位为 1/step，总弧长 = n_steps。
    调用方需乘以物理缩放因子以获得米制坐标：

    .. code-block::

        scale = L_iso / n_steps   # L_iso：物理弧长（m）
        xs_phys = xs * scale + x_start
        ys_phys = ys * scale + y_start

    **坐标积分方式**（前向欧拉）：
    步骤 i（0-indexed）沿当前 ramp 倾角 θᵢ = theta_start + i·d_theta 方向前进：

    .. code-block::

        xs[i+1] = xs[i] + cos(θᵢ)
        ys[i+1] = ys[i] + sin(θᵢ)

    马赫数在每步后更新：M_{i+1} = M_from_pm_angle(ν_start − (i+1)·d_theta_rad)。
    """
    if M_start <= 1.0:
        raise ValueError(f"M_start 须 > 1，当前 {M_start}。")
    if delta_theta_deg <= 0.0:
        raise ValueError(f"delta_theta_deg 须 > 0，当前 {delta_theta_deg}。")
    if n_steps < 1:
        raise ValueError(f"n_steps 须 ≥ 1，当前 {n_steps}。")

    nu_start    = prandtl_meyer_angle(M_start, gamma)
    d_theta_deg = float(delta_theta_deg) / float(n_steps)
    d_theta_rad = math.radians(d_theta_deg)

    xs = np.zeros(n_steps + 1)
    ys = np.zeros(n_steps + 1)

    theta_curr_deg = float(theta_start_deg)
    M_curr         = float(M_start)

    for i in range(n_steps):
        # 沿当前斜面方向前进 ds=1
        xs[i + 1] = xs[i] + math.cos(math.radians(theta_curr_deg))
        ys[i + 1] = ys[i] + math.sin(math.radians(theta_curr_deg))
        # 更新倾角和马赫数
        theta_curr_deg += d_theta_deg
        nu_new = nu_start - (i + 1) * d_theta_rad
        M_curr = M_from_pm_angle(nu_new, gamma)

    theta_final_deg = float(theta_start_deg) + float(delta_theta_deg)
    return xs, ys, float(M_curr), theta_final_deg
