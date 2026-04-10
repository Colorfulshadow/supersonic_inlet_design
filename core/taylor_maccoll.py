"""
core/taylor_maccoll.py
======================
Taylor-Maccoll 方程求解器，用于轴对称锥形激波计算。

物理背景
--------
超声速流绕锥体流动为等熵锥形流（conical flow）。速度用 V_max 归一化：
    V_max² = 2/(γ-1) * a_0²
其中 a_0 为滞止声速。归一化后的速度分量 (V_r, V_θ) 满足：
    V_r² + V_θ² ≤ 1
    a² = (γ-1)/2 * (1 - V_r² - V_θ²)

坐标系
------
- θ 从锥轴（θ=0）量起，锥半角 δ_c，激波角 β_c（弧度）
- V_r：沿球径方向（向外为正）
- V_θ：垂直于 V_r，θ 增大方向为正；锥面无穿透条件 V_θ(δ_c)=0

禁止在本模块中导入 geometry/、gui/ 或 inlets/ 的任何内容。
"""

from __future__ import annotations

import math

from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from core.compressible_flow import (
    M2_after_oblique_shock,
    shock_pt_ratio,
    theta_from_beta_M,
)


# ---------------------------------------------------------------------------
# ODE 右端项
# ---------------------------------------------------------------------------

def taylor_maccoll_ode(
    theta: float,
    y: list[float],
    gamma: float = 1.4,
) -> list[float]:
    """Taylor-Maccoll ODE 右端项。

    Parameters
    ----------
    theta : float
        当前流场角度（弧度），从锥轴算起。
    y : list[float]
        [V_r, V_θ]，两个速度分量，均以 V_max 归一化（无量纲）。
    gamma : float
        比热比，默认 1.4。

    Returns
    -------
    list[float]
        [dV_r/dθ, dV_θ/dθ]。

    Notes
    -----
    由无旋条件得 dV_r/dθ = V_θ。
    由连续方程 + 欧拉方程化简（Anderson《现代可压缩流》eq.10.15）：

        dV_θ/dθ = [V_r·V_θ² - a²·(2·V_r + V_θ·cot θ)] / (a² - V_θ²)

    其中 a² = (γ-1)/2·(1 - V_r² - V_θ²)。
    """
    V_r, V_theta = y[0], y[1]

    a_sq = (gamma - 1) / 2.0 * (1.0 - V_r ** 2 - V_theta ** 2)

    # 防止数值越界（超声速极限附近 a² 可能微小负数）
    if a_sq < 0.0:
        a_sq = 0.0

    cot_theta = math.cos(theta) / math.sin(theta)

    numerator = V_r * V_theta ** 2 - a_sq * (2.0 * V_r + V_theta * cot_theta)
    denominator = a_sq - V_theta ** 2

    dV_r = V_theta
    dV_theta = numerator / denominator

    return [dV_r, dV_theta]


# ---------------------------------------------------------------------------
# 内部辅助：激波面处初始条件
# ---------------------------------------------------------------------------

def _shock_initial_conditions(
    beta_c_rad: float,
    M0: float,
    gamma: float,
) -> tuple[float, float]:
    """由斜激波关系计算激波面（θ=β_c）处的 V_r、V_θ 初始值。

    流程
    ----
    1. 由 θ-β-M 关系求激波后气流偏转角 θ_c（等于锥面角，shock-on-lip 假设）。
    2. 由斜激波关系求激波后马赫数 M2。
    3. 将 M2 转换为 V_max 归一化速度 V2_hat。
    4. 分解为径向和切向分量。

    Parameters
    ----------
    beta_c_rad : float
        锥形激波角，弧度。
    M0 : float
        来流马赫数。
    gamma : float
        比热比。

    Returns
    -------
    tuple[float, float]
        (V_r, V_θ) at θ = β_c。V_θ < 0（气流朝向锥轴方向）。
    """
    theta_c_rad = theta_from_beta_M(beta_c_rad, M0, gamma)
    M2 = M2_after_oblique_shock(M0, beta_c_rad, theta_c_rad, gamma)

    # 归一化速度幅值：V/V_max = sqrt((γ-1)*M² / (2 + (γ-1)*M²))
    V2_hat = math.sqrt((gamma - 1) * M2 ** 2 / (2.0 + (gamma - 1) * M2 ** 2))

    # 激波后气流方向与锥轴夹角 = θ_c
    # 径向方向（β_c）与气流方向（θ_c）之间夹角 = β_c - θ_c
    deflection = beta_c_rad - theta_c_rad
    V_r = V2_hat * math.cos(deflection)
    V_theta = -V2_hat * math.sin(deflection)   # 负号：气流向轴，V_θ 递减方向

    return V_r, V_theta


def _integrate_to_cone(
    beta_c_rad: float,
    delta_c_rad: float,
    M0: float,
    gamma: float,
) -> tuple[float, float]:
    """从激波面积分 T-M ODE 到锥面，返回锥面处的 (V_r, V_θ)。

    Parameters
    ----------
    beta_c_rad : float
        激波角（弧度），积分起点。
    delta_c_rad : float
        锥半角（弧度），积分终点。
    M0 : float
        来流马赫数。
    gamma : float
        比热比。

    Returns
    -------
    tuple[float, float]
        (V_r, V_θ) at θ = δ_c。
    """
    V_r0, V_theta0 = _shock_initial_conditions(beta_c_rad, M0, gamma)

    span = (beta_c_rad, delta_c_rad)   # θ 从大到小积分（solve_ivp 支持反向）
    y0 = [V_r0, V_theta0]

    sol = solve_ivp(
        taylor_maccoll_ode,
        span,
        y0,
        args=(gamma,),
        method="RK45",
        rtol=1e-9,
        atol=1e-11,
        max_step=(beta_c_rad - delta_c_rad) / 200,
        dense_output=False,
    )

    return float(sol.y[0, -1]), float(sol.y[1, -1])


# ---------------------------------------------------------------------------
# 主求解函数
# ---------------------------------------------------------------------------

def solve_taylor_maccoll(
    delta_c_deg: float,
    M0: float,
    gamma: float = 1.4,
) -> dict:
    """求解 Taylor-Maccoll 问题：给定锥半角，求激波角及锥面马赫数。

    求解策略
    --------
    在 β_c ∈ (δ_c, 90°) 上二分搜索使 V_θ(δ_c) = 0 的激波角（scipy.brentq）。
    积分采用 solve_ivp（RK45），从激波面（θ=β_c）向锥面（θ=δ_c）推进。

    Parameters
    ----------
    delta_c_deg : float
        锥半角，单位：度。
    M0 : float
        来流马赫数。
    gamma : float
        比热比，默认 1.4。

    Returns
    -------
    dict
        包含以下字段：

        - ``beta_c_deg``  (float)：锥形激波角，度。
        - ``M_cone``      (float)：锥面马赫数。
        - ``V_r_cone``    (float)：锥面径向速度（V_max 归一化）。
        - ``V_theta_cone``(float)：锥面切向速度（应接近 0）。

    Raises
    ------
    ValueError
        若在搜索区间内找不到符号变化（锥角超出附体激波范围）。

    Notes
    -----
    验证基准（γ=1.4，M₀=2.0，δ_c=31.37°）：
        β_c ≈ 50.79°，M_cone ≈ 1.55。
    """
    delta_c_rad = math.radians(delta_c_deg)

    def _residual(beta_c_rad: float) -> float:
        _, V_theta_cone = _integrate_to_cone(beta_c_rad, delta_c_rad, M0, gamma)
        return V_theta_cone   # 目标：= 0（无穿透条件）

    # V_θ(δ_c) 随 β_c 不单调（先负→正→负），物理解（附体弱激波）位于第一个
    # 零交叉处（最小 β_c）。通过粗扫描定位符号变化区间，再用 brentq 精化。
    # 物理下界：激波角须大于 Mach 角（M_n=M0*sin(beta)>1）且大于 δ_c
    mu_rad = math.asin(1.0 / M0)
    # 偏移量用弧度 1e-7（≈ 0.000006°），足够小以不跨过极弱激波的符号变化区间
    search_lo = max(delta_c_rad, mu_rad) + 1e-7

    n_scan = 300
    import numpy as _np
    betas_scan = _np.linspace(search_lo, math.pi / 2 - 1e-4, n_scan)
    f_scan = _np.array([_residual(b) for b in betas_scan])

    # 找到第一个符号变化（负→正，对应弱激波解）
    sign_changes = _np.where(_np.diff(_np.sign(f_scan)))[0]
    if len(sign_changes) == 0:
        raise ValueError(
            f"在搜索区间内未找到 V_θ=0 的符号变化，"
            f"锥角 δ_c={delta_c_deg}° 可能超出附体激波范围（M₀={M0}）。"
        )

    # 取第一个符号变化（弱激波解）
    idx = sign_changes[0]
    beta_lo = float(betas_scan[idx])
    beta_hi = float(betas_scan[idx + 1])

    beta_c_rad = brentq(_residual, beta_lo, beta_hi, xtol=1e-10, maxiter=200)

    V_r_cone, V_theta_cone = _integrate_to_cone(beta_c_rad, delta_c_rad, M0, gamma)

    # 由归一化速度计算锥面马赫数
    V_sq = V_r_cone ** 2 + V_theta_cone ** 2
    a_sq = (gamma - 1) / 2.0 * (1.0 - V_sq)
    M_cone = math.sqrt(V_sq / a_sq) if a_sq > 0 else float("inf")

    return {
        "beta_c_deg": math.degrees(beta_c_rad),
        "M_cone": M_cone,
        "V_r_cone": V_r_cone,
        "V_theta_cone": V_theta_cone,
    }


# ---------------------------------------------------------------------------
# 锥形激波总压比
# ---------------------------------------------------------------------------

def cone_shock_pt_ratio(
    M0: float,
    beta_c_rad: float,
    gamma: float = 1.4,
) -> float:
    """锥形激波总压比 p_t2 / p_t1。

    复用 :func:`core.compressible_flow.shock_pt_ratio`，
    以法向马赫数 M_n = M₀·sin(β_c) 作为入参。

    Parameters
    ----------
    M0 : float
        来流马赫数。
    beta_c_rad : float
        锥形激波角，单位：弧度。
    gamma : float
        比热比，默认 1.4。

    Returns
    -------
    float
        总压比 p_t2/p_t1，范围 (0, 1]。
        对于锥形激波（斜激波），通常 > 0.98（M₀=2.0 时）。
    """
    M_n = M0 * math.sin(beta_c_rad)
    return shock_pt_ratio(M_n, gamma)
