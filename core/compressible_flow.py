"""
core/compressible_flow.py
=========================
通用可压缩流动关系式，供所有进气道构型共用。

单位约定
--------
- 角度：带 ``_rad`` 后缀的参数/返回值用弧度，带 ``_deg`` 后缀的用度。
- 压力比、温度比均为无量纲。
- 马赫数无量纲。

禁止在本模块中导入 geometry/、gui/ 或 inlets/ 的任何内容。
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# 激波关系式（法向马赫数接口）
# ---------------------------------------------------------------------------

def shock_pt_ratio(M_n: float, gamma: float = 1.4) -> float:
    """激波总压比 p_t2 / p_t1（Rayleigh Pitot 公式）。

    Parameters
    ----------
    M_n : float
        法向马赫数。
        - 正激波：M_n = M（来流马赫数）。
        - 斜激波：M_n = M * sin(beta)，beta 为激波角（弧度）。
    gamma : float
        比热比，默认 1.4。

    Returns
    -------
    float
        总压比 p_t2 / p_t1，范围 (0, 1]。

    Notes
    -----
    验证基准（gamma=1.4）：
        M_n=2.0  → 0.7209
        M_n=1.40 → 0.9582
        M_n=1.30 → 0.9794
    """
    A = (
        (gamma + 1) * M_n ** 2 / (2 + (gamma - 1) * M_n ** 2)
    ) ** (gamma / (gamma - 1))
    B = (
        (gamma + 1) / (2 * gamma * M_n ** 2 - (gamma - 1))
    ) ** (1 / (gamma - 1))
    return float(A * B)


def shock_p_ratio(M_n: float, gamma: float = 1.4) -> float:
    """激波静压比 p2 / p1。

    Parameters
    ----------
    M_n : float
        法向马赫数（含义同 :func:`shock_pt_ratio`）。
    gamma : float
        比热比，默认 1.4。

    Returns
    -------
    float
        静压比 p2 / p1 ≥ 1。
    """
    return float(1 + 2 * gamma / (gamma + 1) * (M_n ** 2 - 1))


def shock_T_ratio(M_n: float, gamma: float = 1.4) -> float:
    """激波静温比 T2 / T1。

    Parameters
    ----------
    M_n : float
        法向马赫数（含义同 :func:`shock_pt_ratio`）。
    gamma : float
        比热比，默认 1.4。

    Returns
    -------
    float
        静温比 T2 / T1 ≥ 1。
    """
    p_ratio = shock_p_ratio(M_n, gamma)
    # 由兰金-雨果尼奥关系：T2/T1 = (p2/p1) * (2 + (γ-1)*M_n²) / (M_n² * (2γ - (γ-1)))
    numerator = p_ratio * (2 + (gamma - 1) * M_n ** 2)
    denominator = M_n ** 2 * (2 * gamma - (gamma - 1))
    return float(numerator / denominator)


# ---------------------------------------------------------------------------
# 正激波后马赫数
# ---------------------------------------------------------------------------

def M2_after_normal_shock(M1: float, gamma: float = 1.4) -> float:
    """正激波后马赫数 M2。

    Parameters
    ----------
    M1 : float
        激波前马赫数（> 1）。
    gamma : float
        比热比，默认 1.4。

    Returns
    -------
    float
        激波后马赫数 M2 < 1。

    Notes
    -----
    验证：M1=2.0 → M2 ≈ 0.5774。
    """
    numerator = 1 + (gamma - 1) / 2 * M1 ** 2
    denominator = gamma * M1 ** 2 - (gamma - 1) / 2
    return float(math.sqrt(numerator / denominator))


# ---------------------------------------------------------------------------
# 等熵关系式
# ---------------------------------------------------------------------------

def isentropic_T_ratio(M: float, gamma: float = 1.4) -> float:
    """等熵总温比 T0 / T = 1 + (γ-1)/2 * M²。

    Parameters
    ----------
    M : float
        当地马赫数。
    gamma : float
        比热比，默认 1.4。

    Returns
    -------
    float
        总温与静温之比 T0/T ≥ 1。
    """
    return float(1 + (gamma - 1) / 2 * M ** 2)


def isentropic_p_ratio(M: float, gamma: float = 1.4) -> float:
    """等熵总压比 p0 / p = (1 + (γ-1)/2 * M²)^(γ/(γ-1))。

    Parameters
    ----------
    M : float
        当地马赫数。
    gamma : float
        比热比，默认 1.4。

    Returns
    -------
    float
        总压与静压之比 p0/p ≥ 1。
    """
    return float(isentropic_T_ratio(M, gamma) ** (gamma / (gamma - 1)))


def isentropic_M_from_pt_ratio(
    pt_ratio: float,
    M_init: float,
    gamma: float = 1.4,
) -> float:
    """由等熵总压比 p0/p 反算马赫数（用 brentq 数值求解）。

    Parameters
    ----------
    pt_ratio : float
        目标总压比 p0/p（> 1）。
    M_init : float
        初始猜测马赫数，用于确定搜索区间方向（> 0）。
    gamma : float
        比热比，默认 1.4。

    Returns
    -------
    float
        对应马赫数 M > 0，满足 isentropic_p_ratio(M) ≈ pt_ratio。
    """
    def _residual(M: float) -> float:
        return isentropic_p_ratio(M, gamma) - pt_ratio

    # 搜索区间：[1e-6, 50]，pt_ratio 单调递增
    return float(brentq(_residual, 1e-6, 50.0, xtol=1e-10))


# ---------------------------------------------------------------------------
# θ-β-M 斜激波关系式（二元构型，方程 11-13）
# ---------------------------------------------------------------------------

def theta_from_beta_M(
    beta_rad: float,
    M: float,
    gamma: float = 1.4,
) -> float:
    """由激波角 β 和来流马赫数 M 求楔角 θ（弱激波解）。

    Parameters
    ----------
    beta_rad : float
        激波角，单位：弧度。
    M : float
        来流马赫数（> 1）。
    gamma : float
        比热比，默认 1.4。

    Returns
    -------
    float
        楔角 θ，单位：弧度。
    """
    sin_b = math.sin(beta_rad)
    cos_2b = math.cos(2 * beta_rad)
    numerator = 2 * (M ** 2 * sin_b ** 2 - 1) / math.tan(beta_rad)
    denominator = M ** 2 * (gamma + cos_2b) + 2
    return float(math.atan(numerator / denominator))


def beta_from_theta_M(
    theta_deg: float,
    M: float,
    gamma: float = 1.4,
) -> float:
    """由楔角 θ 和来流马赫数 M 数值反解弱激波角 β（brentq）。

    Parameters
    ----------
    theta_deg : float
        楔角，单位：度。
    M : float
        来流马赫数（> 1）。
    gamma : float
        比热比，默认 1.4。

    Returns
    -------
    float
        弱激波角 β，单位：度。

    Notes
    -----
    θ(β) 在 [μ, 90°] 上先升后降，两端均趋近 0，最大转折角在中间某处取得。
    弱激波解在 [μ, β_at_max_theta] 区间内，在此范围内用 brentq 求根。
    """
    theta_rad = math.radians(theta_deg)

    mu_rad = math.asin(1.0 / M)   # Mach 角（下界）

    # 扫描找 theta(beta) 最大值对应的 beta（即弱/强激波分界点）
    n_scan = 5000
    betas_scan = np.linspace(mu_rad + 1e-7, math.pi / 2 - 1e-7, n_scan)
    thetas_scan = np.array([theta_from_beta_M(b, M, gamma) for b in betas_scan])
    idx_max = int(np.argmax(thetas_scan))
    beta_at_max = float(betas_scan[idx_max])

    if theta_rad > thetas_scan[idx_max]:
        raise ValueError(
            f"theta_deg={theta_deg}° 超过 M={M} 时的最大转折角 "
            f"{math.degrees(thetas_scan[idx_max]):.4f}°，无附体斜激波解。"
        )

    def _residual(beta_rad: float) -> float:
        return theta_from_beta_M(beta_rad, M, gamma) - theta_rad

    # 弱激波解在 [mu, beta_at_max] 区间
    beta_lo = mu_rad + 1e-9
    beta_hi = beta_at_max

    beta_rad = brentq(_residual, beta_lo, beta_hi, xtol=1e-12)
    return float(math.degrees(beta_rad))


def M2_after_oblique_shock(
    M1: float,
    beta_rad: float,
    theta_rad: float,
    gamma: float = 1.4,
) -> float:
    """斜激波后马赫数 M2（方程 12）。

    Parameters
    ----------
    M1 : float
        激波前来流马赫数。
    beta_rad : float
        激波角，单位：弧度。
    theta_rad : float
        气流偏转角（楔角），单位：弧度。
    gamma : float
        比热比，默认 1.4。

    Returns
    -------
    float
        激波后马赫数 M2。
    """
    M_n1 = M1 * math.sin(beta_rad)
    M_n2_sq = (
        (1 + (gamma - 1) / 2 * M_n1 ** 2)
        / (gamma * M_n1 ** 2 - (gamma - 1) / 2)
    )
    sin_deflect = math.sin(beta_rad - theta_rad)
    return float(math.sqrt(M_n2_sq / sin_deflect ** 2))


# ---------------------------------------------------------------------------
# 最大转折角
# ---------------------------------------------------------------------------

def max_turning_angle(M: float, gamma: float = 1.4) -> float:
    """给定来流马赫数下的最大气流转折角（即最强斜激波仍附着时的楔角上限）。

    通过在激波角 β ∈ (μ, 90°) 上对 θ(β) 求最大值获得。

    Parameters
    ----------
    M : float
        来流马赫数（> 1）。
    gamma : float
        比热比，默认 1.4。

    Returns
    -------
    float
        最大转折角，单位：度。
    """
    mu_rad = math.asin(1.0 / M)
    # 在 [mu+eps, pi/2-eps] 上对 theta_from_beta_M 取最大值（数值扫描）
    betas = np.linspace(mu_rad + 1e-6, math.pi / 2 - 1e-6, 10_000)
    thetas = np.array([theta_from_beta_M(b, M, gamma) for b in betas])
    return float(math.degrees(float(thetas.max())))


# ---------------------------------------------------------------------------
# 流量函数
# ---------------------------------------------------------------------------

def mass_flow_function(M: float, gamma: float = 1.4) -> float:
    """流量函数 φ(M)，用于面积-马赫数关系中的质量守恒。

    φ = M * sqrt(γ/R) * (1 + (γ-1)/2 * M²)^(-(γ+1)/(2(γ-1)))

    其中 R = 287.05 J/(kg·K)。γ=1.4 时指数 = -3.0。

    Parameters
    ----------
    M : float
        马赫数（≥ 0）。
    gamma : float
        比热比，默认 1.4。

    Returns
    -------
    float
        流量函数值（量纲：1/√K 量级，但比值无量纲）。

    Notes
    -----
    用于质量守恒：A1 * φ(M1) = A2 * φ(M2)（同一总压、总温条件下）。
    """
    R = 287.05
    exp = -(gamma + 1) / (2 * (gamma - 1))
    return float(M * (gamma / R) ** 0.5 * (1 + (gamma - 1) / 2 * M ** 2) ** exp)


# ---------------------------------------------------------------------------
# 模块级数值自验证（import 时执行）
# ---------------------------------------------------------------------------

def _self_verify() -> None:
    tol = 0.0005
    cases = [
        (2.0, 0.7209),
        (1.40, 0.9582),
        (1.30, 0.9794),
    ]
    for M_n, expected in cases:
        result = shock_pt_ratio(M_n)
        assert abs(result - expected) < tol, (
            f"shock_pt_ratio({M_n}) = {result:.6f}，期望 {expected}±{tol}"
        )


_self_verify()
