"""
core/atmosphere.py
==================
ICAO 标准大气（ISA）模型，供所有构型共用。

单位约定
--------
- 高度 H：米（m）
- 温度 T：开尔文（K）
- 压力 p：帕斯卡（Pa）
- 密度 rho：千克/立方米（kg/m³）
- 声速 a：米/秒（m/s）
- 质量流量 mdot：千克/秒（kg/s）
- 面积 A_cap：平方米（m²）

禁止在本模块中导入 geometry/、gui/ 或 inlets/ 的任何内容。
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# 物理常数
# ---------------------------------------------------------------------------

_R: float = 287.05      # 空气气体常数，J/(kg·K)
_GAMMA: float = 1.4     # 比热比（标准空气）

# ISA 标准大气分段常数
_T0: float = 288.15     # 海平面标准温度，K
_P0: float = 101325.0   # 海平面标准压力，Pa
_L: float = 0.0065      # 对流层温度递减率，K/m

_H_TROP: float = 11000.0   # 对流层顶高度，m
_T_STRAT: float = 216.65   # 平流层等温温度，K

# 平流层底部标准压力（由对流层公式在 H=11000 m 处取得）
_P_11: float = 22632.1     # Pa（ICAO 标准值）

# 平流层压力衰减系数 g₀/(R·T_strat)
_LAMBDA: float = 9.80665 / (_R * _T_STRAT)   # ≈ 0.00015772 m⁻¹


# ---------------------------------------------------------------------------
# ISA 标准大气
# ---------------------------------------------------------------------------

def isa_atmosphere(H: float, gamma: float = 1.4) -> tuple[float, float, float, float]:
    """ICAO 标准大气，计算给定高度处的热力学参数。

    分段范围
    --------
    - 对流层：H ≤ 11 000 m（线性温度递减）
    - 平流层下层：11 000 < H ≤ 20 000 m（等温层）

    Parameters
    ----------
    H : float
        飞行高度，单位：米（m）。支持范围：0 ≤ H ≤ 20 000 m。
    gamma : float
        比热比，默认 1.4。

    Returns
    -------
    T : float
        静温，K。
    p : float
        静压，Pa。
    rho : float
        密度，kg/m³。
    a : float
        声速，m/s。

    Notes
    -----
    验证基准（H = 20 000 m）：
        T   = 216.65 K
        p   = 5474.9 Pa
        ρ   = 0.08803 kg/m³
        a   = 295.07 m/s
    """
    if H <= _H_TROP:
        # 对流层：线性温度分布
        T = _T0 - _L * H
        p = _P0 * (T / _T0) ** (9.80665 / (_L * _R))   # 指数 = g₀/(L·R) ≈ 5.2561
    else:
        # 平流层下层：等温层（T = 216.65 K）
        T = _T_STRAT
        p = _P11_at_tropopause() * math.exp(-_LAMBDA * (H - _H_TROP))

    rho = p / (_R * T)
    a = math.sqrt(gamma * _R * T)
    return float(T), float(p), float(rho), float(a)


def _P11_at_tropopause() -> float:
    """对流层顶（H = 11 000 m）处的标准压力。

    直接使用 ICAO 标准值 22632.1 Pa，避免由对流层公式累积误差。
    """
    return _P_11


# ---------------------------------------------------------------------------
# 捕获面积正向计算
# ---------------------------------------------------------------------------

def capture_area(
    mdot: float,
    M0: float,
    H: float,
    gamma: float = 1.4,
) -> float:
    """由质量流量、设计马赫数和飞行高度计算真实捕获面积。

    A_cap = ṁ / (ρ∞ · V∞) = ṁ / (ρ∞ · M₀ · a∞)

    Parameters
    ----------
    mdot : float
        质量流量，单位：kg/s。
    M0 : float
        设计马赫数（自由来流）。
    H : float
        飞行高度，单位：m。
    gamma : float
        比热比，默认 1.4。

    Returns
    -------
    float
        捕获面积 A_cap，单位：m²。

    Notes
    -----
    设计工况（M₀=2.0，H=20 000 m，ṁ=100 kg/s）验证值：
        V∞    = 2.0 × 295.07 ≈ 590.14 m/s
        A_cap = 100 / (0.08803 × 590.14) ≈ 1.924 m²
    """
    _, _, rho, a = isa_atmosphere(H, gamma)
    V_inf = M0 * a
    return float(mdot / (rho * V_inf))
