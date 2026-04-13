"""
core/atmosphere.py
==================
ICAO 标准大气（ISA）模型，供所有构型共用。

公开接口
--------
- ``isa_atmosphere(H)``            → (T, p, rho, a)           函数式 API
- ``capture_area(mdot, M0, H)``    → A_cap (m²)                静条件公式
- ``capture_area_from_total(...)`` → A_cap (m²)                总条件公式
- ``ISAAtmosphere(H)``             面向对象封装，含总温总压方法

单位约定
--------
- 高度 H：米（m）；h_km：千米（km）
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


# ---------------------------------------------------------------------------
# 捕获面积（总条件公式）
# ---------------------------------------------------------------------------

def capture_area_from_total(
    mdot: float,
    M: float,
    T0: float,
    P0: float,
    gamma: float = 1.4,
    R: float = 287.05,
) -> float:
    """由来流总条件（T0、P0）和马赫数计算捕获面积。

    A_cap = ṁ / [P0 · √(γ/(R·T0)) · M · (1+(γ-1)/2·M²)^(-(γ+1)/(2(γ-1)))]

    该公式来自连续方程的总条件形式，与 ``capture_area`` 的静条件形式在
    等熵条件下数值等价，但接口更适合已知总条件的场合。

    Parameters
    ----------
    mdot : float
        质量流量，kg/s。
    M : float
        来流马赫数。
    T0 : float
        来流总温，K。
    P0 : float
        来流总压，Pa。
    gamma : float
        比热比，默认 1.4。
    R : float
        气体常数，J/(kg·K)，默认 287.05。

    Returns
    -------
    float
        捕获面积 A_cap，m²。

    Raises
    ------
    ValueError
        若 mdot ≤ 0、M ≤ 0、T0 ≤ 0 或 P0 ≤ 0。
    """
    if mdot <= 0.0:
        raise ValueError(f"质量流量 mdot 必须 > 0，当前 {mdot}。")
    if M <= 0.0:
        raise ValueError(f"马赫数 M 必须 > 0，当前 {M}。")
    if T0 <= 0.0:
        raise ValueError(f"总温 T0 必须 > 0，当前 {T0}。")
    if P0 <= 0.0:
        raise ValueError(f"总压 P0 必须 > 0，当前 {P0}。")

    exp = -(gamma + 1.0) / (2.0 * (gamma - 1.0))
    mass_flux = P0 * math.sqrt(gamma / (R * T0)) * M * (1.0 + (gamma - 1.0) / 2.0 * M ** 2) ** exp
    return float(mdot / mass_flux)


# ---------------------------------------------------------------------------
# ISAAtmosphere 面向对象封装
# ---------------------------------------------------------------------------

class ISAAtmosphere:
    """ICAO 标准大气对象，封装给定高度处的静态和总条件参数。

    Parameters
    ----------
    H : float
        飞行高度，单位：**米（m）**。支持范围：0 ≤ H ≤ 20 000 m。
        若使用千米单位，请先乘以 1000，或直接传入 ``h_km * 1000``。
    gamma : float
        比热比，默认 1.4。

    Attributes
    ----------
    H         : float   高度（m）
    gamma     : float   比热比
    T_static  : float   静温（K）
    p_static  : float   静压（Pa）
    rho       : float   密度（kg/m³）
    a         : float   声速（m/s）

    Examples
    --------
    >>> atm = ISAAtmosphere(20000.0)
    >>> atm.T_static        # 216.65 K
    >>> atm.total_temperature(2.0)   # 总温
    >>> atm.total_pressure(2.0)      # 总压
    """

    def __init__(self, H: float, gamma: float = 1.4) -> None:
        self.H = float(H)
        self.gamma = float(gamma)
        self.T_static, self.p_static, self.rho, self.a = isa_atmosphere(H, gamma)

    # ------------------------------------------------------------------
    # 总条件（等熵）
    # ------------------------------------------------------------------

    def total_temperature(self, M: float) -> float:
        """来流总温 T₀ = T·(1 + (γ-1)/2·M²)。"""
        return self.T_static * (1.0 + (self.gamma - 1.0) / 2.0 * M ** 2)

    def total_pressure(self, M: float) -> float:
        """来流总压 P₀ = p·(T₀/T)^(γ/(γ-1))。

        等效：P₀ = p · (1 + (γ-1)/2·M²)^(γ/(γ-1))
        """
        ratio = 1.0 + (self.gamma - 1.0) / 2.0 * M ** 2
        return self.p_static * ratio ** (self.gamma / (self.gamma - 1.0))

    def velocity(self, M: float) -> float:
        """来流速度 V = M·a（m/s）。"""
        return M * self.a

    def capture_area(self, mdot: float, M: float) -> float:
        """由质量流量计算捕获面积 A_cap = ṁ/(ρ·V)（m²）。

        Parameters
        ----------
        mdot : float  质量流量（kg/s），必须 > 0。
        M    : float  马赫数，必须 > 0。
        """
        if mdot <= 0.0:
            raise ValueError(f"质量流量 mdot 必须 > 0，当前 {mdot}。")
        if M <= 0.0:
            raise ValueError(f"马赫数 M 必须 > 0，当前 {M}。")
        return float(mdot / (self.rho * self.velocity(M)))

    def __repr__(self) -> str:
        return (
            f"ISAAtmosphere(H={self.H:.0f} m, "
            f"T={self.T_static:.2f} K, "
            f"p={self.p_static:.1f} Pa, "
            f"rho={self.rho:.5f} kg/m³, "
            f"a={self.a:.2f} m/s)"
        )
