"""
inlets/external_2d/aero_design.py
==================================
二元外压式进气道气动设计：N 级斜激波（Oswatitsch 等强度准则）+ 终端正激波。

Oswatitsch 准则
---------------
各级斜激波法向马赫数相等（M_n1 = M_n2 = … = M_nN），在给定来流 M0、
终端激波前马赫数 M_EX 和级数 N 的前提下使总压恢复最大化。

求解策略
--------
对给定 M_EX，用 brentq 搜索公共法向马赫数 M_n∈(1, M_EX)，使 N 级等强度
斜激波从 M0 出发恰好到达 M_EX。

若 M_EX 未指定（自动模式），则用全 Oswatitsch 准则：N 级斜激波 + 终端正激波
全部强度相等，即固定点条件 simulate(M_n) = M_n，同样用 brentq 求解。

单位约定
--------
- 角度：楔角列表以度（°）返回；内部计算用弧度
- 总压归一化：st0.p_t = 1.0

禁止在本模块重复实现激波总压比，必须调用 core.compressible_flow.shock_pt_ratio。
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from scipy.optimize import brentq

from core.atmosphere import ISAAtmosphere
from core.compressible_flow import (
    M2_after_normal_shock,
    M2_after_oblique_shock,
    max_turning_angle,
    shock_pt_ratio,
    theta_from_beta_M,
)
from core.flow_stations import FlowState, InletFlowStations


# ---------------------------------------------------------------------------
# 内部辅助：N 级等 M_n 斜激波链
# ---------------------------------------------------------------------------

def _oblique_chain(
    M0: float,
    N: int,
    M_n: float,
    gamma: float = 1.4,
) -> tuple:
    """运行 N 级法向马赫数均为 M_n 的斜激波链。

    Parameters
    ----------
    M0 : float
        入口马赫数。
    N : int
        级数。
    M_n : float
        公共法向马赫数（须 > 1 且 < 当前 M）。
    gamma : float
        比热比。

    Returns
    -------
    tuple
        ``(M_final, thetas_deg, pt)``——出口马赫数、各级楔角（度）、总压比乘积。
        若任意一级无效（M_n >= M 或流场进入亚声速），返回 ``(inf, [], 0.0)``。
    """
    M = float(M0)
    thetas: List[float] = []
    pt = 1.0
    for _ in range(N):
        if M_n >= M or M_n <= 1.0:
            return math.inf, [], 0.0
        beta = math.asin(M_n / M)
        theta_rad = theta_from_beta_M(beta, M, gamma)
        if theta_rad <= 0.0:
            return math.inf, [], 0.0
        pt *= shock_pt_ratio(M_n, gamma)
        M_new = M2_after_oblique_shock(M, beta, theta_rad, gamma)
        if M_new <= 1.0:
            return math.inf, [], 0.0
        M = M_new
        thetas.append(math.degrees(theta_rad))
    return M, thetas, pt


def _scan_upper_bound(
    M0: float,
    N: int,
    target: float,
    gamma: float,
    n_scan: int = 500,
) -> float:
    """扫描找到第一个使 simulate(M_n) < target 的 M_n，作为 brentq 上界。

    Parameters
    ----------
    target : float
        目标出口马赫数（对固定点搜索传入 M_n 本身，用 lambda 包装）。
    """
    scan = np.linspace(1.001, M0 - 0.001, n_scan)
    prev_above = True
    for M_n_try in scan:
        M_f, _, _ = _oblique_chain(M0, N, M_n_try, gamma)
        if M_f == math.inf:
            break
        if M_f < target:
            return float(M_n_try)
        prev_above = M_f >= target
    raise ValueError(
        f"无法找到 brentq 上界：M0={M0}, N={N}, target={target}。"
        "请检查 M_EX 是否在 (1, M0) 范围内，且 N 级设计在物理上可行。"
    )


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def oswatitsch_angles(
    M0: float,
    N_stages: int,
    M_EX: float,
    gamma: float = 1.4,
) -> List[float]:
    """Oswatitsch 准则：等强度斜激波楔角序列。

    在给定 M0、M_EX 和 N_stages 的条件下，搜索公共法向马赫数 M_n，
    使 N 级等 M_n 斜激波将来流从 M0 精确减速至 M_EX。

    Parameters
    ----------
    M0 : float
        来流马赫数（> 1）。
    N_stages : int
        外部斜激波级数（≥ 1）。
    M_EX : float
        终端正激波前目标马赫数（1 < M_EX < M0）。
    gamma : float
        比热比，默认 1.4。

    Returns
    -------
    list of float
        长度为 N_stages 的楔角列表（度），按流向顺序（第 1 级 → 第 N 级）。

    Notes
    -----
    验证基准（M0=2.0，N=3，M_EX=1.40，γ=1.4）：

    - ``sum(angles)`` ≈ 17.22°（总外压偏转，对应 Slater 2023 Table 2 "theta_stg1"=17.34°±0.5°）
    - 各级角度约 5.4°、5.8°、6.0°（等 M_n 导致各级角度略有不同）

    关于 Slater 2023 基准 "theta_stg1 = 17.34°"
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    该值为三级楔角之和（总外压偏转角），不是第一级单独的楔角（≈5.4°）。
    """
    if N_stages < 1:
        raise ValueError(f"N_stages 必须 ≥ 1，当前 {N_stages}。")
    if not (1.0 < M_EX < M0):
        raise ValueError(f"需要 1 < M_EX < M0，当前 M_EX={M_EX}, M0={M0}。")

    M_n_lo = 1.0 + 1e-9

    def f(M_n: float) -> float:
        M_f, _, _ = _oblique_chain(M0, N_stages, M_n, gamma)
        return (M_f if M_f != math.inf else M0) - M_EX

    M_n_hi = _scan_upper_bound(M0, N_stages, M_EX, gamma)
    M_n_opt = brentq(f, M_n_lo, M_n_hi, xtol=1e-12, maxiter=200)

    _, thetas, _ = _oblique_chain(M0, N_stages, M_n_opt, gamma)
    return thetas


def design_external_2d(
    M0: float,
    N_stages: int = 3,
    M_EX: Optional[float] = None,
    gamma: float = 1.4,
    h_km: Optional[float] = None,
    m_dot: Optional[float] = None,
) -> InletFlowStations:
    """二元外压式进气道气动设计（Oswatitsch 最优准则）。

    Parameters
    ----------
    M0 : float
        来流马赫数（站位 0），须 > 1.0。
    N_stages : int
        外部斜激波级数，默认 3。
    M_EX : float, optional
        终端正激波前马赫数（站位 EX）。

        - 若给定：由 Oswatitsch 准则为该 M_EX 求各级楔角；
        - 若为 ``None``：全 Oswatitsch 优化——N 级斜激波与终端正激波强度
          全部相等，通过固定点条件 ``simulate(M_n) = M_n`` 自动求解最优 M_EX。
    gamma : float
        比热比，默认 1.4。
    h_km : float, optional
        飞行高度，单位：千米（km）。若同时提供 ``m_dot``，则附加真实物理量。
    m_dot : float, optional
        质量流量，单位：kg/s。仅在同时提供 ``h_km`` 时生效。

    Returns
    -------
    InletFlowStations
        填充以下站位：

        - **st0**  — 来流，M=M0，p_t=1.0（归一化），T_t=1.0
        - **stEX** — N 级斜激波后，p_t = 各级 shock_pt_ratio 之积
        - **stNS** — 终端正激波后
        - **st1**  — 与 stNS 相同（无唇口损失）
        - **st2**  — 与 st1 相同（扩压段损失 sigma_diff=1.0）

        同时在返回对象上附加动态属性 ``wedge_angles``（楔角列表，度）。

    Raises
    ------
    ValueError
        若 M0 ≤ 1.0 或 N_stages < 1。

    Notes
    -----
    验证基准（M0=2.0，N=3，M_EX=1.40，γ=1.4）：

    - ``stEX.M`` = 1.40 ± 0.01
    - ``total_pressure_recovery()`` ≥ 0.930
    - ``sum(wedge_angles)`` ≈ 17.22° （Slater 2023 "theta_stg1"=17.34°±0.5°）
    """
    if M0 <= 1.0:
        raise ValueError(f"来流马赫数 M0 须 > 1.0，当前 M0={M0}。")
    if N_stages < 1:
        raise ValueError(f"N_stages 须 ≥ 1，当前 {N_stages}。")

    # ------------------------------------------------------------------
    # 1. 确定 M_EX
    # ------------------------------------------------------------------
    if M_EX is None:
        # 全 Oswatitsch：固定点 simulate(M_n) = M_n
        def f_fp(M_n: float) -> float:
            M_f, _, _ = _oblique_chain(M0, N_stages, M_n, gamma)
            return (M_f if M_f != math.inf else M0) - M_n

        # 上界：找第一个使 simulate(M_n) < M_n 的点
        M_n_hi_fp = _scan_upper_bound_fp(M0, N_stages, gamma)
        M_EX = float(brentq(f_fp, 1.0 + 1e-9, M_n_hi_fp, xtol=1e-12, maxiter=200))

    # ------------------------------------------------------------------
    # 2. 求各级楔角及总压
    # ------------------------------------------------------------------
    def f_chain(M_n: float) -> float:
        M_f, _, _ = _oblique_chain(M0, N_stages, M_n, gamma)
        return (M_f if M_f != math.inf else M0) - M_EX

    M_n_hi = _scan_upper_bound(M0, N_stages, M_EX, gamma)
    M_n_opt = brentq(f_chain, 1.0 + 1e-9, M_n_hi, xtol=1e-12, maxiter=200)

    M_final, thetas, pt_EX = _oblique_chain(M0, N_stages, M_n_opt, gamma)

    # ------------------------------------------------------------------
    # 3. 终端正激波
    # ------------------------------------------------------------------
    pt_NS = pt_EX * shock_pt_ratio(M_EX, gamma)
    M_NS = M2_after_normal_shock(M_EX, gamma)

    # ------------------------------------------------------------------
    # 4. 组装流场站位
    # ------------------------------------------------------------------
    st0  = FlowState(M=M0,    p_t=1.0,   T_t=1.0, label="0")
    stEX = FlowState(M=M_EX,  p_t=pt_EX, T_t=1.0, label="EX")
    stNS = FlowState(M=M_NS,  p_t=pt_NS, T_t=1.0, label="NS")
    st1  = FlowState(M=M_NS,  p_t=pt_NS, T_t=1.0, label="1")
    st2  = FlowState(M=M_NS,  p_t=pt_NS, T_t=1.0, label="2")

    stations = InletFlowStations(st0=st0, stEX=stEX, stNS=stNS, st1=st1, st2=st2)
    stations.wedge_angles = thetas  # 动态属性，方便外部访问

    # ------------------------------------------------------------------
    # 可选：附加真实物理量
    # ------------------------------------------------------------------
    if h_km is not None and m_dot is not None:
        atm = ISAAtmosphere(h_km * 1000.0, gamma)
        stations.attach_physical_conditions(atm, M0, m_dot)

    return stations


def _scan_upper_bound_fp(
    M0: float,
    N: int,
    gamma: float,
    n_scan: int = 500,
) -> float:
    """固定点搜索专用上界扫描：找第一个使 simulate(M_n) < M_n 的点。"""
    scan = np.linspace(1.001, M0 - 0.001, n_scan)
    for M_n_try in scan:
        M_f, _, _ = _oblique_chain(M0, N, M_n_try, gamma)
        if M_f == math.inf:
            break
        if M_f < M_n_try:
            return float(M_n_try)
    raise ValueError(
        f"无法找到全 Oswatitsch 固定点搜索上界：M0={M0}, N={N}。"
    )
