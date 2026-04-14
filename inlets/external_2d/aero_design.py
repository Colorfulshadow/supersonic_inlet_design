"""
inlets/external_2d/aero_design.py
==================================
二元外压式进气道气动设计：双模式实现。

Mode 1（Oswatitsch 等强度准则）
--------------------------------
各级斜激波法向马赫数相等（M_n1 = M_n2 = … = M_nN），在给定来流 M0、
终端激波前马赫数 M_EX 和级数 N 的前提下使总压恢复最大化。

Mode 2（用户自定义楔角，正向计算）
------------------------------------
用户直接指定各级楔角列表；工具正向遍历波系，计算 M_EX 和总压恢复。
用于雷达隐身、结构约束等工程权衡场景的性能折损评估。

公开接口
--------
- ``design_external_2d(M0, N_stages, M_EX, gamma, mode, wedge_angles)``
  统一入口，mode=1（默认）走 Oswatitsch，mode=2 走用户自定义。
- ``design_external_2d_mode2(M0, wedge_angles_deg, gamma)``
  Mode 2 专用接口，正向计算楔角链。
- ``oswatitsch_angles(M0, N_stages, M_EX, gamma)``
  Mode 1 辅助：返回最优楔角序列。

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
    beta_from_theta_M,
    max_turning_angle,
    shock_pt_ratio,
    theta_from_beta_M,
)
from core.flow_stations import FlowState, InletFlowStations
from core.prandtl_meyer import M_from_pm_angle, prandtl_meyer_angle


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
    mode: int = 1,
    wedge_angles: Optional[List[float]] = None,
    theta_iso_deg: float = 0.0,
) -> InletFlowStations:
    """二元外压式进气道气动设计，统一入口。

    Parameters
    ----------
    M0 : float
        来流马赫数（站位 0），须 > 1.0。
    N_stages : int
        外部斜激波级数，默认 3（仅 Mode 1 使用）。
    M_EX : float, optional
        终端正激波前马赫数（站位 EX，仅 Mode 1 使用）。

        - 若给定：由 Oswatitsch 准则为该 M_EX 求各级楔角；
        - 若为 ``None``：全 Oswatitsch 优化，通过固定点条件自动求解。
    gamma : float
        比热比，默认 1.4。
    h_km : float, optional
        飞行高度（km）。若同时提供 ``m_dot``，则附加真实物理量。
    m_dot : float, optional
        质量流量（kg/s）。仅在同时提供 ``h_km`` 时生效。
    mode : int
        设计模式：

        - ``1``（默认）—— Oswatitsch 等强度最优准则。
        - ``2`` —— 用户自定义楔角，须同时传入 ``wedge_angles``。
    wedge_angles : list of float, optional
        Mode 2 专用：各级楔角（度）列表，如 ``[8.0, 9.0, 9.5]``。
    theta_iso_deg : float, optional
        等熵压缩段额外偏转角（度），默认 0.0（无等熵段）。
        须满足 ``theta_iso_deg < ν(M_EX)``（M_EX 时的 Prandtl-Meyer 角）。
        > 0 时在斜激波链之后插入 Prandtl-Meyer 等熵压缩，终端正激波在
        更低马赫数 M_ISO 前施加，总压恢复提高。

    Returns
    -------
    InletFlowStations
        填充以下站位：

        - **st0**   — 来流，M=M0，p_t=1.0（归一化），T_t=1.0
        - **stEX**  — N 级斜激波后
        - **stISO** — 等熵段出口（仅 theta_iso_deg > 0 时存在）
        - **stNS**  — 终端正激波后
        - **st1**   — 与 stNS 相同（无唇口损失）
        - **st2**   — 与 st1 相同（扩压段损失 sigma_diff=1.0）

        动态属性 ``wedge_angles``（楔角列表，度）。

    Raises
    ------
    ValueError
        - Mode 1：M0 ≤ 1.0 或 N_stages < 1。
        - Mode 2：M0 ≤ 1.0，或 ``wedge_angles`` 为 None/空，
          或某级楔角超过当前马赫数的最大偏转角。

    Notes
    -----
    验证基准（M0=2.0，N=3，M_EX=1.40，Mode 1，γ=1.4）：

    - ``stEX.M`` = 1.40 ± 0.01
    - ``total_pressure_recovery()`` ≥ 0.930
    - ``sum(wedge_angles)`` ≈ 17.22°（Slater 2023 "theta_stg1"=17.34°±0.5°）
    """
    # ------------------------------------------------------------------
    # Mode 2：直接委托给专用函数
    # ------------------------------------------------------------------
    if mode == 2:
        if wedge_angles is None or len(wedge_angles) == 0:
            raise ValueError(
                "mode=2 时必须通过 wedge_angles 参数传入楔角列表（非空）。"
            )
        return design_external_2d_mode2(
            M0=M0,
            wedge_angles_deg=wedge_angles,
            gamma=gamma,
            h_km=h_km,
            m_dot=m_dot,
            theta_iso_deg=theta_iso_deg,
        )

    # ------------------------------------------------------------------
    # Mode 1：Oswatitsch 等强度准则（原有逻辑）
    # ------------------------------------------------------------------
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
    # 3. 可选等熵压缩段（Prandtl-Meyer）
    # ------------------------------------------------------------------
    stISO: Optional[FlowState] = None
    M_before_NS = float(M_EX)   # 终端正激波前马赫数（默认=M_EX）

    if theta_iso_deg > 0.0:
        nu_EX = prandtl_meyer_angle(M_EX, gamma)
        if math.degrees(nu_EX) <= theta_iso_deg:
            raise ValueError(
                f"theta_iso_deg={theta_iso_deg:.4f}° 须 < ν(M_EX={M_EX:.4f})="
                f"{math.degrees(nu_EX):.4f}°（Prandtl-Meyer 角上限）。"
            )
        nu_ISO = nu_EX - math.radians(theta_iso_deg)
        M_ISO  = M_from_pm_angle(nu_ISO, gamma)
        stISO  = FlowState(M=M_ISO, p_t=pt_EX, T_t=1.0, label="ISO")
        M_before_NS = M_ISO

    # ------------------------------------------------------------------
    # 4. 终端正激波
    # ------------------------------------------------------------------
    pt_NS = pt_EX * shock_pt_ratio(M_before_NS, gamma)
    M_NS  = M2_after_normal_shock(M_before_NS, gamma)

    # ------------------------------------------------------------------
    # 5. 组装流场站位
    # ------------------------------------------------------------------
    st0  = FlowState(M=M0,          p_t=1.0,   T_t=1.0, label="0")
    stEX = FlowState(M=M_EX,        p_t=pt_EX, T_t=1.0, label="EX")
    stNS = FlowState(M=M_NS,        p_t=pt_NS, T_t=1.0, label="NS")
    st1  = FlowState(M=M_NS,        p_t=pt_NS, T_t=1.0, label="1")
    st2  = FlowState(M=M_NS,        p_t=pt_NS, T_t=1.0, label="2")

    stations = InletFlowStations(
        st0=st0, stEX=stEX, stISO=stISO, stNS=stNS, st1=st1, st2=st2
    )
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


# ---------------------------------------------------------------------------
# Mode 2：用户自定义楔角，正向计算
# ---------------------------------------------------------------------------

def design_external_2d_mode2(
    M0: float,
    wedge_angles_deg: List[float],
    gamma: float = 1.4,
    h_km: Optional[float] = None,
    m_dot: Optional[float] = None,
    theta_iso_deg: float = 0.0,
) -> InletFlowStations:
    """Mode 2：用户指定楔角列表，正向计算波系和总压恢复。

    对每级楔角 θᵢ，求解弱斜激波角 βᵢ（``beta_from_theta_M``），推进流场马赫数
    并累乘总压比，最后施加终端正激波。

    Parameters
    ----------
    M0 : float
        来流马赫数，须 > 1.0。
    wedge_angles_deg : list of float
        各级楔角（度）列表，长度 ≥ 1。第 i 个元素为第 i 级斜面半角。
    gamma : float
        比热比，默认 1.4。
    h_km : float, optional
        飞行高度（km）。若同时提供 ``m_dot``，则附加真实物理量。
    m_dot : float, optional
        质量流量（kg/s）。仅在同时提供 ``h_km`` 时生效。

    Returns
    -------
    InletFlowStations
        填充站位：st0、stEX、stNS、st1、st2。

        动态属性：

        - ``wedge_angles`` — 输入楔角列表的副本（度）
        - ``extra`` — dict，包含以下键：

          - ``sigma``                  总压恢复系数
          - ``M_EX``                   终端正激波前马赫数
          - ``beta_list``              各级激波角（度）
          - ``M_after_list``           各级斜激波后马赫数
          - ``theta_list``             原样返回楔角列表（度）
          - ``sigma_oblique_stages``   各级斜激波总压比

    Raises
    ------
    ValueError
        - M0 ≤ 1.0
        - ``wedge_angles_deg`` 为空
        - 某级楔角 θᵢ ≥ 当前马赫数 Mᵢ 时的最大偏转角（提示最大允许值）
        - 某级激波后马赫数降至 ≤ 1.0

    Notes
    -----
    Mode 2 **不保证** Oswatitsch 最优。设计意图是评估给定楔角约束下的性能折损。

    **自洽性验证基准（CLAUDE.md §5.1 Mode 2）**：

    - 输入 ``wedge_angles_deg = oswatitsch_angles(M0=2.0, N=3, M_EX=1.40)`` 时，
      Mode 2 输出 σ 与 Mode 1 偏差 < 0.001。
    - N=3 等强度条件：各级 ``sigma_oblique_stages`` 相等（偏差 < 0.001）。
    """
    if M0 <= 1.0:
        raise ValueError(f"来流马赫数 M0 须 > 1.0，当前 M0={M0}。")
    if not wedge_angles_deg:
        raise ValueError("楔角列表 wedge_angles_deg 不能为空。")

    M_curr = float(M0)
    pt_EX = 1.0
    beta_list:             List[float] = []
    M_after_list:          List[float] = []
    sigma_oblique_stages:  List[float] = []

    for stage_idx, theta_deg in enumerate(wedge_angles_deg, start=1):
        # ---- 最大偏转角检查（比 beta_from_theta_M 内部错误更友好）----
        theta_max = max_turning_angle(M_curr, gamma)
        if theta_deg >= theta_max:
            raise ValueError(
                f"第 {stage_idx} 级楔角 {theta_deg:.4f}° 超过当前马赫数 "
                f"M={M_curr:.4f} 时的最大偏转角 {theta_max:.4f}°。"
                f"请将第 {stage_idx} 级楔角减小至 < {theta_max:.2f}°。"
            )

        # ---- 激波角（弱激波，返回度） ----
        beta_deg = beta_from_theta_M(theta_deg, M_curr, gamma)
        beta_rad = math.radians(beta_deg)
        theta_rad = math.radians(theta_deg)

        # ---- 法向马赫数 → 总压比 ----
        M_n = M_curr * math.sin(beta_rad)
        sigma_i = shock_pt_ratio(M_n, gamma)
        pt_EX *= sigma_i

        # ---- 激波后马赫数 ----
        M_curr = M2_after_oblique_shock(M_curr, beta_rad, theta_rad, gamma)
        if M_curr <= 1.0:
            raise ValueError(
                f"第 {stage_idx} 级斜激波后马赫数 {M_curr:.4f} ≤ 1.0，"
                "流场进入亚声速，楔角序列不合法（请减小楔角或减少级数）。"
            )

        beta_list.append(beta_deg)
        M_after_list.append(float(M_curr))
        sigma_oblique_stages.append(float(sigma_i))

    # ---- 等熵压缩段（Prandtl-Meyer，可选）----
    M_EX = float(M_curr)
    stISO: Optional[FlowState] = None
    M_before_NS = M_EX   # 终端正激波前马赫数

    if theta_iso_deg > 0.0:
        nu_EX = prandtl_meyer_angle(M_EX, gamma)
        if math.degrees(nu_EX) <= theta_iso_deg:
            raise ValueError(
                f"theta_iso_deg={theta_iso_deg:.4f}° 须 < ν(M_EX={M_EX:.4f})="
                f"{math.degrees(nu_EX):.4f}°（Prandtl-Meyer 角上限）。"
            )
        nu_ISO = nu_EX - math.radians(theta_iso_deg)
        M_ISO  = M_from_pm_angle(nu_ISO, gamma)
        stISO  = FlowState(M=M_ISO, p_t=pt_EX, T_t=1.0, label="ISO")
        M_before_NS = M_ISO

    # ---- 终端正激波 ----
    pt_NS = pt_EX * shock_pt_ratio(M_before_NS, gamma)
    M_NS  = M2_after_normal_shock(M_before_NS, gamma)

    # ---- 组装流场站位 ----
    st0  = FlowState(M=M0,    p_t=1.0,   T_t=1.0, label="0")
    stEX = FlowState(M=M_EX,  p_t=pt_EX, T_t=1.0, label="EX")
    stNS = FlowState(M=M_NS,  p_t=pt_NS, T_t=1.0, label="NS")
    st1  = FlowState(M=M_NS,  p_t=pt_NS, T_t=1.0, label="1")
    st2  = FlowState(M=M_NS,  p_t=pt_NS, T_t=1.0, label="2")

    stations = InletFlowStations(
        st0=st0, stEX=stEX, stISO=stISO, stNS=stNS, st1=st1, st2=st2
    )
    stations.wedge_angles = list(wedge_angles_deg)   # type: ignore[attr-defined]

    # ---- 附加详细波系信息 ----
    stations.extra = {                               # type: ignore[attr-defined]
        "sigma":                 stations.total_pressure_recovery(),
        "M_EX":                  M_EX,
        "beta_list":             beta_list,
        "M_after_list":          M_after_list,
        "theta_list":            list(wedge_angles_deg),
        "sigma_oblique_stages":  sigma_oblique_stages,
    }

    # ---- 可选：附加真实物理量 ----
    if h_km is not None and m_dot is not None:
        atm = ISAAtmosphere(h_km * 1000.0, gamma)
        stations.attach_physical_conditions(atm, M0, m_dot)

    return stations
