"""
inlets/axisymmetric/aero_design.py
====================================
轴对称锥形激波进气道气动设计。

物理模型
--------
来流经锥形激波（Taylor-Maccoll 锥形流）减速，再经等熵压缩至 M_EX，
最后经终端正激波减速至亚声速。

站位链：st0 → 锥形激波 → 等熵压缩 → stEX → 终端正激波 → stNS → st1 → st2

单位约定
--------
- 压力以 st0 总压归一化（p_t0 = 1.0）
- 总温以 st0 总温归一化（T_t0 = 1.0）
- 扩压段损失暂设 sigma_diff = 1.0
- 角度公开接口用度（°），内部计算用弧度
"""

from __future__ import annotations

import math
from typing import Optional

from core.atmosphere import ISAAtmosphere
from core.compressible_flow import (
    M2_after_normal_shock,
    shock_pt_ratio,
)
from core.flow_stations import FlowState, InletFlowStations
from core.taylor_maccoll import cone_shock_pt_ratio, solve_taylor_maccoll


def design_axisymmetric(
    M0: float,
    delta_c_deg: float | None = None,
    M_EX: float = 1.30,
    gamma: float = 1.4,
    h_km: Optional[float] = None,
    m_dot: Optional[float] = None,
) -> InletFlowStations:
    """轴对称锥形激波进气道气动设计。

    Parameters
    ----------
    M0 : float
        来流马赫数（站位 0），必须 > 1.0。
    delta_c_deg : float | None
        锥半角（度）。若为 None，自动搜索使总压恢复最大的锥角。
    M_EX : float
        终端正激波前马赫数（站位 EX），默认 1.30。
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

        - **st0**  — 来流，M=M0，p_t=1.0，T_t=1.0
        - **stEX** — 等熵压缩末端（正激波前），M=M_EX
        - **stNS** — 终端正激波后
        - **st1**  — 与 stNS 相同（无唇口损失）
        - **st2**  — 与 st1 相同（扩压段损失 sigma_diff=1.0）

        返回对象附加属性（通过 ``extra`` 字典访问）：

        - ``delta_c_deg``       : 实际使用的锥半角（度）
        - ``beta_c_deg``        : 锥形激波角（度）
        - ``M_cone``            : 锥面马赫数
        - ``sigma_cone``        : 锥形激波总压比
        - ``sigma_isentropic``  : 等熵压缩段总压比（=1.0）

    Raises
    ------
    ValueError
        若 M0 ≤ 1.0。

    Notes
    -----
    验证基准（M0=2.0，δ_c=22.0°，M_EX=1.30，γ=1.4）：

    - total_pressure_recovery() ≥ 0.958（σ ≈ 0.963）
    - stEX.M = 1.30 ± 0.01
    - M_cone ≈ 1.51 > M_EX（等熵段为真正压缩）

    物理约束：M_cone ≥ M_EX，否则抛出 ValueError。
    δ_c=31.37° 在 M0=2.0 下给出 M_cone=1.21 < M_EX=1.30，违反此约束。
    """
    if M0 <= 1.0:
        raise ValueError(
            f"轴对称进气道要求来流马赫数 M0 > 1.0，当前 M0={M0}。"
        )

    # ------------------------------------------------------------------
    # 确定锥半角（若未给定，自动优化）
    # ------------------------------------------------------------------
    if delta_c_deg is None:
        delta_c_deg = _find_optimal_delta_c(M0, M_EX, gamma)

    # ------------------------------------------------------------------
    # 调用 Taylor-Maccoll 求解器求激波角和锥面马赫数
    # ------------------------------------------------------------------
    tm_result = solve_taylor_maccoll(delta_c_deg, M0, gamma)
    beta_c_deg: float = tm_result["beta_c_deg"]
    M_cone: float = tm_result["M_cone"]

    beta_c_rad = math.radians(beta_c_deg)

    # ------------------------------------------------------------------
    # 物理合法性检查：M_cone 必须 ≥ M_EX
    # 等熵段从 M_cone 减速至 M_EX（压缩），若 M_cone < M_EX 则为膨胀，物理错误。
    # ------------------------------------------------------------------
    if M_cone < M_EX:
        raise ValueError(
            f"物理矛盾：锥面马赫数 M_cone={M_cone:.4f} < M_EX={M_EX:.4f}，"
            f"等熵段将变为膨胀而非压缩。请减小 delta_c_deg 或减小 M_EX。"
            f"当前 delta_c_deg={delta_c_deg:.2f}° 在 M0={M0} 下对应 M_cone={M_cone:.4f}，"
            f"建议 M_EX <= {M_cone:.2f} 或 delta_c_deg <= 22°。"
        )

    # ------------------------------------------------------------------
    # 锥形激波总压比
    # ------------------------------------------------------------------
    sigma_cone = cone_shock_pt_ratio(M0, beta_c_rad, gamma)

    # ------------------------------------------------------------------
    # 站位 0：自由来流（归一化基准）
    # ------------------------------------------------------------------
    st0 = FlowState(M=M0, p_t=1.0, T_t=1.0, label="0")

    # ------------------------------------------------------------------
    # 站位 EX：等熵压缩末端（终端正激波上游）
    # 等熵压缩：总压仅受锥形激波损失，M_cone → M_EX 段等熵无损失
    # ------------------------------------------------------------------
    pt_EX = sigma_cone   # p_tEX / p_t0 = sigma_cone（等熵段无损失）
    stEX = FlowState(M=M_EX, p_t=pt_EX, T_t=1.0, label="EX")

    # ------------------------------------------------------------------
    # 站位 NS：终端正激波后
    # ------------------------------------------------------------------
    sigma_NS = shock_pt_ratio(M_EX, gamma)   # p_tNS / p_tEX
    pt_NS = pt_EX * sigma_NS
    M_NS = M2_after_normal_shock(M_EX, gamma)

    stNS = FlowState(M=M_NS, p_t=pt_NS, T_t=1.0, label="NS")

    # ------------------------------------------------------------------
    # 站位 1：唇口入口（无唇口损失，与 stNS 相同）
    # ------------------------------------------------------------------
    st1 = FlowState(M=M_NS, p_t=pt_NS, T_t=1.0, label="1")

    # ------------------------------------------------------------------
    # 站位 2：发动机面（扩压段损失 sigma_diff=1.0，暂与 st1 相同）
    # ------------------------------------------------------------------
    st2 = FlowState(M=M_NS, p_t=pt_NS, T_t=1.0, label="2")

    # ------------------------------------------------------------------
    # 组装并返回
    # ------------------------------------------------------------------
    stations = InletFlowStations(
        st0=st0,
        stEX=stEX,
        stNS=stNS,
        st1=st1,
        st2=st2,
    )

    # 附加气动参数挂载到 extra 字典
    stations.extra = {  # type: ignore[attr-defined]
        "delta_c_deg": delta_c_deg,
        "beta_c_deg": beta_c_deg,
        "M_cone": M_cone,
        "sigma_cone": sigma_cone,
        "sigma_isentropic": 1.0,
    }

    # ------------------------------------------------------------------
    # 可选：附加真实物理量
    # ------------------------------------------------------------------
    if h_km is not None and m_dot is not None:
        atm = ISAAtmosphere(h_km * 1000.0, gamma)
        stations.attach_physical_conditions(atm, M0, m_dot)

    return stations


def _find_optimal_delta_c(
    M0: float,
    M_EX: float,
    gamma: float,
    n_search: int = 40,
) -> float:
    """自动搜索使总压恢复最大的锥半角。

    在锥角可行范围内粗扫描 + 精化，选取 sigma_cone 最大（即激波损失最小）
    同时 M_cone ≥ M_EX 的锥角。

    Parameters
    ----------
    M0 : float
        来流马赫数。
    M_EX : float
        终端正激波前马赫数下限（需 M_cone ≥ M_EX）。
    gamma : float
        比热比。
    n_search : int
        粗扫描点数。

    Returns
    -------
    float
        最优锥半角（度）。
    """
    import numpy as np

    # 搜索范围：锥角从 1° 到约 45°（Mach 角仅是激波存在的下限，
    # 但 M_cone ≥ M_EX 可能要求 delta_c 远小于 Mach 角附近）
    delta_lo = 1.0
    delta_hi = 45.0

    best_delta = None
    best_sigma = -1.0

    candidates = np.linspace(delta_lo, delta_hi, n_search)
    for delta in candidates:
        try:
            result = solve_taylor_maccoll(float(delta), M0, gamma)
            if result["M_cone"] < M_EX:
                continue
            beta_rad = math.radians(result["beta_c_deg"])
            sigma = cone_shock_pt_ratio(M0, beta_rad, gamma)
            # 最小激波损失 → 最大 sigma_cone，即最小锥角（弱激波）且满足 M_cone≥M_EX
            if sigma > best_sigma:
                best_sigma = sigma
                best_delta = float(delta)
        except (ValueError, Exception):
            continue

    if best_delta is None:
        raise ValueError(
            f"无法找到满足 M_cone ≥ {M_EX} 的锥半角（M0={M0}）。"
        )

    return best_delta
