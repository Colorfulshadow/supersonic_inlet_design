"""
inlets/pitot/aero_design.py
============================
皮托管（Pitot）进气道气动设计。

物理模型
--------
来流经单道正激波直接减速至亚声速，无外部预压缩。
站位链：st0 → stEX（与 st0 相同）→ stNS（正激波后）→ st1 → st2。

单位约定
--------
- 压力以 st0 总压归一化（p_t0 = 1.0）
- 总温以 st0 总温归一化（T_t0 = 1.0）
- 扩压段损失暂设 sigma_diff = 1.0
"""

from __future__ import annotations

from core.compressible_flow import (
    M2_after_normal_shock,
    shock_pt_ratio,
    isentropic_p_ratio,
    isentropic_T_ratio,
)
from core.flow_stations import FlowState, InletFlowStations


def design_pitot(M0: float, gamma: float = 1.4) -> InletFlowStations:
    """皮托管进气道气动设计。

    来流经单道正激波减速，激波后直接进入亚声速扩压段。

    Parameters
    ----------
    M0 : float
        来流马赫数（站位 0），必须 > 1.0。
    gamma : float
        比热比，默认 1.4。

    Returns
    -------
    InletFlowStations
        填充以下站位：

        - **st0**  — 来流，M=M0，p_t=1.0（归一化基准），T_t=1.0
        - **stEX** — 正激波前（与 st0 相同，皮托管无外部压缩）
        - **stNS** — 正激波后
        - **st1**  — 与 stNS 相同（无唇口损失假设）
        - **st2**  — 与 st1 相同（扩压段损失 sigma_diff=1.0）

    Raises
    ------
    ValueError
        若 M0 ≤ 1.0。

    Notes
    -----
    验证基准（M0=2.0，γ=1.4）：

    - total_pressure_recovery() = 0.7209 ± 0.0005
    - stNS.M = 0.5774 ± 0.001
    """
    if M0 <= 1.0:
        raise ValueError(
            f"皮托管进气道要求来流马赫数 M0 > 1.0，当前 M0={M0}。"
        )

    # ------------------------------------------------------------------
    # 站位 0：自由来流（归一化基准）
    # ------------------------------------------------------------------
    st0 = FlowState(M=M0, p_t=1.0, T_t=1.0, label="0")

    # ------------------------------------------------------------------
    # 站位 EX：正激波前（皮托管无外部压缩，与 st0 相同）
    # ------------------------------------------------------------------
    stEX = FlowState(M=M0, p_t=1.0, T_t=1.0, label="EX")

    # ------------------------------------------------------------------
    # 站位 NS：正激波后
    # 正激波：M_n = M0（法向马赫数等于来流马赫数）
    # ------------------------------------------------------------------
    pt_NS = shock_pt_ratio(M0, gamma)          # p_tNS / p_tEX
    M_NS = M2_after_normal_shock(M0, gamma)    # 激波后马赫数

    # 正激波绝热，总温不变
    T_t_NS = 1.0

    stNS = FlowState(M=M_NS, p_t=pt_NS, T_t=T_t_NS, label="NS")

    # ------------------------------------------------------------------
    # 站位 1：唇口入口（无唇口损失，与 stNS 相同）
    # ------------------------------------------------------------------
    st1 = FlowState(M=M_NS, p_t=pt_NS, T_t=T_t_NS, label="1")

    # ------------------------------------------------------------------
    # 站位 2：发动机面（扩压段损失 sigma_diff=1.0，暂与 st1 相同）
    # ------------------------------------------------------------------
    st2 = FlowState(M=M_NS, p_t=pt_NS, T_t=T_t_NS, label="2")

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
    return stations
