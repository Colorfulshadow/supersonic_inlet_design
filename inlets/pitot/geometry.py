"""
inlets/pitot/geometry.py
========================
皮托管进气道 2D 型线关键点计算。

坐标系
------
- 原点：进口唇口圆心
- x 轴：沿轴线向下游（正方向）
- r 轴：径向（≥ 0）

几何模型
--------
皮托管无外部压缩，正激波位于唇口平面（x=0）。
来流捕获面与唇口面重合，r_capture = r_cowl。
亚声速扩压段从唇口（x=0）线性扩张至发动机面（x=L_diffuser）。
"""

from __future__ import annotations

import math
from typing import Optional

from core.compressible_flow import mass_flow_function
from core.flow_stations import InletFlowStations


def pitot_geometry(
    stations: InletFlowStations,
    D2: float,
    L_diffuser: Optional[float] = None,
    gamma: float = 1.4,
) -> dict:
    """皮托管进气道 2D 型线关键点计算。

    Parameters
    ----------
    stations : InletFlowStations
        :func:`~inlets.pitot.aero_design.design_pitot` 返回的流场站位。
    D2 : float
        出口直径（发动机面），单位 m，须 > 0。
    L_diffuser : float, optional
        亚声速扩压段轴向长度（m）。
        若为 ``None``，按等效半锥角 3° 自动估算：
        ``L_diffuser = (r_exit - r_throat) / tan(3°)``。
    gamma : float
        比热比，默认 1.4。

    Returns
    -------
    dict
        包含以下键（坐标单位 m，原点为进口唇口圆心）：

        ``'r_capture'``
            来流捕获半径（= r_cowl，皮托管 shock-on-lip）。
        ``'r_throat'``
            喉道半径（皮托管无内收缩，= r_capture）。
        ``'r_exit'``
            出口半径（= D2 / 2）。
        ``'x_shock'``
            正激波轴向位置（唇口平面，x = 0）。
        ``'x_diffuser_end'``
            扩压段末端轴向位置（x > 0）。
        ``'profile'``
            型线离散点序列 ``list of (x, r)``，从来流捕获面到出口，
            至少含 4 个关键点，x 坐标单调递增。

    Notes
    -----
    **面积-马赫数关系（质量守恒）**：

    在 shock-on-lip 条件下，来流经正激波后全部进入进气道，
    质量流量守恒（忽略溢流）：

    .. code-block::

        A_capture * φ(M0) = A_exit * φ(M_exit)

    其中 φ(M) = mass_flow_function(M)，M_exit = stNS.M（扩压段无加速/减速损失时）。
    实际中 A_exit = A2 = π*(D2/2)²，从而：

    .. code-block::

        A_capture = A2 * φ(M_NS) / φ(M0)

    **数值验证（M0=2.0，D2=1.0 m）**：

    - r_capture ≈ 0.500 m（与 r_exit 接近，扩压段向外扩张）
    - 由质量守恒（含总压恢复）：A_capture = σ · A2 · φ(M_NS) / φ(M0) ≈ A2
    """
    if D2 <= 0:
        raise ValueError(f"出口直径 D2 必须 > 0，当前 D2={D2}。")
    if stations.st0 is None or stations.st2 is None:
        raise ValueError("stations 缺少必要站位（st0 / st2）。")

    M0 = stations.st0.M
    M2 = stations.st2.M    # 发动机面马赫数（扩压段出口）

    # ------------------------------------------------------------------
    # 基本半径
    # ------------------------------------------------------------------
    r_exit = D2 / 2.0
    A_exit = math.pi * r_exit ** 2

    # 质量守恒（含总压恢复系数 σ）：
    #   A_capture · p_t0 · φ(M0) = A_exit · p_t2 · φ(M2)
    #   ⟹ A_capture = σ · A_exit · φ(M2) / φ(M0)
    # 其中 σ = p_t2/p_t0 = total_pressure_recovery()，M2 = st2.M（出口马赫数）
    sigma = stations.total_pressure_recovery()
    phi_M0 = mass_flow_function(M0, gamma)
    phi_M2 = mass_flow_function(M2, gamma)
    A_capture = sigma * A_exit * phi_M2 / phi_M0
    r_capture = math.sqrt(A_capture / math.pi)

    if r_exit < r_capture:
        raise ValueError(
            f"D2 too small: diffuser would contract "
            f"(r_capture={r_capture:.4f} m > r_exit={r_exit:.4f} m)。请增大 D2。"
        )

    # 皮托管无内收缩：喉道 = 捕获面
    r_throat = r_capture

    # ------------------------------------------------------------------
    # 轴向坐标
    # ------------------------------------------------------------------
    x_shock = 0.0  # 正激波在唇口平面

    if L_diffuser is None:
        # 等效半锥角 3° 自动估算
        half_cone_rad = math.radians(3.0)
        L_diffuser = (r_exit - r_throat) / math.tan(half_cone_rad)
        # 若 r_exit <= r_throat（理论上不应出现，保护性处理）
        if L_diffuser <= 0:
            L_diffuser = abs(r_exit - r_throat) / math.tan(half_cone_rad) + 1e-6

    x_diffuser_end = x_shock + L_diffuser

    # ------------------------------------------------------------------
    # 型线关键点（至少 4 个，x 严格单调递增）
    # 皮托管唇口即喉道（r_throat = r_capture），两者合并为一点。
    # 补充扩压段中点，确保共 4 个离散点。
    # ------------------------------------------------------------------
    x_upstream = -r_capture * 0.5   # 上游参考面（激波前）
    x_mid_diff = x_shock + L_diffuser * 0.5  # 扩压段中点
    r_mid_diff = r_throat + (r_exit - r_throat) * 0.5  # 线性插值半径

    profile = [
        (x_upstream,    r_capture),   # 0：来流捕获面（正激波上游参考平面）
        (x_shock,       r_capture),   # 1：唇口 / 正激波 / 喉道（三者重合）
        (x_mid_diff,    r_mid_diff),  # 2：亚声速扩压段中点
        (x_diffuser_end, r_exit),     # 3：扩压段出口（发动机面）
    ]

    return {
        "r_capture":           r_capture,
        "r_throat":            r_throat,
        "r_exit":              r_exit,
        "x_shock":             x_shock,
        "x_diffuser_end":      x_diffuser_end,
        "profile":             profile,
        "normal_shock_points": (
            (0.0, -r_capture),  # 正激波下端（轴对称，取负半径）
            (0.0,  r_capture),  # 正激波上端
        ),
    }


# ---------------------------------------------------------------------------
# 快速可视化验证（开发用，不进测试）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
    import matplotlib.pyplot as plt

    from core.compressible_flow import M2_after_normal_shock, shock_pt_ratio
    from core.flow_stations import FlowState, InletFlowStations

    # 构造含亚声速扩压的站位（M2=0.40，典型发动机入口马赫数），
    # 使扩压段在图形中可见地向外扩张。
    M0 = 2.0
    M_NS = M2_after_normal_shock(M0)
    pt_NS = shock_pt_ratio(M0)
    st = InletFlowStations(
        st0 =FlowState(M=M0,   p_t=1.0,    T_t=1.0, label="0"),
        stNS=FlowState(M=M_NS, p_t=pt_NS,  T_t=1.0, label="NS"),
        st2 =FlowState(M=0.40, p_t=pt_NS,  T_t=1.0, label="2"),
    )
    geo = pitot_geometry(st, D2=1.0)

    print(f"r_capture      = {geo['r_capture']:.4f} m")
    print(f"r_throat       = {geo['r_throat']:.4f} m")
    print(f"r_exit         = {geo['r_exit']:.4f} m")
    print(f"x_shock        = {geo['x_shock']:.4f} m")
    print(f"x_diffuser_end = {geo['x_diffuser_end']:.4f} m")
    print("profile points:", geo["profile"])

    xs = [p[0] for p in geo["profile"]]
    rs = [p[1] for p in geo["profile"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, rs, "b-o", label="上壁面")
    ax.plot(xs, [-r for r in rs], "b-o", label="下壁面（对称）")
    ax.axvline(x=geo["x_shock"], color="r", linestyle="--", label="正激波 (x=0)")
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.axis("equal")
    ax.set_title("皮托管进气道型线($M_0=2.0, D_2=1.0$ m)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("r (m)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
