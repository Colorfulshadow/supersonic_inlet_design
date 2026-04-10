"""
inlets/axisymmetric/geometry.py
================================
轴对称锥形激波进气道 2D 母线关键点计算。

坐标系
------
- 原点：cowl 唇口（x=0, r=r_cowl）
- x 轴：沿轴线向下游（正方向）
- r 轴：径向（≥ 0）

几何模型
--------
外部超声速段（x < 0）：
  来流经锥形激波（spike 中心锥，半角 δ_c）及等熵压缩减速至 M_EX，
  再经终端正激波（位于 cowl 唇口截面）进入内部亚声速扩压段。

内部亚声速段（0 ≤ x ≤ x_diffuser_end）：
  cowl 内壁从 (0, r_cowl) 线性扩张至 (x_diffuser_end, r_exit)。
  等效扩张半角 3°：L_diffuser = (r_exit - r_cowl) / tan(3°)。

r_cowl 以 D2_ref=1.0 m 为参考尺度计算（约 0.624 m），与输入 D2 解耦。
要求 D2/2 >= r_cowl（cowl 向外扩张），否则抛出 ValueError。
"""

from __future__ import annotations

import math

from core.compressible_flow import mass_flow_function
from core.flow_stations import InletFlowStations


def axisymmetric_geometry(
    stations: InletFlowStations,
    D2: float,
    gamma: float = 1.4,
) -> dict:
    """轴对称锥形激波进气道 2D 母线关键点。

    Parameters
    ----------
    stations : InletFlowStations
        :func:`~inlets.axisymmetric.aero_design.design_axisymmetric` 返回的流场站位，
        须附带 ``extra`` 属性（含 ``delta_c_deg``、``sigma_cone``）。
    D2 : float
        出口直径（发动机面），单位 m，须 > 0。
    gamma : float
        比热比，默认 1.4。

    Returns
    -------
    dict
        包含以下键（坐标单位 m，原点为 cowl 唇口）：

        ``'r_capture'``
            来流捕获半径（m）。
        ``'r_cowl'``
            cowl 唇口半径（= r_capture，shock-on-lip 条件）。
        ``'r_cb_tip'``
            中心锥尖端半径（= 0，尖锥）。
        ``'r_cb_base'``
            中心锥底部半径（唇口截面处，m）。
        ``'r_throat'``
            环形喉道径向高度 = r_cowl - r_cb_base（m）。
        ``'r_exit'``
            出口半径（= D2 / 2，m）。
        ``'x_cone_tip'``
            锥尖轴向位置（m，< 0，在 cowl 唇口上游）。
        ``'x_cowl'``
            cowl 唇口轴向位置（= 0，坐标原点定义）。
        ``'x_diffuser_end'``
            亚声速扩压段末端轴向位置（m，> 0）。
        ``'profile_cb'``
            中心锥母线离散点 ``list of (x, r)``，从锥尖到底部，
            x 单调递增，r 单调递增（从 0 到 r_cb_base）。
        ``'profile_cowl'``
            cowl 内壁母线离散点 ``list of (x, r)``，从唇口到出口，
            x 单调递增，r 单调递增（r_exit > r_cowl，cowl 向外扩张）。

    Raises
    ------
    ValueError
        若 D2 ≤ 0，或 stations 缺少必要站位/extra 属性，
        或几何结果不合理（如 r_cb_base < 0）。

    Notes
    -----
    **r_capture（质量守恒）**：

    .. code-block::

        A_capture · φ(M₀) = σ · A_exit · φ(M₂)
        ⟹ r_capture = sqrt(σ · A_exit · φ(M₂) / (π · φ(M₀)))

    **r_cb_base（唇口截面质量守恒）**：

    来流从捕获截面至唇口截面，经锥形激波损失 σ_cone：

    .. code-block::

        A_capture · φ(M₀) = σ_cone · A_lip · φ(M_EX)
        A_lip = π·(r_cowl² - r_cb_base²)
        ⟹ r_cb_base = sqrt(r_cowl² - A_lip / π)

    **数值示例（M₀=2.0，δ_c=22°，M_EX=1.30，D2=1.37 m）**：

    - r_cowl ≈ 0.624 m（参考尺度，与 D2 无关），r_cb_base ≈ 0.373 m，r_throat ≈ 0.251 m
    - r_exit = 0.685 m > r_cowl（cowl 向外扩张约 10%）
    - x_cone_tip ≈ -0.923 m，x_diffuser_end ≈ 1.164 m
    """
    if D2 <= 0:
        raise ValueError(f"出口直径 D2 必须 > 0，当前 D2={D2}。")
    if stations.st0 is None or stations.st2 is None or stations.stEX is None:
        raise ValueError("stations 缺少必要站位（st0 / stEX / st2）。")

    extra = getattr(stations, "extra", None)
    if extra is None or "delta_c_deg" not in extra or "sigma_cone" not in extra:
        raise ValueError(
            "stations.extra 缺少 'delta_c_deg' 或 'sigma_cone'，"
            "请使用 design_axisymmetric() 生成 stations。"
        )

    M0 = stations.st0.M
    M_EX = stations.stEX.M
    M2 = stations.st2.M
    sigma_cone: float = extra["sigma_cone"]
    delta_c_deg: float = extra["delta_c_deg"]

    # ------------------------------------------------------------------
    # 捕获半径 / cowl 唇口半径（参考尺度 D2_ref=1.0 m，与输入 D2 解耦）
    # 使 r_cowl 成为由气动设计决定的结构参数，而非随 D2 线性缩放的量。
    # A_capture_ref · φ(M0) = σ_total · A_exit_ref · φ(M2)，A_exit_ref = π·0.5²
    # ------------------------------------------------------------------
    sigma_total = stations.total_pressure_recovery()
    phi_M0 = mass_flow_function(M0, gamma)
    phi_M2 = mass_flow_function(M2, gamma)

    r_exit_ref = 0.5                               # D2_ref = 1.0 m
    A_exit_ref = math.pi * r_exit_ref ** 2
    A_capture_ref = sigma_total * A_exit_ref * phi_M2 / phi_M0
    r_cowl = math.sqrt(A_capture_ref / math.pi)    # ≈ 0.624 m，固定不随 D2 变化
    r_capture = r_cowl

    # 实际出口半径（由 D2 决定）
    r_exit = D2 / 2.0

    if r_exit < r_cowl:
        raise ValueError(
            f"D2 太小：r_exit={r_exit:.4f}m < r_cowl={r_cowl:.4f}m，"
            f"cowl 将向内收缩。请将 D2 设为 >= {2 * r_cowl:.3f}m。"
        )

    # ------------------------------------------------------------------
    # 中心锥底部半径 r_cb_base（唇口截面质量守恒，基于参考尺度）
    # A_capture_ref · φ(M0) = σ_cone · A_lip · φ(M_EX)
    # A_lip = π·(r_cowl² - r_cb_base²)
    # ------------------------------------------------------------------
    phi_MEX = mass_flow_function(M_EX, gamma)
    A_lip = A_capture_ref * phi_M0 / (sigma_cone * phi_MEX)

    A_cb_base = math.pi * r_cowl ** 2 - A_lip
    if A_cb_base < 0:
        raise ValueError(
            f"几何矛盾：唇口环形面积 A_lip={A_lip:.4f} m² 超过 cowl 总面积 "
            f"π·r_cowl²={math.pi * r_cowl**2:.4f} m²，导致 r_cb_base 虚数。"
            f"请检查 stations 参数（M_EX={M_EX}，sigma_cone={sigma_cone:.4f}）。"
        )
    r_cb_base = math.sqrt(A_cb_base / math.pi)
    r_throat = r_cowl - r_cb_base

    # ------------------------------------------------------------------
    # 轴向坐标
    # ------------------------------------------------------------------
    x_cowl = 0.0

    # 锥尖位置：锥面斜率 = tan(delta_c_deg)，锥底在 x=0 处半径为 r_cb_base
    x_cone_tip = -r_cb_base / math.tan(math.radians(delta_c_deg))

    # 扩压段长度（等效扩张半角 3°）
    # r_exit > r_cowl，cowl 向下游外扩
    half_cone_rad = math.radians(3.0)
    L_diffuser = (r_exit - r_cowl) / math.tan(half_cone_rad)
    x_diffuser_end = x_cowl + L_diffuser

    # ------------------------------------------------------------------
    # 母线离散点
    # 中心锥：从锥尖 (x_cone_tip, 0) → 底部 (0, r_cb_base)，4 点插值
    # ------------------------------------------------------------------
    n_cb = 4
    profile_cb = []
    for i in range(n_cb):
        t = i / (n_cb - 1)
        xi = x_cone_tip + t * (x_cowl - x_cone_tip)
        ri = t * r_cb_base     # 线性（锥面即为线性）
        profile_cb.append((xi, ri))

    # cowl 内壁：从唇口 (0, r_cowl) → 出口 (x_diffuser_end, r_exit)，4 点插值
    # r_exit > r_cowl，线性外扩
    n_cw = 4
    profile_cowl = []
    for i in range(n_cw):
        t = i / (n_cw - 1)
        xi = x_cowl + t * (x_diffuser_end - x_cowl)
        ri = r_cowl + t * (r_exit - r_cowl)   # 线性外扩（r_exit > r_cowl）
        profile_cowl.append((xi, ri))

    return {
        "r_capture":           r_capture,
        "r_cowl":              r_cowl,
        "r_cb_tip":            0.0,
        "r_cb_base":           r_cb_base,
        "r_throat":            r_throat,
        "r_exit":              r_exit,
        "x_cone_tip":          x_cone_tip,
        "x_cowl":              x_cowl,
        "x_diffuser_end":      x_diffuser_end,
        "profile_cb":          profile_cb,
        "profile_cowl":        profile_cowl,
        "normal_shock_points": (
            (0.0, r_cb_base),  # 正激波内缘：中心锥底部半径
            (0.0, r_cowl),     # 正激波外缘：cowl 唇口半径
        ),
    }


# ---------------------------------------------------------------------------
# 快速可视化（开发用，不进测试）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

    from inlets.axisymmetric.aero_design import design_axisymmetric
    st = design_axisymmetric(M0=2.0, delta_c_deg=22.0, M_EX=1.30)

    # 计算参考尺度 r_cowl（D2_ref=1.0），确定保证 cowl 外扩的最小 D2
    _sigma = st.total_pressure_recovery()
    _phi_M0 = mass_flow_function(st.st0.M)
    _phi_M2 = mass_flow_function(st.st2.M)
    r_cowl = math.sqrt(_sigma * 0.25 * _phi_M2 / _phi_M0)   # ≈ 0.624 m
    D2 = round(2 * r_cowl * 1.1, 2)                          # cowl 扩张约 10%
    geo = axisymmetric_geometry(st, D2=D2)

    print("=== 轴对称进气道母线关键点 ===")
    for k, v in geo.items():
        if not isinstance(v, list):
            print(f"  {k:20s} = {v:.4f}")
    print(f"  {'profile_cb':20s} = {geo['profile_cb']}")
    print(f"  {'profile_cowl':20s} = {geo['profile_cowl']}")

    fig, ax = plt.subplots(figsize=(12, 5))

    # 中心锥母线
    xs_cb, rs_cb = zip(*geo['profile_cb'])
    ax.plot(xs_cb, rs_cb, 'b-o', linewidth=2, label='中心锥（上半）')
    ax.plot(xs_cb, [-r for r in rs_cb], 'b-o', linewidth=2, label='中心锥（下半）')

    # cowl 内壁
    xs_cw, rs_cw = zip(*geo['profile_cowl'])
    ax.plot(xs_cw, rs_cw, 'g-o', linewidth=2, label='cowl 内壁（上半）')
    ax.plot(xs_cw, [-r for r in rs_cw], 'g-o', linewidth=2, label='cowl 内壁（下半）')

    # 参考线
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, label='轴线')
    ax.axvline(x=0, color='r', linestyle=':', linewidth=1.2, label='cowl 唇口 (x=0)')

    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('r (m)')
    ax.set_title(rf'轴对称锥形激波进气道型线($M_0=2.0, \delta_c=22^\circ, M_{{EX}}=1.30, D_2={D2}$ m)')    
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()
