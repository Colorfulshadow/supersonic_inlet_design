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
import warnings
from typing import Optional

from core.flow_stations import InletFlowStations


def pitot_geometry(
    stations: InletFlowStations,
    D2: float,
    L_diffuser: Optional[float] = None,
    N_throat: float = 2.0,
    area_ratio_diff: float = 1.192,
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
        若为 ``None``，按 ``N_throat * D2`` 计算（文献参考：Slater 2023 Table 3，
        M₀=2.0 Axi-Pitot，L_subd/D₂ ≈ 1.0；本项目默认取 2.0，确保几何展示清晰）。
    N_throat : float
        扩压段长度系数，``L_diffuser = N_throat * D2``，默认 2.0。
        仅在 ``L_diffuser`` 为 ``None`` 时生效。
    area_ratio_diff : float
        亚声速扩压段面积比，``A_exit / A_capture``，默认 1.192。
        依据：Slater 2023 Table 3，Axi-Pitot M₀=2.0，A2*/ASD = 1.192，
        对应等效锥角 θeqSD ≈ 2.4°（L_subd/D2=1.0 时）。
    gamma : float
        比热比，默认 1.4（当前版本未使用，保留供后续扩展）。

    Returns
    -------
    dict
        包含以下键（坐标单位 m，原点为进口唇口圆心）：

        ``'r_capture'``
            来流捕获半径（= r_cowl = 扩压段入口半径）。
        ``'r_throat'``
            喉道半径（皮托管无内收缩，= r_capture）。
        ``'r_exit'``
            出口半径（= D2 / 2）。
        ``'x_shock'``
            正激波轴向位置（唇口平面，x = 0）。
        ``'L_diffuser'``
            亚声速扩压段轴向长度（m）。
        ``'x_diffuser_end'``
            扩压段末端轴向位置（x > 0）。
        ``'area_ratio_diff'``
            实际使用的扩压段面积比 A_exit / A_capture。
        ``'theta_eq_diffuser'``
            等效扩压锥半角（度），由 arctan((r_exit-r_capture)/L_diffuser) 计算。
        ``'profile'``
            型线离散点序列 ``list of (x, r)``，从来流捕获面到出口，
            4 个关键点，x 坐标单调递增。

    Notes
    -----
    **捕获半径由扩压段面积比倒推**（Slater 2023 Table 3）：

    皮托管无外部压缩，正激波在唇口平面，捕获面 = 喉道面 = 扩压段入口面：

    .. code-block::

        A_capture = A_exit / area_ratio_diff
        r_capture = r_exit / sqrt(area_ratio_diff)

    默认 area_ratio_diff = 1.192（文献值），对应 M₀=2.0 Axi-Pitot 设计点。
    等效锥角验证（L_subd/D2 = 1.0 时）：

    .. code-block::

        θeqSD = arctan((r_exit - r_capture) / L_subd) ≈ 2.4°  ✓

    本项目默认 N_throat=2.0（L_subd = 2×D2），等效锥角约 1.2°，
    物理上为更温和的扩压，仍在合理范围（1°~3°）内。
    """
    if D2 <= 0:
        raise ValueError(f"出口直径 D2 必须 > 0，当前 D2={D2}。")
    if area_ratio_diff <= 0:
        raise ValueError(f"area_ratio_diff 必须 > 0，当前值={area_ratio_diff}。")
    if stations.st0 is None or stations.st2 is None:
        raise ValueError("stations 缺少必要站位（st0 / st2）。")

    # ------------------------------------------------------------------
    # 出口半径（由 D2 给定）
    # ------------------------------------------------------------------
    r_exit = D2 / 2.0

    # ------------------------------------------------------------------
    # 捕获半径：由扩压段面积比倒推
    # 依据：Slater 2023 Table 3，Axi-Pitot M0=2.0，A2*/ASD = 1.192
    #   A_capture = A_exit / area_ratio_diff
    #   r_capture = r_exit / sqrt(area_ratio_diff)
    # ------------------------------------------------------------------
    r_capture = r_exit / math.sqrt(area_ratio_diff)

    if r_exit < r_capture:
        # area_ratio_diff < 1 时触发，物理上不合理（扩压段收缩）
        warnings.warn(
            f"area_ratio_diff={area_ratio_diff:.4f} < 1：扩压段为收缩管道 "
            f"(r_capture={r_capture:.4f} m > r_exit={r_exit:.4f} m)。"
            f"物理上需 area_ratio_diff ≥ 1。",
            UserWarning,
            stacklevel=2,
        )

    # 皮托管无内收缩：喉道 = 捕获面
    r_throat = r_capture

    # ------------------------------------------------------------------
    # 轴向坐标
    # ------------------------------------------------------------------
    x_shock = 0.0  # 正激波在唇口平面

    if L_diffuser is None:
        # 依据 Slater 2023 Table 3（M0=2.0 Axi-Pitot），L_subd/D2 ≈ 1.0。
        # 默认取 N_throat=2.0（稍大于文献值，确保几何展示清晰）。
        # 直接以 D2 为基准，避免 r_exit ≈ r_throat 时退化为零。
        L_diffuser = N_throat * D2

    x_diffuser_end = x_shock + L_diffuser

    # ------------------------------------------------------------------
    # 等效扩压锥半角（验证用，推荐范围 1°~5°）
    # 正值：扩张管道；负值：收缩管道（r_exit < r_capture）
    # ------------------------------------------------------------------
    theta_eq_diffuser = math.degrees(math.atan2(r_exit - r_capture, L_diffuser))

    # ------------------------------------------------------------------
    # 型线关键点（至少 4 个，x 严格单调递增）
    # 皮托管唇口即喉道（r_throat = r_capture），两者合并为一点。
    # 补充扩压段中点，确保共 4 个离散点。
    # ------------------------------------------------------------------
    x_upstream = -r_capture * 0.5           # 上游参考面（激波前，x < 0）
    x_mid_diff = x_shock + L_diffuser * 0.5  # 扩压段中点
    r_mid_diff = r_throat + (r_exit - r_throat) * 0.5  # 线性插值半径

    profile = [
        (x_upstream,     r_capture),   # 0：来流捕获面（正激波上游参考平面）
        (x_shock,        r_capture),   # 1：唇口 / 正激波 / 喉道（三者重合）
        (x_mid_diff,     r_mid_diff),  # 2：亚声速扩压段中点
        (x_diffuser_end, r_exit),      # 3：扩压段出口（发动机面）
    ]

    return {
        "r_capture":           r_capture,
        "r_throat":            r_throat,
        "r_exit":              r_exit,
        "x_shock":             x_shock,
        "L_diffuser":          L_diffuser,
        "x_diffuser_end":      x_diffuser_end,
        "area_ratio_diff":     area_ratio_diff,
        "theta_eq_diffuser":   theta_eq_diffuser,
        "profile":             profile,
        "normal_shock_points": (
            (0.0, -r_capture),  # 正激波下端（轴对称，取负半径）
            (0.0,  r_capture),  # 正激波上端
        ),
    }


# ---------------------------------------------------------------------------
# 关键截面提取（用于三维建模验证）
# ---------------------------------------------------------------------------

def extract_key_sections(geo: dict) -> list[dict]:
    """从 pitot_geometry 返回值中提取关键截面坐标，用于三维建模和数据验证。

    皮托管为轴对称构型，截面为圆或圆环，用 (x, r_inner, r_outer) 描述。

    Parameters
    ----------
    geo : dict
        ``pitot_geometry()`` 的返回值。

    Returns
    -------
    list of dict
        每个截面包含：
        ``name``      截面名称（中文）
        ``x``         截面轴向坐标 (m)
        ``r_inner``   内径 (m)，皮托管全为 0
        ``r_outer``   外径 (m)
        ``note``      备注
    """
    sections = [
        {
            "name":    "来流捕获面",
            "x":       -geo["r_capture"] * 0.5,    # 激波上游参考面
            "r_inner": 0.0,
            "r_outer": geo["r_capture"],
            "note":    "正激波上游来流管道",
        },
        {
            "name":    "唇口 / 喉道（正激波面）",
            "x":       geo["x_shock"],              # = 0
            "r_inner": 0.0,
            "r_outer": geo["r_throat"],
            "note":    "正激波位置，r_throat = r_capture",
        },
        {
            "name":    "亚声速扩压中点",
            "x":       geo["x_diffuser_end"] * 0.5,
            "r_inner": 0.0,
            "r_outer": (geo["r_throat"] + geo["r_exit"]) * 0.5,
            "note":    "线性扩压段中间截面",
        },
        {
            "name":    "出口（发动机面）",
            "x":       geo["x_diffuser_end"],
            "r_inner": 0.0,
            "r_outer": geo["r_exit"],
            "note":    "D2 = 2 * r_exit",
        },
    ]
    return sections


# ---------------------------------------------------------------------------
# 快速可视化验证（开发用，不进测试）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
    import matplotlib.pyplot as plt

    # 使用 design_pitot 构造标准站位（M0=2.0，与 aero_design 保持一致）
    from inlets.pitot.aero_design import design_pitot
    st = design_pitot(M0=2.0)
    geo = pitot_geometry(st, D2=1.0)   # N_throat=2.0, area_ratio_diff=1.192（默认）

    import math as _math
    A_cap  = _math.pi * geo['r_capture'] ** 2
    A_exit = _math.pi * geo['r_exit'] ** 2

    print(f"r_capture           = {geo['r_capture']:.6f} m")
    print(f"r_throat            = {geo['r_throat']:.6f} m")
    print(f"r_exit              = {geo['r_exit']:.6f} m")
    print(f"x_shock             = {geo['x_shock']:.6f} m")
    print(f"L_diffuser          = {geo['L_diffuser']:.6f} m  (= N_throat * D2 = 2.0 * 1.0)")
    print(f"x_diffuser_end      = {geo['x_diffuser_end']:.6f} m")
    print(f"area_ratio_diff     = {geo['area_ratio_diff']:.4f}   (A_exit/A_capture, Slater 2023 = 1.192)")
    print(f"theta_eq_diffuser   = {geo['theta_eq_diffuser']:.4f} deg  (arctan((r_exit-r_capture)/L); ~2.4 deg @ N_throat=1)")
    print(f"A_capture           = {A_cap:.6f} m^2")
    print(f"A_exit              = {A_exit:.6f} m^2")
    print(f"A_exit/A_capture    = {A_exit/A_cap:.6f}  (should ~= area_ratio_diff)")
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
