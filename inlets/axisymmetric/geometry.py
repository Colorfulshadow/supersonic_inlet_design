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

物理定尺（CLAUDE.md §三 原则 8）：
  r_capture 由 ISA 标准大气 + 质量流量正向推算（core/atmosphere.capture_area）。
  设计工况（ṁ=100 kg/s，M₀=2.0，H=20 km）：A_cap≈1.924 m²，r_capture≈0.782 m。
  禁止使用无量纲参考尺度（如 D2_ref）作为主驱动量。

shock-on-lip 几何：
  锥形激波从锥尖出发恰好打在 cowl 唇口：
    L_cone = r_capture / tan(β_c)    （锥尖到唇口的轴向距离）
    r_cb_base = L_cone × tan(δ_c)   （唇口截面中心锥底部半径）

要求 D2/2 >= r_cowl（cowl 向外扩张），否则抛出 ValueError。
"""

from __future__ import annotations

import math

import numpy as np

from core.atmosphere import capture_area as _isa_capture_area
from core.flow_stations import InletFlowStations


def axisymmetric_geometry(
    stations: InletFlowStations,
    D2: float,
    gamma: float = 1.4,
    lip_mode: int = 1,
    r_lip: float = 0.0,
    mdot: float = 100.0,
    H: float = 20000.0,
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
    lip_mode : int
        唇口模式：1 = 尖唇口（sharp），2 = 圆弧唇口（rounded）。默认 1。
    r_lip : float
        圆弧唇口半径（m），仅 lip_mode=2 时生效。须 > 0 且 < r_throat。
    mdot : float
        设计质量流量（kg/s），用于 ISA 正向推算捕获面积，默认 100.0。
    H : float
        飞行高度（m），用于 ISA 正向推算捕获面积，默认 20 000.0。

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
        ``'lip_mode'``
            当前唇口模式（1 或 2）。
        ``'lip_coords'``
            Mode 1 时为 None；Mode 2 时不存在此键。
        ``'lip_outer_x'``, ``'lip_outer_y'``
            Mode 2：外弧（cowl 外表面）离散坐标（在 (x, r) 平面），各 20 点。
            外弧从 (-r_lip, r_cowl) 到 (0, r_cowl-r_lip)，四分之一圆。
        ``'lip_inner_x'``, ``'lip_inner_y'``
            Mode 2：内弧（进气道内侧）离散坐标，各 20 点。
            内弧从 (0, r_cowl-r_lip) 沿扩压方向延伸，与内壁 C1 连续。

    Raises
    ------
    ValueError
        若 D2 ≤ 0，或 stations 缺少必要站位/extra 属性，
        或几何结果不合理（如 r_cb_base < 0）。

    Notes
    -----
    **r_capture（ISA 大气 + 质量流量正向推算）**：

    .. code-block::

        A_cap = ṁ / (ρ∞ · V∞) = ṁ / (ρ∞ · M₀ · a∞)
        r_capture = sqrt(A_cap / π)

    **shock-on-lip 几何（r_cb_base）**：

    锥形激波从锥尖出发恰好打在 cowl 唇口，由此确定锥长和锥底半径：

    .. code-block::

        L_cone    = r_capture / tan(β_c)   （锥尖到唇口的轴向距离）
        r_cb_base = L_cone · tan(δ_c)      （锥面在唇口截面处的半径）

    **数值示例（M₀=2.0，δ_c=22°，β_c≈39.60°，ṁ=100 kg/s，H=20 km，D2=1.7 m）**：

    - A_cap ≈ 1.924 m²，r_capture = r_cowl ≈ 0.782 m（ISA 物理尺寸）
    - L_cone ≈ 0.949 m，r_cb_base ≈ 0.383 m，r_throat ≈ 0.399 m
    - r_exit = 0.850 m > r_cowl（cowl 向外扩张约 9%）
    - x_cone_tip ≈ -0.949 m，x_diffuser_end ≈ 1.303 m
    """
    if D2 <= 0:
        raise ValueError(f"出口直径 D2 必须 > 0，当前 D2={D2}。")
    if stations.st0 is None or stations.st2 is None or stations.stEX is None:
        raise ValueError("stations 缺少必要站位（st0 / stEX / st2）。")

    extra = getattr(stations, "extra", None)
    if extra is None or "delta_c_deg" not in extra or "beta_c_deg" not in extra:
        raise ValueError(
            "stations.extra 缺少 'delta_c_deg' 或 'beta_c_deg'，"
            "请使用 design_axisymmetric() 生成 stations。"
        )

    M0 = stations.st0.M
    delta_c_deg: float = extra["delta_c_deg"]
    beta_c_deg: float = extra["beta_c_deg"]

    # ------------------------------------------------------------------
    # 捕获半径（ISA 大气 + 质量流量正向推算，CLAUDE.md §三 原则 8）
    # A_cap = ṁ / (ρ∞ · V∞)，r_capture = sqrt(A_cap / π)
    # 设计工况：ṁ=100 kg/s，M₀=2.0，H=20 000 m → A_cap≈1.924 m²，r_capture≈0.782 m
    # ------------------------------------------------------------------
    A_cap_physical = _isa_capture_area(mdot=mdot, M0=M0, H=H, gamma=gamma)
    r_capture = math.sqrt(A_cap_physical / math.pi)
    r_cowl = r_capture    # shock-on-lip 条件下 cowl 唇口半径等于捕获半径

    # 实际出口半径（由 D2 决定）
    r_exit = D2 / 2.0

    if r_exit < r_cowl:
        raise ValueError(
            f"D2 太小：r_exit={r_exit:.4f}m < r_cowl={r_cowl:.4f}m，"
            f"cowl 将向内收缩。请将 D2 设为 >= {2 * r_cowl:.3f}m。"
        )

    # ------------------------------------------------------------------
    # shock-on-lip 几何：锥尖到唇口的轴向距离
    # 锥形激波从锥尖出发，恰好打在 cowl 唇口：
    # L_cone = r_capture / tan(β_c)
    # ------------------------------------------------------------------
    L_cone = r_capture / math.tan(math.radians(beta_c_deg))

    # ------------------------------------------------------------------
    # 中心锥底部半径（唇口截面，锥面几何）
    # 锥面从 (x_cone_tip, 0) 延伸至 (0, r_cb_base)，斜率 tan(δ_c)：
    # r_cb_base = L_cone * tan(δ_c)
    # ------------------------------------------------------------------
    r_cb_base = L_cone * math.tan(math.radians(delta_c_deg))
    r_throat = r_cowl - r_cb_base

    if r_cb_base < 0:
        raise ValueError(
            f"几何矛盾：r_cb_base={r_cb_base:.6f} m < 0，请检查锥角参数。"
        )

    # ------------------------------------------------------------------
    # 轴向坐标
    # ------------------------------------------------------------------
    x_cowl = 0.0

    # 锥尖位置：锥形激波 shock-on-lip，锥尖在 cowl 唇口上游 L_cone 处
    x_cone_tip = -L_cone

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

    # ------------------------------------------------------------------
    # 唇口圆弧几何（lip geometry，与 external_2d 同构，坐标 (x, r) 代替 (x, y)）
    # ------------------------------------------------------------------
    _N_LIP = 20
    # 扩压段初始半角（用于内弧切线方向），固定 3°（与 L_diffuser 定义一致）
    _DIFF_HALF_RAD = math.radians(3.0)

    if lip_mode == 1:
        lip_extra: dict = {"lip_coords": None}
    elif lip_mode == 2:
        if r_lip <= 0.0:
            raise ValueError(f"lip_mode=2 时 r_lip 须 > 0，当前 r_lip={r_lip}。")
        if r_lip >= r_throat:
            raise ValueError(
                f"r_lip={r_lip:.5f} m 超过环形喉道径向高度 r_throat={r_throat:.5f} m，"
                "唇口圆弧将与中心锥干涉，请减小 r_lip。"
            )
        # 外弧（cowl 外表面）：从 (-r_lip, r_cowl) 到 (0, r_cowl-r_lip)，四分之一圆
        #   中心 = (-r_lip, r_cowl - r_lip)
        cx_out = -r_lip
        cy_out = r_cowl - r_lip
        _angs_out = np.linspace(np.pi / 2, 0.0, _N_LIP)
        lip_outer_x = cx_out + r_lip * np.cos(_angs_out)
        lip_outer_y = cy_out + r_lip * np.sin(_angs_out)

        # 内弧（进气道内侧）：C1 连续，切线从水平过渡到扩压方向（3°）
        #   起点 = (0, r_cowl - r_lip)，水平切线
        #   中心 = (0, r_cowl - 2*r_lip)
        #   顺时针扫过扩压半角（3°）
        cx_in = 0.0
        cy_in = r_cowl - 2.0 * r_lip
        _start_a = np.pi / 2
        _end_a   = np.pi / 2 - _DIFF_HALF_RAD
        _angs_in = np.linspace(_start_a, _end_a, _N_LIP)
        lip_inner_x = cx_in + r_lip * np.cos(_angs_in)
        lip_inner_y = cy_in + r_lip * np.sin(_angs_in)

        lip_extra = {
            "lip_outer_x": lip_outer_x,
            "lip_outer_y": lip_outer_y,
            "lip_inner_x": lip_inner_x,
            "lip_inner_y": lip_inner_y,
        }
    else:
        raise ValueError(f"lip_mode 须为 1 或 2，当前 lip_mode={lip_mode}。")

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
        "lip_mode":            lip_mode,
        **lip_extra,
    }


# ---------------------------------------------------------------------------
# 关键截面提取（用于三维建模验证）
# ---------------------------------------------------------------------------

def extract_key_sections(geo: dict) -> list[dict]:
    """从 axisymmetric_geometry 返回值中提取关键截面坐标，用于三维建模和数据验证。

    轴对称构型截面为圆或圆环，用 (x, r_inner, r_outer) 描述。

    Parameters
    ----------
    geo : dict
        ``axisymmetric_geometry()`` 的返回值。

    Returns
    -------
    list of dict
        每个截面包含：
        ``name``      截面名称（中文）
        ``x``         截面轴向坐标 (m)
        ``r_inner``   内径 (m)
        ``r_outer``   外径 (m)
        ``note``      备注
    """
    sections = [
        {
            "name":    "来流捕获面（锥尖截面）",
            "x":       geo["x_cone_tip"],
            "r_inner": 0.0,
            "r_outer": geo["r_capture"],
            "note":    "锥尖处全圆截面，等效捕获流管",
        },
        {
            "name":    "cowl 唇口 / 喉道",
            "x":       geo["x_cowl"],               # = 0
            "r_inner": geo["r_cb_base"],
            "r_outer": geo["r_cowl"],
            "note":    f"环形喉道高度 r_throat={geo['r_throat']:.4f} m",
        },
        {
            "name":    "亚声速扩压中点",
            "x":       geo["x_diffuser_end"] * 0.5,
            "r_inner": 0.0,
            "r_outer": (geo["r_cowl"] + geo["r_exit"]) * 0.5,
            "note":    "扩压段轴向中间截面（cowl 内壁线性扩张）",
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
# 快速可视化（开发用，不进测试）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

    from inlets.axisymmetric.aero_design import design_axisymmetric
    st = design_axisymmetric(M0=2.0, delta_c_deg=22.0, M_EX=1.30)

    # 由 ISA 大气 + 质量流量正向推算物理 r_cowl，确定保证 cowl 外扩的最小 D2
    # 设计工况：ṁ=100 kg/s，M₀=2.0，H=20 km → r_cowl ≈ 0.782 m
    _A_cap = _isa_capture_area(mdot=100.0, M0=2.0, H=20000.0)
    r_cowl_ref = math.sqrt(_A_cap / math.pi)   # ≈ 0.782 m
    D2 = round(2 * r_cowl_ref * 1.1, 2)        # cowl 扩张约 10%
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
    ax.set_title(rf'轴对称锥形激波进气道型线($M_0=2.0, \delta_c=22^\circ, M_{{EX}}=1.30, \dot{{m}}=100$ kg/s, $D_2={D2}$ m)')    
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()
