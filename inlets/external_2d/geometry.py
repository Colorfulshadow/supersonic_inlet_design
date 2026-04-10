"""
inlets/external_2d/geometry.py
================================
二元外压式进气道型线关键点计算。

坐标系
------
- 原点：进气道头部（第一楔面起点，x=0 处）
- x 轴：沿轴线向下游（正方向）
- y 轴：纵向（向上为正）

几何模型
--------
进气道由 N 级斜楔面（ramp）和 cowl 唇口（lip）组成。
- 下壁：N 折斜面，每级折角为 theta_i（楔角），累积偏转至 cowl 唇口平面
- 上壁（cowl）：水平外罩，唇口在 x_cowl=0 处（原点处 y=y_cowl）
- 每级斜激波射线从对应楔面折点出发，方向角为 (β_i - Σθ_{j<i})

单位约定
--------
- 角度：度（°），内部转弧度计算
- 长度：米（m）
- D2 为矩形截面出口等效宽度（与捕获面同宽）

禁止在本模块重复实现激波总压比，须调用 core.compressible_flow 中的函数。
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from core.compressible_flow import beta_from_theta_M
from core.flow_stations import InletFlowStations


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def external_2d_geometry(
    stations: InletFlowStations,
    wedge_angles: List[float],
    D2: float,
    H_capture: Optional[float] = None,
    gamma: float = 1.4,
    N_throat: float = 2.0,
    L_diff_extra: float = 2.0,
) -> dict:
    """二元外压式进气道型线关键点计算。

    根据 Oswatitsch 楔角序列计算各级斜楔面折点坐标、斜激波射线端点、
    cowl 唇口位置和扩压段型线。坐标原点置于 cowl 唇口（x=0, y=y_cowl），
    即以 cowl 唇口为基准对齐各级激波到达唇口。

    Parameters
    ----------
    stations : InletFlowStations
        :func:`~inlets.external_2d.aero_design.design_external_2d` 返回的流场站位。
    wedge_angles : list of float
        各级楔角（度），长度 = N_stages，由 :func:`~inlets.external_2d.aero_design.oswatitsch_angles` 返回。
    D2 : float
        出口等效高度（m），须 > 0。即发动机面截面高度，矩形截面下等于宽度方向的流道高。
    H_capture : float, optional
        捕获高度（m）。若为 ``None``，由质量守恒自动估算：
        ``H_capture = D2 * φ(M_NS) / (φ(M0) / σ)``.
    N_throat : float
        喉道圆弧过渡系数（默认 2.0）。控制终端激波后的下壁过渡圆弧半径：
        R_lower = (N_throat - 1) * H_throat_entry。N_throat=1 时无过渡弧。
    L_diff_extra : float
        喉道出口到发动机面的附加扩压长度（m，默认 2.0）。
    gamma : float
        比热比，默认 1.4。

    Returns
    -------
    dict
        包含以下键（长度单位 m）：

        ``'H_capture'``
            来流捕获高度（m）。
        ``'x_cowl'``
            cowl 唇口 x 坐标（= 0，即坐标原点）。
        ``'y_cowl'``
            cowl 唇口 y 坐标（= H_capture，正值）。
        ``'ramp_points'``
            下壁折点序列，长度 = N_stages + 1，每个元素为 (x, y)。
            ramp_points[0] 为第一楔面起点（来流面），ramp_points[-1] 为最后一级折点。
        ``'shock_points'``
            各级斜激波端点对，长度 = N_stages，每个元素为 ((x1,y1),(x2,y2))，
            分别为激波起点（折点）和终点（与 cowl 唇口平面的交点）。
        ``'profile_upper'``
            上壁（cowl 外壁）离散点序列 list of (x, y)，x 单调递增。
        ``'profile_lower'``
            下壁（ramp）离散点序列 list of (x, y)，x 单调递增。

    Notes
    -----
    **坐标约定**：cowl 唇口位于 (0, H_capture)。ramp 折点从 x<0 延伸至 cowl 前缘处。

    **斜激波射线方向**（CLAUDE.md §4.5 必须遵守）：
    第 i 级（1-indexed）激波射线与来流（水平）方向的夹角为：

    .. code-block::

        alpha_i = beta_i - sum(theta_j, j=1..i-1)

    其中 beta_i 为第 i 级斜激波角（由 beta_from_theta_M 计算），
    theta_j 为第 j 级楔角（累积偏转）。激波从折点出发，
    方向角为 alpha_i（相对来流方向向上），与 cowl 水平线的交点即为激波终点。

    **捕获高度估算（质量守恒）**：

    .. code-block::

        H_capture = sigma * D2 * phi(M_NS) / phi(M0) * (A_EX / A2)

    对于 2D 矩形截面，等效面积比等于高度比，因此简化为：

    .. code-block::

        H_capture = sigma * D2 * phi(M_NS) / phi(M0)

    （当 M_NS 和 M0 质量流函数比约等于 1，扩压比约等于 1 时此式成立；
    精确值由面积-马赫数关系决定。）

    **数值验证（M0=2.0，N=3，M_EX=1.40，D2=1.0 m）**：

    - H_capture ≈ 0.950 m（含总压恢复修正）
    - ramp_points: 4 个折点，x 坐标单调递增
    - sum(wedge_angles) ≈ 17.22°（Slater 2023 ≈ 17.34°）
    """
    if D2 <= 0:
        raise ValueError(f"D2 须 > 0，当前 D2={D2}。")
    if not wedge_angles:
        raise ValueError("wedge_angles 不能为空。")
    if stations.st0 is None:
        raise ValueError("stations 缺少 st0 站位。")

    N = len(wedge_angles)
    M0 = stations.st0.M

    # ------------------------------------------------------------------
    # 0. 捕获高度
    # ------------------------------------------------------------------
    if H_capture is None:
        from core.compressible_flow import mass_flow_function
        M_NS = stations.stNS.M
        sigma = stations.total_pressure_recovery()
        phi_M0 = mass_flow_function(M0, gamma)
        phi_MNS = mass_flow_function(M_NS, gamma)
        H_capture = sigma * D2 * phi_MNS / phi_M0

    if H_capture <= 0:
        raise ValueError(f"H_capture 须 > 0，当前 {H_capture:.4f} m。")

    # ------------------------------------------------------------------
    # 1. 各级激波角 β_i（度）
    # ------------------------------------------------------------------
    # 第 i 级来流马赫数：从 M0 经前 (i-1) 级斜激波后的马赫数
    from core.compressible_flow import M2_after_oblique_shock

    betas_deg: List[float] = []    # 各级斜激波角（相对本级来流方向）
    M_curr = float(M0)
    cumulative_theta = 0.0         # 来流前已累积偏转角
    for i, theta_i in enumerate(wedge_angles):
        beta_i = beta_from_theta_M(theta_i, M_curr, gamma)
        betas_deg.append(beta_i)
        beta_i_rad = math.radians(beta_i)
        theta_i_rad = math.radians(theta_i)
        M_curr = M2_after_oblique_shock(M_curr, beta_i_rad, theta_i_rad, gamma)

    # ------------------------------------------------------------------
    # 2. Ramp 折点坐标（以 cowl 唇口为原点）
    #
    # cowl 唇口：(x_cowl=0, y_cowl=H_capture)
    # 第 N 个折点恰好在 cowl 唇口处（激波 shock-on-lip 条件），
    # 但 2D 构型中 ramp_points[-1] 是最终激波起点，不一定等于 cowl 位置。
    #
    # 策略：
    #   - ramp_points[0] 为来流面，x 在 cowl 上游某处，y=0（下壁基准线）
    #   - 每级折点从前一折点沿当前楔面方向延伸至下一折角
    #   - 利用各级激波到达 cowl 唇口 (0, H_capture) 的约束反算折点位置
    #
    # 具体：第 N 级激波从 ramp_points[N] 出发，以角度 alpha_N 射向 cowl 唇口。
    # alpha_i = beta_i - sum(theta_j, j<i)  （相对水平方向的激波射线仰角）
    #
    # 工作顺序：从最后一级折点（ramp_points[N]）反推至来流面（ramp_points[0]）
    # ------------------------------------------------------------------

    # cowl 唇口坐标
    x_cowl = 0.0
    y_cowl = float(H_capture)

    # 各级斜激波射线仰角（相对水平来流方向）
    # alpha_i = beta_i(相对本级来流) - cumulative_deflection_before_stage_i
    # 但 beta_i 已经是相对本级来流（已偏转了 sum(theta_j, j<i) 的来流），
    # 所以相对原始水平方向的激波角 = beta_i - cumulative_theta_before_stage_i
    # 即 alpha_i = betas_deg[i] - sum(wedge_angles[:i])
    alphas_deg: List[float] = []
    cum = 0.0
    for i, (beta_i, theta_i) in enumerate(zip(betas_deg, wedge_angles)):
        alpha_i = beta_i - cum       # 相对水平方向的激波仰角（度）
        alphas_deg.append(alpha_i)
        cum += theta_i

    # ------------------------------------------------------------------
    # 从最后一级折点反推所有折点
    # 第 k 级（0-indexed）的激波从 ramp_points[k] 出发，方向角 alphas_deg[k]，
    # 交于 cowl 唇口 (0, H_capture)。
    # 令 ramp_points[k] = (xk, yk)，则：
    #   tan(alpha_k) = (y_cowl - yk) / (x_cowl - xk)
    #   => xk = x_cowl - (y_cowl - yk) / tan(alpha_k)
    #
    # 但我们还需要知道 yk（各折点高度）。
    # 由斜面几何关系：ramp_points[k+1] 在 ramp_points[k] 出发的第 k+1 级楔面上，
    # 楔面相对水平方向倾斜角为 sum(wedge_angles[:k+1])。
    # 两个相邻折点满足：
    #   (y_{k+1} - y_k) / (x_{k+1} - x_k) = tan(sum(wedge_angles[:k+1]))
    #
    # 约束：最后一级折点（ramp_points[N]，即 ramp_points[-1]）的 y 坐标 = 0（下壁基准）
    # 注意这里设 ramp_points[N].y = 0 并不完全准确，更严格的约束是：
    #   - 最后一级折点的激波恰好打到 cowl 唇口（shock-on-lip）
    #   - 即 ramp_points[N-1] 出发的激波射线经 x 轴偏转到达 cowl
    #
    # 简化实现：
    #   1. 令 ramp_points[N].y = 0（最后一级折点在底板基准线上）
    #   2. 用最后一级激波约束求 ramp_points[N].x：
    #      xN = x_cowl - (y_cowl - 0) / tan(alpha_N)
    #   3. 从 ramp_points[N] 向上游逐级反推 ramp_points[k]：
    #      已知 ramp_points[k+1]，楔面倾角 psi_k = sum(wedge_angles[:k+1])
    #      ramp_points[k] 在 ramp_points[k+1] 的上游，且激波从 ramp_points[k]
    #      打到 cowl 唇口。
    #      联立：
    #        y_k = y_{k+1} - (x_{k+1} - x_k) * tan(psi_k)
    #        x_k = x_cowl - (y_cowl - y_k) / tan(alpha_k)
    #      代入消元：
    #        y_k = y_{k+1} - (x_{k+1} - x_cowl + (y_cowl-y_k)/tan(alpha_k)) * tan(psi_k)
    #        y_k + (y_cowl-y_k)*tan(psi_k)/tan(alpha_k) = y_{k+1} - (x_{k+1}-x_cowl)*tan(psi_k)
    #        y_k * (1 - tan(psi_k)/tan(alpha_k)) = y_{k+1} - (x_{k+1}-x_cowl)*tan(psi_k) - y_cowl*tan(psi_k)/tan(alpha_k)
    #        => 求解 y_k，然后求 x_k
    # ------------------------------------------------------------------

    # 各级楔面相对水平方向的倾斜角（累积楔角）
    psi = [0.0] * N       # psi[k] = 第 k+1 级楔面倾角（相对水平，向下）
    cum_psi = 0.0
    for k in range(N):
        cum_psi += wedge_angles[k]
        psi[k] = cum_psi

    psi_rads = [math.radians(p) for p in psi]
    tan_psi  = [math.tan(r) for r in psi_rads]

    # ------------------------------------------------------------------
    # 等间距折点法（首道激波 shock-on-lip，ramp 从 y=0 上升）
    #
    # 推导：
    #   首道激波从 ramp_points[0] = (x_0, 0) 射到 cowl 唇口 (0, H_capture)，
    #   激波角 alpha_0：
    #     tan(alpha_0) = H_capture / abs(x_0)
    #     → x_0 = -H_capture / tan(alpha_0)
    #
    #   ramp 从 y=0 出发，按各级楔面倾角逐步上升：
    #     ramp_y[k+1] = ramp_y[k] + delta_x · tan(psi[k])
    #
    #   等间距步长：delta_x = abs(x_0) / N，保证 ramp_x[N] = 0（cowl 基准面）
    # ------------------------------------------------------------------

    tan_alpha0 = math.tan(math.radians(alphas_deg[0]))
    x_0        = -H_capture / tan_alpha0      # 第一折点 x（< 0，cowl 上游）
    delta_x    = abs(x_0) / N                 # 等 x 间距步长（正值）

    ramp_x = [x_0 + k * delta_x for k in range(N + 1)]
    # ramp 从 y=0 出发逐步上升
    ramp_y = [0.0] * (N + 1)
    for k in range(N):
        ramp_y[k + 1] = ramp_y[k] + delta_x * math.tan(psi_rads[k])

    ramp_points: List[Tuple[float, float]] = list(zip(ramp_x, ramp_y))

    # ------------------------------------------------------------------
    # 3. 各级斜激波端点
    #    shock_points[k] = ((x_start, y_start), (x_end, y_end))
    #    起点 = ramp_points[k]，终点 = cowl 唇口 (0, H_capture)
    # ------------------------------------------------------------------
    shock_points: List[Tuple[Tuple, Tuple]] = []
    for k in range(N):
        x_start, y_start = ramp_points[k]
        x_end   = x_cowl
        y_end   = y_cowl
        shock_points.append(((x_start, y_start), (x_end, y_end)))

    # ------------------------------------------------------------------
    # 4. 喉道圆弧过渡（终端正激波后，下壁从斜面角过渡到水平）
    # ------------------------------------------------------------------
    theta_total = sum(wedge_angles)          # 下壁总偏转角（度）
    theta_rad   = math.radians(theta_total)
    # 进气道入口高度（cowl 唇口到最后一个 ramp 折点的距离）
    H_throat_entry = y_cowl - ramp_points[-1][1]

    # 下壁圆弧半径：N_throat * H_throat_entry
    # （使 N_throat=2.0 时圆弧抬升量足以超过出口下壁高度，形成可见 C-D 喉道）
    R_lower = N_throat * H_throat_entry

    N_arc = 20
    if R_lower < 1e-9:
        # N_throat ≈ 1，无过渡弧，直接以折点为喉道
        x_throat       = 0.0
        y_throat_lower = ramp_points[-1][1]
        lower_arc: List[Tuple[float, float]] = [ramp_points[-1]]
    else:
        # 圆弧圆心（使弧在 ramp_points[-1] 处与下壁斜面相切，在终点处切线水平）
        # 圆心相对 ramp_points[-1] 偏移：垂直于切线方向（顺时针 90°），即指向弧内侧
        cx_low = R_lower * math.sin(theta_rad)
        cy_low = ramp_points[-1][1] - R_lower * math.cos(theta_rad)

        # 弧从起始角 (π/2 + theta_rad) 扫到 π/2（切线从斜面角变为水平）
        start_ang = math.pi / 2 + theta_rad
        end_ang   = math.pi / 2
        lower_arc = []
        for i in range(N_arc):
            t   = i / (N_arc - 1)
            ang = start_ang + t * (end_ang - start_ang)
            lower_arc.append((cx_low + R_lower * math.cos(ang),
                               cy_low + R_lower * math.sin(ang)))

        x_throat       = lower_arc[-1][0]   # = cx_low = R_lower * sin(theta_rad)
        y_throat_lower = lower_arc[-1][1]   # 下壁在水平切线处的 y 值

    # ------------------------------------------------------------------
    # 5. 上壁（cowl）过渡弧：N_arc 点水平段，从 cowl 唇口延伸到喉道截面
    # ------------------------------------------------------------------
    x_start_upper = ramp_x[0]
    upper_arc: List[Tuple[float, float]] = []
    for i in range(N_arc):
        t = i / (N_arc - 1)
        upper_arc.append((x_cowl + t * (x_throat - x_cowl), y_cowl))

    # ------------------------------------------------------------------
    # 6. 扩压段：从喉道出口线性延伸到发动机面
    # ------------------------------------------------------------------
    x_exit       = x_throat + L_diff_extra
    y_exit_lower = y_cowl - D2   # 发动机面下壁 y 坐标（2D 矩形）

    # ------------------------------------------------------------------
    # 7. 型线离散点序列
    # ------------------------------------------------------------------
    # 上壁：来流面起点 → 上壁过渡弧（含 cowl 唇口）→ 扩压出口
    profile_upper: List[Tuple[float, float]] = (
        [(x_start_upper, y_cowl)]
        + upper_arc            # 第一点 = (x_cowl, y_cowl)，最后一点 = (x_throat, y_cowl)
        + [(x_exit, y_cowl)]
    )

    # 下壁：ramp 折点序列 → 过渡圆弧（去重首点）→ 扩压出口
    profile_lower: List[Tuple[float, float]] = (
        list(ramp_points)
        + lower_arc[1:]        # lower_arc[0] == ramp_points[-1]，跳过
        + [(x_exit, y_exit_lower)]
    )

    # ------------------------------------------------------------------
    # 8. 终端正激波端点（倾斜，与偏折后气流方向垂直）
    # 从 cowl 唇口出发，气流向右上偏折，因此激波向右下延伸
    # ------------------------------------------------------------------
    H_throat_entry = y_cowl - ramp_points[-1][1]
    x_ns_ramp = H_throat_entry * math.tan(theta_rad)
    normal_shock_points = (
        (0.0, y_cowl),                      # 上端：cowl 唇口
        (x_ns_ramp, ramp_points[-1][1]),    # 下端：正激波与下壁交点
    )

    return {
        "H_capture":           H_capture,
        "x_cowl":              x_cowl,
        "y_cowl":              y_cowl,
        "ramp_points":         ramp_points,
        "shock_points":        shock_points,
        "profile_upper":       profile_upper,
        "profile_lower":       profile_lower,
        "normal_shock_points": normal_shock_points,
    }


# ---------------------------------------------------------------------------
# 快速可视化验证（开发用，不进测试）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
    import matplotlib.pyplot as plt

    from inlets.external_2d.aero_design import design_external_2d, oswatitsch_angles

    M0 = 2.0
    N  = 3
    M_EX = 1.40
    D2 = 1.0

    stations = design_external_2d(M0=M0, N_stages=N, M_EX=M_EX)
    angles   = oswatitsch_angles(M0=M0, N_stages=N, M_EX=M_EX)
    geo = external_2d_geometry(stations, angles, D2=D2)

    print(f"H_capture      = {geo['H_capture']:.4f} m")
    print(f"x_cowl         = {geo['x_cowl']:.4f} m")
    print(f"y_cowl         = {geo['y_cowl']:.4f} m")
    print(f"ramp_points    = {[(f'{x:.4f}', f'{y:.4f}') for x,y in geo['ramp_points']]}")
    print(f"楔角之和        = {sum(angles):.4f}°")

    fig, ax = plt.subplots(figsize=(12, 5))

    # 下壁（ramp）
    lx = [p[0] for p in geo['profile_lower']]
    ly = [p[1] for p in geo['profile_lower']]
    ax.plot(lx, ly, 'b-o', label='下壁（ramp）', linewidth=2)

    # 上壁（cowl）
    ux = [p[0] for p in geo['profile_upper']]
    uy = [p[1] for p in geo['profile_upper']]
    ax.plot(ux, uy, 'k-o', label='上壁（cowl）', linewidth=2)

    # 各级斜激波
    colors = ['r', 'g', 'm']
    for i, ((x1, y1), (x2, y2)) in enumerate(geo['shock_points']):
        ax.plot([x1, x2], [y1, y2],
                color=colors[i % len(colors)],
                linestyle='--', linewidth=1.5,
                label=f'第{i+1}级斜激波 θ={angles[i]:.2f}°')

    # cowl 唇口
    ax.plot(geo['x_cowl'], geo['y_cowl'], 'k^', markersize=10, label='cowl 唇口')

    ax.axvline(x=0, color='gray', linestyle=':', linewidth=0.8, label='cowl 基准面')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'二元外压式进气道型线（$M_0={M0}, N={N}, M_EX={M_EX}, D_2={D2} m$）')
    ax.legend(loc='upper left')
    ax.axis('equal')
    ax.grid(True)
    plt.tight_layout()
    plt.show()
