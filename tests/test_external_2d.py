"""
tests/test_external_2d.py
=========================
二元外压式进气道气动设计验证（CLAUDE.md §5.1，Slater 2023 Table 2）。

基准说明（M0=2.0，N=3，M_EX=1.40）
--------------------------------------
- stEX.M = 1.40 ± 0.01
- total_pressure_recovery() ≥ 0.930
- Slater 2023 "theta_stg1=17.34°" 指三级楔角之和（总外压偏转），
  对应 sum(oswatitsch_angles()) ≈ 17.22°（等 M_n Oswatitsch，±0.5° 内）。
  各级单独楔角约 5.4°、5.8°、6.0°（不是 17.34°）。
"""

import math

import numpy as np
import pytest

from core.compressible_flow import beta_from_theta_M, M2_after_oblique_shock, max_turning_angle
from core.flow_stations import InletFlowStations
from core.prandtl_meyer import prandtl_meyer_angle
from inlets.external_2d.aero_design import (
    design_external_2d,
    design_external_2d_mode2,
    oswatitsch_angles,
)
from inlets.external_2d.geometry import external_2d_geometry


# ---------------------------------------------------------------------------
# 主验证基准：M0=2.0，N=3，M_EX=1.40
# ---------------------------------------------------------------------------

class TestBenchmarkM2N3MEX14:
    """Slater 2023 Table 2 基准（§5.1）。"""

    def setup_method(self):
        self.stations = design_external_2d(M0=2.0, N_stages=3, M_EX=1.40)
        self.angles = oswatitsch_angles(M0=2.0, N_stages=3, M_EX=1.40)

    def test_returns_inlet_flow_stations(self):
        assert isinstance(self.stations, InletFlowStations)

    def test_stEX_mach(self):
        """stEX.M = 1.40 ± 0.01（§5.1）。"""
        assert abs(self.stations.stEX.M - 1.40) < 0.01, (
            f"stEX.M = {self.stations.stEX.M:.4f}，期望 1.40±0.01"
        )

    def test_total_pressure_recovery(self):
        """σ ≥ 0.930（§5.1）。"""
        sigma = self.stations.total_pressure_recovery()
        assert sigma >= 0.930, (
            f"σ = {sigma:.4f}，期望 ≥ 0.930"
        )

    def test_total_deflection_angle(self):
        """总外压偏转角 ≈ 17.34° ± 0.5°（对应 Slater 2023 theta_stg1）。

        等 M_n 准则下各级角度略有不同（≈5.4°、5.8°、6.0°），
        但三者之和与 Slater 2023 基准 17.34° 吻合（±0.5°）。
        """
        total = sum(self.angles)
        assert abs(total - 17.34) < 0.5, (
            f"总偏转角 sum(angles) = {total:.4f}°，期望 17.34°±0.5°"
        )

    def test_st0_normalized(self):
        assert self.stations.st0.p_t == pytest.approx(1.0)
        assert self.stations.st0.T_t == pytest.approx(1.0)
        assert self.stations.st0.M == pytest.approx(2.0)

    def test_stEX_total_pressure_less_than_st0(self):
        """经 N 级斜激波后总压应有损失。"""
        assert self.stations.stEX.p_t < 1.0

    def test_stNS_subsonic(self):
        """终端正激波后为亚声速。"""
        assert self.stations.stNS.M < 1.0

    def test_stNS_total_pressure_less_than_stEX(self):
        """正激波造成额外总压损失。"""
        assert self.stations.stNS.p_t < self.stations.stEX.p_t

    def test_st1_equals_stNS(self):
        st1 = self.stations.st1
        stNS = self.stations.stNS
        assert st1.M == pytest.approx(stNS.M)
        assert st1.p_t == pytest.approx(stNS.p_t)

    def test_st2_equals_st1(self):
        st2 = self.stations.st2
        st1 = self.stations.st1
        assert st2.M == pytest.approx(st1.M)
        assert st2.p_t == pytest.approx(st1.p_t)

    def test_total_temperature_conserved(self):
        """全程绝热，总温不变。"""
        for st in [self.stations.stEX, self.stations.stNS, self.stations.st1, self.stations.st2]:
            assert st.T_t == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# oswatitsch_angles() 单元测试
# ---------------------------------------------------------------------------

class TestOswatitschAngles:
    def test_length_equals_N_stages(self):
        for N in [1, 2, 3, 4]:
            angles = oswatitsch_angles(M0=2.0, N_stages=N, M_EX=1.40)
            assert len(angles) == N, f"N={N} 时 len(angles)={len(angles)}"

    def test_sum_less_than_max_turning_angle(self):
        """各级楔角之和不超过 M0 时的最大转折角。"""
        angles = oswatitsch_angles(M0=2.0, N_stages=3, M_EX=1.40)
        assert sum(angles) < max_turning_angle(M=2.0)

    def test_all_angles_positive(self):
        angles = oswatitsch_angles(M0=2.0, N_stages=3, M_EX=1.40)
        for i, a in enumerate(angles):
            assert a > 0, f"第 {i+1} 级楔角 {a:.4f}° 不为正"

    def test_N1_single_stage(self):
        """N=1 单级斜激波：角度应为正且使 M 从 M0→M_EX。"""
        angles = oswatitsch_angles(M0=2.0, N_stages=1, M_EX=1.60)
        assert len(angles) == 1
        assert angles[0] > 0

    def test_M15_N2(self):
        """M0=1.5，N=2，M_EX=1.20。"""
        angles = oswatitsch_angles(M0=1.5, N_stages=2, M_EX=1.20)
        assert len(angles) == 2
        assert all(a > 0 for a in angles)


# ---------------------------------------------------------------------------
# N=1 退化验证：单斜激波 σ 低于 N=3
# ---------------------------------------------------------------------------

class TestNStagesComparison:
    def test_N3_better_than_N1(self):
        """N=3 总压恢复高于 N=1（多级压缩优于单级）。"""
        st3 = design_external_2d(M0=2.0, N_stages=3, M_EX=1.40)
        st1 = design_external_2d(M0=2.0, N_stages=1, M_EX=1.40)
        sigma3 = st3.total_pressure_recovery()
        sigma1 = st1.total_pressure_recovery()
        assert sigma3 > sigma1, (
            f"N=3 σ={sigma3:.4f} 应 > N=1 σ={sigma1:.4f}"
        )

    def test_N1_sigma_lower_than_N3(self):
        """N=1 单斜激波进气道 σ < N=3 设计（体现多级优势）。"""
        st1 = design_external_2d(M0=2.0, N_stages=1, M_EX=1.40)
        st3 = design_external_2d(M0=2.0, N_stages=3, M_EX=1.40)
        assert st1.total_pressure_recovery() < st3.total_pressure_recovery()


# ---------------------------------------------------------------------------
# 自动 M_EX 优化（M_EX=None，全 Oswatitsch）
# ---------------------------------------------------------------------------

class TestAutoMEX:
    def test_auto_returns_stations(self):
        st = design_external_2d(M0=2.0, N_stages=3, M_EX=None)
        assert isinstance(st, InletFlowStations)

    def test_auto_stEX_supersonic(self):
        st = design_external_2d(M0=2.0, N_stages=3, M_EX=None)
        assert st.stEX.M > 1.0

    def test_auto_sigma_reasonable(self):
        """全 Oswatitsch 优化应给出高总压恢复。"""
        st = design_external_2d(M0=2.0, N_stages=3, M_EX=None)
        assert st.total_pressure_recovery() > 0.90


# ---------------------------------------------------------------------------
# 无效输入
# ---------------------------------------------------------------------------

class TestInvalidInput:
    def test_M0_equal_1_raises(self):
        with pytest.raises(ValueError):
            design_external_2d(M0=1.0)

    def test_M0_subsonic_raises(self):
        with pytest.raises(ValueError):
            design_external_2d(M0=0.8)

    def test_N_stages_zero_raises(self):
        with pytest.raises(ValueError):
            design_external_2d(M0=2.0, N_stages=0)

    def test_oswatitsch_angles_invalid_MEX(self):
        with pytest.raises(ValueError):
            oswatitsch_angles(M0=2.0, N_stages=3, M_EX=3.0)  # M_EX > M0


# ---------------------------------------------------------------------------
# 二元外压式几何（Step ⑦）
# ---------------------------------------------------------------------------

class TestExternal2DGeometry:
    """external_2d_geometry() 关键几何量验证。"""

    def setup_method(self):
        self.stations = design_external_2d(M0=2.0, N_stages=3, M_EX=1.40)
        self.angles   = oswatitsch_angles(M0=2.0, N_stages=3, M_EX=1.40)
        self.geo = external_2d_geometry(self.stations, self.angles, D2=1.0)

    # --- 基本结构 ---

    def test_returns_dict(self):
        assert isinstance(self.geo, dict)

    def test_required_keys(self):
        for key in ("H_capture", "x_cowl", "y_cowl",
                    "ramp_points", "shock_points",
                    "profile_upper", "profile_lower"):
            assert key in self.geo, f"缺少键 '{key}'"

    # --- cowl 唇口 ---

    def test_x_cowl_is_zero(self):
        """cowl 唇口为坐标原点。"""
        assert self.geo["x_cowl"] == pytest.approx(0.0)

    def test_y_cowl_equals_H_capture(self):
        assert self.geo["y_cowl"] == pytest.approx(self.geo["H_capture"])

    # --- 捕获高度 ---

    def test_H_capture_positive(self):
        assert self.geo["H_capture"] > 0.0

    def test_H_capture_reasonable(self):
        """M0=2.0，D2=1.0 时捕获高度在合理范围（1.3~1.7 m）。
        由质量守恒：H_capture = σ·D2·φ(M_NS)/φ(M0)，
        φ(M_NS=0.74)/φ(M0=2.0)≈1.58，σ≈0.95 ⟹ H_capture≈1.50 m。
        """
        assert 1.3 <= self.geo["H_capture"] <= 1.7

    # --- ramp 折点 ---

    def test_ramp_points_length(self):
        """ramp_points 长度 = N_stages + 1。"""
        N = len(self.angles)
        assert len(self.geo["ramp_points"]) == N + 1

    def test_ramp_points_x_monotone_increasing(self):
        """ramp 折点 x 坐标严格单调递增（来流→喉道方向）。"""
        xs = [p[0] for p in self.geo["ramp_points"]]
        for i in range(1, len(xs)):
            assert xs[i] > xs[i - 1], (
                f"ramp_points x 在第 {i} 点不单调：{xs[i-1]:.4f} → {xs[i]:.4f}"
            )

    def test_ramp_last_point_at_cowl_x(self):
        """最后一个折点 x 坐标 = x_cowl = 0。"""
        last_x = self.geo["ramp_points"][-1][0]
        assert last_x == pytest.approx(0.0, abs=1e-9)

    def test_ramp_first_x_upstream(self):
        """第一个折点（来流面）在 cowl 上游（x < 0）。"""
        assert self.geo["ramp_points"][0][0] < 0.0

    # --- 斜激波端点 ---

    def test_shock_points_length(self):
        """shock_points 数量 = N_stages。"""
        assert len(self.geo["shock_points"]) == len(self.angles)

    def test_shock_points_end_at_cowl(self):
        """所有斜激波终点都在 cowl 唇口 (0, H_capture)。"""
        x_cowl = self.geo["x_cowl"]
        y_cowl = self.geo["y_cowl"]
        for i, (start, end) in enumerate(self.geo["shock_points"]):
            assert end[0] == pytest.approx(x_cowl, abs=1e-9), (
                f"第{i+1}级激波终点 x={end[0]:.4f} ≠ x_cowl={x_cowl:.4f}"
            )
            assert end[1] == pytest.approx(y_cowl, abs=1e-9), (
                f"第{i+1}级激波终点 y={end[1]:.4f} ≠ y_cowl={y_cowl:.4f}"
            )

    def test_shock_starts_at_ramp_points(self):
        """第 k 级激波起点 = ramp_points[k]。"""
        rp = self.geo["ramp_points"]
        for k, (start, _) in enumerate(self.geo["shock_points"]):
            assert start[0] == pytest.approx(rp[k][0], abs=1e-9)
            assert start[1] == pytest.approx(rp[k][1], abs=1e-9)

    # --- 型线 ---

    def test_profile_upper_x_monotone(self):
        xs = [p[0] for p in self.geo["profile_upper"]]
        for i in range(1, len(xs)):
            assert xs[i] > xs[i - 1]

    def test_profile_lower_x_monotone(self):
        xs = [p[0] for p in self.geo["profile_lower"]]
        for i in range(1, len(xs)):
            assert xs[i] > xs[i - 1]

    def test_profile_upper_y_constant_before_cowl(self):
        """cowl 唇口上游的上壁 y 坐标等于 y_cowl（水平外壁）。"""
        y_cowl = self.geo["y_cowl"]
        for x, y in self.geo["profile_upper"]:
            if x <= 0.0:
                assert y == pytest.approx(y_cowl, abs=1e-9)

    def test_ramp_y_monotone_increasing(self):
        """ramp_y 单调递增：ramp 从 y=0 出发逐步上升（正确压缩方向）。"""
        ramp_y = [p[1] for p in self.geo["ramp_points"]]
        for i in range(1, len(ramp_y)):
            assert ramp_y[i] >= ramp_y[i - 1], (
                f"ramp_y 在第 {i} 点不单调递增：{ramp_y[i-1]:.4f} → {ramp_y[i]:.4f}"
            )
        assert ramp_y[0] == pytest.approx(0.0, abs=1e-9), "ramp_y[0] 应从 0 出发"

    def test_throat_smaller_than_capture(self):
        """喉道通道高度 < 捕获高度（通道收窄，压缩正确）。"""
        H_capture = self.geo["H_capture"]
        ramp_y_last = self.geo["ramp_points"][-1][1]
        throat_height = H_capture - ramp_y_last
        assert throat_height < H_capture, (
            f"喉道高度 {throat_height:.4f} m 应 < H_capture {H_capture:.4f} m"
        )

    def test_normal_shock_tilted(self):
        """激波两端 x 坐标不等，且下端（斜面侧）在下端（cowl 侧）下游。"""
        (x1, y1), (x2, y2) = self.geo["normal_shock_points"]
        assert abs(x2 - x1) > 1e-4
        assert x2 > x1

    def test_normal_shock_perpendicular_to_flow(self):
        """激波切向量与偏折气流方向点积趋近于零（误差 < sin 0.5°）。"""
        import math
        Theta_rad = math.radians(sum(self.angles))
        (x1, y1), (x2, y2) = self.geo["normal_shock_points"]
        dx, dy = x2 - x1, y2 - y1
        dot = dx * math.cos(Theta_rad) + dy * math.sin(Theta_rad)
        assert abs(dot / math.hypot(dx, dy)) < math.sin(math.radians(0.5))

    # --- 无效输入 ---

    def test_invalid_D2_raises(self):
        with pytest.raises(ValueError):
            external_2d_geometry(self.stations, self.angles, D2=0.0)

    def test_negative_D2_raises(self):
        with pytest.raises(ValueError):
            external_2d_geometry(self.stations, self.angles, D2=-1.0)

    def test_empty_wedge_angles_raises(self):
        with pytest.raises(ValueError):
            external_2d_geometry(self.stations, [], D2=1.0)


# ===========================================================================
# Mode 2：用户自定义楔角（CLAUDE.md §4.8 / §5.1 Mode 2 验证基准）
# ===========================================================================

class TestMode2BasicBehavior:
    """Mode 2 基本行为验证。"""

    def test_returns_inlet_flow_stations(self):
        st = design_external_2d_mode2(2.0, [8.0, 9.0])
        assert isinstance(st, InletFlowStations)

    def test_wedge_angles_attribute_preserved(self):
        """stations.wedge_angles 应与输入一致。"""
        angles = [8.0, 9.0]
        st = design_external_2d_mode2(2.0, angles)
        assert hasattr(st, "wedge_angles")
        assert st.wedge_angles == angles

    def test_extra_dict_present(self):
        """stations.extra 应包含详细波系信息。"""
        st = design_external_2d_mode2(2.0, [8.0, 9.0])
        assert hasattr(st, "extra")
        for key in ("sigma", "M_EX", "beta_list", "M_after_list",
                    "theta_list", "sigma_oblique_stages"):
            assert key in st.extra, f"extra 缺少键 '{key}'"

    def test_extra_beta_list_length(self):
        angles = [5.0, 6.0, 7.0]   # 三级楔角，在 M=2.0 下均合法
        st = design_external_2d_mode2(2.0, angles)
        assert len(st.extra["beta_list"]) == 3
        assert len(st.extra["sigma_oblique_stages"]) == 3
        assert len(st.extra["M_after_list"]) == 3

    def test_st0_normalized(self):
        st = design_external_2d_mode2(2.0, [8.0, 9.0])
        assert st.st0.p_t == pytest.approx(1.0)
        assert st.st0.T_t == pytest.approx(1.0)
        assert st.st0.M   == pytest.approx(2.0)

    def test_stEX_supersonic(self):
        st = design_external_2d_mode2(2.0, [8.0, 9.0])
        assert st.stEX.M > 1.0

    def test_stNS_subsonic(self):
        st = design_external_2d_mode2(2.0, [8.0, 9.0])
        assert st.stNS.M < 1.0

    def test_pt_chain_decreasing(self):
        """总压沿流向单调递减。"""
        st = design_external_2d_mode2(2.0, [8.0, 9.0])
        assert st.stEX.p_t < st.st0.p_t
        assert st.stNS.p_t < st.stEX.p_t

    def test_st1_equals_stNS(self):
        st = design_external_2d_mode2(2.0, [8.0, 9.0])
        assert st.st1.M   == pytest.approx(st.stNS.M)
        assert st.st1.p_t == pytest.approx(st.stNS.p_t)

    def test_st2_equals_st1(self):
        st = design_external_2d_mode2(2.0, [8.0, 9.0])
        assert st.st2.M   == pytest.approx(st.st1.M)
        assert st.st2.p_t == pytest.approx(st.st1.p_t)

    def test_total_temperature_conserved(self):
        """全程绝热，总温不变。"""
        st = design_external_2d_mode2(2.0, [8.0, 9.0])
        for attr in ("stEX", "stNS", "st1", "st2"):
            assert getattr(st, attr).T_t == pytest.approx(1.0)


class TestMode2Accuracy:
    """Mode 2 数值精度：与手算波系一致（误差 < 0.001）。"""

    @staticmethod
    def _manual_chain(M0, theta_list):
        """逐级手算波系，返回最终马赫数（用于测试对比）。"""
        M = float(M0)
        for theta_deg in theta_list:
            beta_deg = beta_from_theta_M(theta_deg, M)
            beta_rad = math.radians(beta_deg)
            theta_rad = math.radians(theta_deg)
            M = M2_after_oblique_shock(M, beta_rad, theta_rad)
        return M

    def test_stEX_M_two_stage(self):
        """M0=2.0，楔角=[8°, 9°]：stEX.M 与手算一致（误差 < 0.001）。"""
        angles = [8.0, 9.0]
        M_EX_ref = self._manual_chain(2.0, angles)
        st = design_external_2d_mode2(2.0, angles)
        assert abs(st.stEX.M - M_EX_ref) < 0.001, (
            f"stEX.M={st.stEX.M:.6f}，手算参考值={M_EX_ref:.6f}"
        )

    def test_stEX_M_three_stage(self):
        """M0=2.0，楔角=[6°, 7°, 8°]：stEX.M 与手算一致。"""
        angles = [6.0, 7.0, 8.0]
        M_EX_ref = self._manual_chain(2.0, angles)
        st = design_external_2d_mode2(2.0, angles)
        assert abs(st.stEX.M - M_EX_ref) < 0.001

    def test_single_stage(self):
        """N=1 单级：结果等价于单斜激波。"""
        theta = 10.0
        M_EX_ref = self._manual_chain(2.0, [theta])
        st = design_external_2d_mode2(2.0, [theta])
        assert abs(st.stEX.M - M_EX_ref) < 0.001

    def test_sigma_extra_matches_total_pressure_recovery(self):
        """extra['sigma'] 与 total_pressure_recovery() 一致。"""
        st = design_external_2d_mode2(2.0, [8.0, 9.0])
        assert abs(st.extra["sigma"] - st.total_pressure_recovery()) < 1e-12


class TestMode2SelfConsistency:
    """§5.1 Mode 2 自洽性验证基准（CLAUDE.md 要求）。"""

    def test_mode2_oswatitsch_input_matches_mode1_sigma(self):
        """Mode 2 输入 Oswatitsch 最优角 → σ 与 Mode 1 偏差 < 0.001。"""
        M0, N, M_EX_target = 2.0, 3, 1.40
        optimal_angles = oswatitsch_angles(M0, N, M_EX_target)

        st1 = design_external_2d(M0, N_stages=N, M_EX=M_EX_target)
        st2 = design_external_2d_mode2(M0, optimal_angles)

        sigma1 = st1.total_pressure_recovery()
        sigma2 = st2.total_pressure_recovery()
        assert abs(sigma1 - sigma2) < 0.001, (
            f"Mode 1 σ={sigma1:.6f}，Mode 2 σ={sigma2:.6f}，偏差 {abs(sigma1-sigma2):.2e}"
        )

    def test_mode2_uniform_oblique_pt_ratio(self):
        """N=3 等强度楔角 → 各级斜激波总压比相等（偏差 < 0.001）。"""
        optimal_angles = oswatitsch_angles(2.0, 3, 1.40)
        st = design_external_2d_mode2(2.0, optimal_angles)
        sigmas = st.extra["sigma_oblique_stages"]
        for i in range(1, len(sigmas)):
            assert abs(sigmas[i] - sigmas[0]) < 0.001, (
                f"各级斜激波总压比不等：sigma[0]={sigmas[0]:.6f}，"
                f"sigma[{i}]={sigmas[i]:.6f}"
            )

    def test_non_optimal_sigma_less_than_mode1(self):
        """不等强度楔角（非 Oswatitsch）→ 斜激波段总压比 < 等强度最优值。

        Oswatitsch 定理：相同 M_EX 条件下，等强度多级斜激波总压比最大。
        将 Oswatitsch 最优角的分配打散（不等强度，但保持 M_EX 相同），
        斜激波链总压比应 ≤ 等强度最优值（Oswatitsch 定理核心结论）。
        """
        M0, N, M_EX_target = 2.0, 3, 1.40
        optimal_angles = oswatitsch_angles(M0, N, M_EX_target)

        # 等强度（最优）Mode 2
        st_optimal = design_external_2d_mode2(M0, optimal_angles)
        sigma_oblique_opt = 1.0
        for s in st_optimal.extra["sigma_oblique_stages"]:
            sigma_oblique_opt *= s

        # 不等强度：将相同总偏转量集中到 1 级（单级斜激波 + 终端正激波）
        # 单级斜激波等效于最差的分配，总压比应 < 等强度 N 级
        total_theta = sum(optimal_angles)
        st_single = design_external_2d_mode2(M0, [total_theta])
        sigma_oblique_single = st_single.extra["sigma_oblique_stages"][0]

        assert sigma_oblique_single <= sigma_oblique_opt + 1e-6, (
            f"单级斜激波总压比 {sigma_oblique_single:.6f} 应 ≤ 等强度 N 级乘积 "
            f"{sigma_oblique_opt:.6f}（Oswatitsch 定理）"
        )


class TestMode2ErrorHandling:
    """Mode 2 异常处理。"""

    def test_M0_subsonic_raises(self):
        with pytest.raises(ValueError, match="M0"):
            design_external_2d_mode2(0.8, [8.0])

    def test_M0_equal_1_raises(self):
        with pytest.raises(ValueError):
            design_external_2d_mode2(1.0, [8.0])

    def test_empty_angles_raises(self):
        with pytest.raises(ValueError):
            design_external_2d_mode2(2.0, [])

    def test_angle_exceeds_max_deflection_raises(self):
        """楔角 30° 超过 M=2.0 时最大偏转角（≈23°）→ ValueError。"""
        with pytest.raises(ValueError) as exc_info:
            design_external_2d_mode2(2.0, [30.0])
        msg = str(exc_info.value)
        # 错误信息应包含最大允许值提示
        assert "最大" in msg or "超过" in msg, f"错误信息无提示最大值：{msg}"

    def test_angle_exceeds_max_deflection_at_stage2(self):
        """第二级楔角超限（第一级合法，第二级超限）→ ValueError，提示第 2 级。"""
        with pytest.raises(ValueError) as exc_info:
            design_external_2d_mode2(2.0, [5.0, 30.0])
        assert "2" in str(exc_info.value), "应提示第 2 级超限"

    def test_error_message_shows_max_value(self):
        """错误信息应包含最大允许楔角的数值。"""
        try:
            design_external_2d_mode2(2.0, [30.0])
        except ValueError as e:
            msg = str(e)
            # 最大偏转角约 23°，错误信息中应出现类似数字
            assert any(c.isdigit() for c in msg), "错误信息应包含数值"


class TestMode2ViaUnifiedInterface:
    """通过统一接口 design_external_2d(mode=2) 的路由验证。"""

    def test_mode2_delegation(self):
        """design_external_2d(mode=2, wedge_angles=[...]) 等价于直接调用 mode2。"""
        angles = [8.0, 9.0]
        st_direct = design_external_2d_mode2(2.0, angles)
        st_unified = design_external_2d(2.0, mode=2, wedge_angles=angles)

        assert abs(st_unified.stEX.M - st_direct.stEX.M) < 1e-10
        assert abs(
            st_unified.total_pressure_recovery() - st_direct.total_pressure_recovery()
        ) < 1e-10

    def test_mode2_no_wedge_angles_raises(self):
        """mode=2 但未提供 wedge_angles → ValueError。"""
        with pytest.raises(ValueError, match="wedge_angles"):
            design_external_2d(2.0, mode=2)

    def test_mode2_empty_wedge_angles_raises(self):
        with pytest.raises(ValueError):
            design_external_2d(2.0, mode=2, wedge_angles=[])

    def test_mode1_default_unaffected(self):
        """mode=1（默认）行为不受影响。"""
        st = design_external_2d(M0=2.0, N_stages=3, M_EX=1.40)
        assert abs(st.stEX.M - 1.40) < 0.01
        assert st.total_pressure_recovery() >= 0.930


# ---------------------------------------------------------------------------
# 等熵压缩段（Prandtl-Meyer，步骤 ⑪）
# ---------------------------------------------------------------------------

class TestIsoThetaZeroRegression:
    """theta_iso_deg=0（默认）时结果与步骤⑦完全一致（回归测试）。"""

    def test_stISO_is_none_by_default(self):
        """theta_iso_deg=0（默认）→ stISO 为 None。"""
        st = design_external_2d(M0=2.0, N_stages=3, M_EX=1.40)
        assert st.stISO is None

    def test_stISO_none_mode2_by_default(self):
        """Mode 2 默认 theta_iso_deg=0 → stISO 为 None。"""
        st = design_external_2d_mode2(2.0, [8.0, 9.0])
        assert st.stISO is None

    def test_sigma_unchanged_theta_iso_zero(self):
        """theta_iso_deg=0 时 σ 与原 σ 完全一致。"""
        st_ref = design_external_2d(M0=2.0, N_stages=3, M_EX=1.40)
        st_iso = design_external_2d(M0=2.0, N_stages=3, M_EX=1.40, theta_iso_deg=0.0)
        assert st_ref.total_pressure_recovery() == pytest.approx(
            st_iso.total_pressure_recovery()
        )

    def test_stEX_M_unchanged_theta_iso_zero(self):
        """theta_iso_deg=0 时 stEX.M 不变。"""
        st_ref = design_external_2d(M0=2.0, N_stages=3, M_EX=1.40)
        st_iso = design_external_2d(M0=2.0, N_stages=3, M_EX=1.40, theta_iso_deg=0.0)
        assert st_ref.stEX.M == pytest.approx(st_iso.stEX.M)

    def test_geometry_iso_ramp_empty_when_zero(self):
        """theta_iso_deg=0 时 iso_ramp_x/iso_ramp_y 为空数组。"""
        st = design_external_2d(M0=2.0, N_stages=3, M_EX=1.40)
        angles = oswatitsch_angles(2.0, 3, 1.40)
        geo = external_2d_geometry(st, angles, D2=1.0, theta_iso_deg=0.0)
        assert len(geo["iso_ramp_x"]) == 0
        assert len(geo["iso_ramp_y"]) == 0


class TestIsoThetaNonzero:
    """theta_iso_deg > 0 时等熵段物理正确性验证。"""

    def setup_method(self):
        self.M0 = 2.0
        self.N  = 3
        self.M_EX_target = 1.40
        self.theta_iso = 5.0   # 5° 等熵压缩，小于 ν(1.40) ≈ 9°
        self.st = design_external_2d(
            M0=self.M0, N_stages=self.N, M_EX=self.M_EX_target,
            theta_iso_deg=self.theta_iso,
        )

    def test_stISO_exists(self):
        """theta_iso_deg > 0 → stISO 不为 None。"""
        assert self.st.stISO is not None

    def test_stISO_M_less_than_stEX_M(self):
        """stISO.M < stEX.M（等熵压缩使马赫数降低）。"""
        assert self.st.stISO.M < self.st.stEX.M, (
            f"stISO.M={self.st.stISO.M:.4f}，stEX.M={self.st.stEX.M:.4f}"
        )

    def test_stISO_p_t_equals_stEX_p_t(self):
        """等熵过程无总压损失：stISO.p_t = stEX.p_t。"""
        assert self.st.stISO.p_t == pytest.approx(self.st.stEX.p_t, rel=1e-9)

    def test_stISO_M_consistent_with_PM(self):
        """stISO.M 与 PM 公式计算值一致（误差 < 0.005）。"""
        from core.prandtl_meyer import M_from_pm_angle
        nu_EX  = prandtl_meyer_angle(self.M_EX_target)
        nu_ISO = nu_EX - math.radians(self.theta_iso)
        M_ISO_expected = M_from_pm_angle(nu_ISO)
        assert abs(self.st.stISO.M - M_ISO_expected) < 0.005, (
            f"stISO.M={self.st.stISO.M:.4f}，PM 期望={M_ISO_expected:.4f}"
        )

    def test_sigma_improves_with_iso(self):
        """等熵段使总压恢复提高（终端正激波在更低 M_ISO 前，损失更小）。"""
        st_no_iso = design_external_2d(M0=self.M0, N_stages=self.N, M_EX=self.M_EX_target)
        sigma_no_iso = st_no_iso.total_pressure_recovery()
        sigma_with_iso = self.st.total_pressure_recovery()
        assert sigma_with_iso > sigma_no_iso, (
            f"σ(with_iso)={sigma_with_iso:.4f} 应 > σ(no_iso)={sigma_no_iso:.4f}"
        )

    def test_stNS_M_lower_with_iso(self):
        """有等熵段时 stNS.M（正激波后）不变，但来自更低 M_ISO 的正激波。"""
        from core.compressible_flow import M2_after_normal_shock
        M_ISO = self.st.stISO.M
        M_NS_expected = M2_after_normal_shock(M_ISO)
        assert abs(self.st.stNS.M - M_NS_expected) < 1e-6

    def test_mode2_stISO_works(self):
        """Mode 2 同样支持 theta_iso_deg。"""
        angles = [8.0, 9.0]
        st = design_external_2d_mode2(2.0, angles, theta_iso_deg=3.0)
        assert st.stISO is not None
        assert st.stISO.M < st.stEX.M
        assert st.stISO.p_t == pytest.approx(st.stEX.p_t, rel=1e-9)


class TestIsoThetaErrorHandling:
    """等熵段参数越界测试。"""

    def test_theta_iso_exceeds_nu_raises(self):
        """theta_iso_deg ≥ ν(M_EX) → ValueError。

        ν(M_EX=1.40) ≈ 9°，theta_iso=9° 应触发错误。
        """
        nu_deg = math.degrees(prandtl_meyer_angle(1.40))
        with pytest.raises(ValueError):
            design_external_2d(M0=2.0, N_stages=3, M_EX=1.40, theta_iso_deg=nu_deg)

    def test_theta_iso_too_large_raises(self):
        """theta_iso_deg = 20°（远超 ν(M_EX)≈9°）→ ValueError。"""
        with pytest.raises(ValueError):
            design_external_2d(M0=2.0, N_stages=3, M_EX=1.40, theta_iso_deg=20.0)

    def test_theta_iso_negative_ok(self):
        """theta_iso_deg=0 时（默认）无错误。"""
        st = design_external_2d(M0=2.0, N_stages=3, M_EX=1.40, theta_iso_deg=0.0)
        assert st is not None

    def test_mode2_theta_iso_exceeds_nu_raises(self):
        """Mode 2 theta_iso_deg ≥ ν(M_EX) → ValueError。"""
        angles = [8.0, 9.0]
        st_ref = design_external_2d_mode2(2.0, angles)
        nu_deg = math.degrees(prandtl_meyer_angle(st_ref.stEX.M))
        with pytest.raises(ValueError):
            design_external_2d_mode2(2.0, angles, theta_iso_deg=nu_deg + 1.0)


class TestIsoGeometry:
    """等熵段几何坐标测试。"""

    def setup_method(self):
        self.st = design_external_2d(M0=2.0, N_stages=3, M_EX=1.40, theta_iso_deg=5.0)
        self.angles = oswatitsch_angles(2.0, 3, 1.40)
        self.geo = external_2d_geometry(
            self.st, self.angles, D2=1.0, theta_iso_deg=5.0,
        )

    def test_iso_ramp_x_nonempty(self):
        """theta_iso_deg=5° → iso_ramp_x 非空。"""
        assert len(self.geo["iso_ramp_x"]) > 0

    def test_iso_ramp_y_nonempty(self):
        """theta_iso_deg=5° → iso_ramp_y 非空。"""
        assert len(self.geo["iso_ramp_y"]) > 0

    def test_iso_ramp_starts_at_last_ramp_point(self):
        """iso_ramp 起点 = ramp_points[-1]。"""
        x_ramp_last = self.geo["ramp_points"][-1][0]
        y_ramp_last = self.geo["ramp_points"][-1][1]
        assert self.geo["iso_ramp_x"][0] == pytest.approx(x_ramp_last, abs=1e-10)
        assert self.geo["iso_ramp_y"][0] == pytest.approx(y_ramp_last, abs=1e-10)

    def test_iso_ramp_y_monotone_increasing(self):
        """iso_ramp_y 单调递增（ramp 向上偏折）。"""
        ys = self.geo["iso_ramp_y"]
        for i in range(len(ys) - 1):
            assert ys[i + 1] >= ys[i], f"ys[{i+1}]={ys[i+1]:.4f} < ys[{i}]={ys[i]:.4f}"

    def test_iso_ramp_x_monotone_increasing(self):
        """iso_ramp_x 单调递增（流向向前）。"""
        xs = self.geo["iso_ramp_x"]
        for i in range(len(xs) - 1):
            assert xs[i + 1] > xs[i]

    def test_iso_ramp_length_51_points(self):
        """默认 n_steps=50 → iso_ramp 有 51 点。"""
        assert len(self.geo["iso_ramp_x"]) == 51
        assert len(self.geo["iso_ramp_y"]) == 51

    def test_iso_ramp_key_exists_in_geo(self):
        """返回字典包含 iso_ramp_x 和 iso_ramp_y 键。"""
        assert "iso_ramp_x" in self.geo
        assert "iso_ramp_y" in self.geo

    def test_geo_theta_iso_zero_still_has_keys(self):
        """theta_iso_deg=0 时 iso_ramp_x/iso_ramp_y 键存在但为空数组。"""
        st0 = design_external_2d(M0=2.0, N_stages=3, M_EX=1.40)
        geo0 = external_2d_geometry(st0, self.angles, D2=1.0, theta_iso_deg=0.0)
        assert "iso_ramp_x" in geo0
        assert "iso_ramp_y" in geo0
        assert isinstance(geo0["iso_ramp_x"], np.ndarray)
        assert len(geo0["iso_ramp_x"]) == 0


# ---------------------------------------------------------------------------
# 唇口圆弧几何（cowl lip geometry，步骤 ⑫）
# ---------------------------------------------------------------------------

class TestLipGeometry:
    """cowl 唇口圆弧几何（lip_mode=1/2）验证。"""

    def setup_method(self):
        self.stations = design_external_2d(M0=2.0, N_stages=3, M_EX=1.40)
        self.angles   = oswatitsch_angles(M0=2.0, N_stages=3, M_EX=1.40)

    # --- Mode 1（尖唇口，默认）---

    def test_lip_mode1_lip_coords_none(self):
        """Mode 1：返回 'lip_coords' = None。"""
        geo = external_2d_geometry(self.stations, self.angles, D2=1.0, lip_mode=1)
        assert "lip_coords" in geo
        assert geo["lip_coords"] is None

    def test_lip_mode1_regression_H_capture(self):
        """Mode 1 与默认调用（无 lip_mode 参数）H_capture 一致（回归）。"""
        geo_default = external_2d_geometry(self.stations, self.angles, D2=1.0)
        geo_mode1   = external_2d_geometry(self.stations, self.angles, D2=1.0, lip_mode=1)
        assert geo_default["H_capture"] == pytest.approx(geo_mode1["H_capture"])

    def test_lip_mode1_regression_ramp_points(self):
        """Mode 1 与默认调用 ramp_points 完全一致（回归）。"""
        geo_default = external_2d_geometry(self.stations, self.angles, D2=1.0)
        geo_mode1   = external_2d_geometry(self.stations, self.angles, D2=1.0, lip_mode=1)
        assert geo_default["ramp_points"] == geo_mode1["ramp_points"]

    def test_lip_mode1_lip_mode_key(self):
        """lip_mode 键值为 1。"""
        geo = external_2d_geometry(self.stations, self.angles, D2=1.0, lip_mode=1)
        assert geo["lip_mode"] == 1

    # --- Mode 2（圆弧唇口）---

    def test_lip_mode2_keys_present(self):
        """Mode 2 返回四个唇口坐标数组键。"""
        geo = external_2d_geometry(self.stations, self.angles, D2=1.0, lip_mode=2, r_lip=0.005)
        for key in ("lip_outer_x", "lip_outer_y", "lip_inner_x", "lip_inner_y"):
            assert key in geo, f"缺少键 '{key}'"

    def test_lip_mode2_inner_x0_continuity(self):
        """lip_inner_x[0] 等于 x_cowl（与外弧终点 C0 连续）。"""
        geo = external_2d_geometry(self.stations, self.angles, D2=1.0, lip_mode=2, r_lip=0.005)
        assert geo["lip_inner_x"][0] == pytest.approx(geo["x_cowl"], abs=1e-9)

    def test_lip_mode2_outer_inner_junction(self):
        """外弧终点与内弧起点坐标完全重合（C0 连续）。"""
        geo = external_2d_geometry(self.stations, self.angles, D2=1.0, lip_mode=2, r_lip=0.005)
        assert geo["lip_outer_x"][-1] == pytest.approx(geo["lip_inner_x"][0], abs=1e-9)
        assert geo["lip_outer_y"][-1] == pytest.approx(geo["lip_inner_y"][0], abs=1e-9)

    def test_lip_mode2_outer_arc_20_points(self):
        """外弧 20 个离散点。"""
        geo = external_2d_geometry(self.stations, self.angles, D2=1.0, lip_mode=2, r_lip=0.005)
        assert len(geo["lip_outer_x"]) == 20
        assert len(geo["lip_outer_y"]) == 20

    def test_lip_mode2_inner_arc_20_points(self):
        """内弧 20 个离散点。"""
        geo = external_2d_geometry(self.stations, self.angles, D2=1.0, lip_mode=2, r_lip=0.005)
        assert len(geo["lip_inner_x"]) == 20
        assert len(geo["lip_inner_y"]) == 20

    def test_lip_mode2_outer_arc_radius(self):
        """外弧各点到圆心距离等于 r_lip（误差 < 1e-9）。"""
        r_lip = 0.005
        geo = external_2d_geometry(self.stations, self.angles, D2=1.0, lip_mode=2, r_lip=r_lip)
        cx = -r_lip
        cy = geo["y_cowl"] - r_lip
        for x, y in zip(geo["lip_outer_x"], geo["lip_outer_y"]):
            dist = math.hypot(x - cx, y - cy)
            assert abs(dist - r_lip) < 1e-9, f"外弧点 ({x:.6f},{y:.6f}) 到圆心距离 {dist:.9f} ≠ {r_lip}"

    def test_lip_mode2_inner_arc_radius(self):
        """内弧各点到圆心距离等于 r_lip（误差 < 1e-9）。"""
        r_lip = 0.005
        geo = external_2d_geometry(self.stations, self.angles, D2=1.0, lip_mode=2, r_lip=r_lip)
        cx = 0.0
        cy = geo["y_cowl"] - 2.0 * r_lip
        for x, y in zip(geo["lip_inner_x"], geo["lip_inner_y"]):
            dist = math.hypot(x - cx, y - cy)
            assert abs(dist - r_lip) < 1e-9, f"内弧点 ({x:.6f},{y:.6f}) 到圆心距离 {dist:.9f} ≠ {r_lip}"

    def test_lip_mode2_outer_arc_starts_upstream(self):
        """外弧起点 x < 0（在 cowl 唇口上游）。"""
        geo = external_2d_geometry(self.stations, self.angles, D2=1.0, lip_mode=2, r_lip=0.005)
        assert geo["lip_outer_x"][0] < 0.0

    def test_lip_mode2_r_lip_too_large_raises(self):
        """r_lip ≥ 喉道高度时抛出 ValueError。"""
        geo_ref = external_2d_geometry(self.stations, self.angles, D2=1.0)
        H_throat = geo_ref["y_cowl"] - geo_ref["ramp_points"][-1][1]
        with pytest.raises(ValueError):
            external_2d_geometry(
                self.stations, self.angles, D2=1.0,
                lip_mode=2, r_lip=H_throat + 0.01,
            )

    def test_lip_mode2_r_lip_zero_raises(self):
        """r_lip=0 时 lip_mode=2 抛出 ValueError（半径须 > 0）。"""
        with pytest.raises(ValueError):
            external_2d_geometry(self.stations, self.angles, D2=1.0, lip_mode=2, r_lip=0.0)

    def test_lip_mode_invalid_raises(self):
        """lip_mode 非 1/2 时抛出 ValueError。"""
        with pytest.raises(ValueError):
            external_2d_geometry(self.stations, self.angles, D2=1.0, lip_mode=3)
