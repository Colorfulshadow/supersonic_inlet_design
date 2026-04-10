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

import pytest
from inlets.external_2d.aero_design import design_external_2d, oswatitsch_angles
from inlets.external_2d.geometry import external_2d_geometry
from core.compressible_flow import max_turning_angle
from core.flow_stations import InletFlowStations


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
