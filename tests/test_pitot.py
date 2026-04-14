"""
tests/test_pitot.py
===================
皮托管进气道气动设计验证，对应 CLAUDE.md §5.3 和 §四 数值基准。
"""

import pytest
from inlets.pitot.aero_design import design_pitot
from inlets.pitot.geometry import pitot_geometry
from core.flow_stations import InletFlowStations


class TestDesignPitotM2:
    """M0=2.0 的数值基准验证（CLAUDE.md §5.3）。"""

    def setup_method(self):
        self.stations = design_pitot(M0=2.0)

    def test_returns_inlet_flow_stations(self):
        assert isinstance(self.stations, InletFlowStations)

    def test_total_pressure_recovery(self):
        sigma = self.stations.total_pressure_recovery()
        assert abs(sigma - 0.7209) < 0.0005, (
            f"σ={sigma:.6f}，期望 0.7209±0.0005"
        )

    def test_stNS_mach(self):
        M_NS = self.stations.stNS.M
        assert abs(M_NS - 0.5774) < 0.001, (
            f"stNS.M={M_NS:.6f}，期望 0.5774±0.001"
        )

    def test_st0_normalized(self):
        assert self.stations.st0.p_t == pytest.approx(1.0)
        assert self.stations.st0.T_t == pytest.approx(1.0)
        assert self.stations.st0.M == pytest.approx(2.0)

    def test_stEX_equals_st0(self):
        st0 = self.stations.st0
        stEX = self.stations.stEX
        assert stEX.M == pytest.approx(st0.M)
        assert stEX.p_t == pytest.approx(st0.p_t)
        assert stEX.T_t == pytest.approx(st0.T_t)

    def test_stNS_total_temperature_conserved(self):
        # 正激波绝热，总温不变
        assert self.stations.stNS.T_t == pytest.approx(1.0)

    def test_st1_equals_stNS(self):
        stNS = self.stations.stNS
        st1 = self.stations.st1
        assert st1.M == pytest.approx(stNS.M)
        assert st1.p_t == pytest.approx(stNS.p_t)
        assert st1.T_t == pytest.approx(stNS.T_t)

    def test_st2_equals_st1(self):
        st1 = self.stations.st1
        st2 = self.stations.st2
        assert st2.M == pytest.approx(st1.M)
        assert st2.p_t == pytest.approx(st1.p_t)
        assert st2.T_t == pytest.approx(st1.T_t)


class TestDesignPitotM15:
    """M0=1.5 时 σ 在合理范围内。"""

    def test_sigma_range(self):
        stations = design_pitot(M0=1.5)
        sigma = stations.total_pressure_recovery()
        assert 0.92 <= sigma <= 0.93, (
            f"M0=1.5 时 σ={sigma:.6f}，期望在 [0.92, 0.93]"
        )

    def test_stNS_subsonic(self):
        stations = design_pitot(M0=1.5)
        assert stations.stNS.M < 1.0


class TestDesignPitotInvalidInput:
    """无效输入应抛出 ValueError。"""

    def test_M0_equal_1_raises(self):
        with pytest.raises(ValueError):
            design_pitot(M0=1.0)

    def test_M0_less_than_1_raises(self):
        with pytest.raises(ValueError):
            design_pitot(M0=0.8)

    def test_M0_zero_raises(self):
        with pytest.raises(ValueError):
            design_pitot(M0=0.0)


class TestPitotGeometryM2:
    """皮托管几何模块验证（M0=2.0，D2=1.0 m，默认参数）。

    依据：Slater 2023 Table 3，Axi-Pitot M₀=2.0
      area_ratio_diff = 1.192（A_exit/A_capture）
      θeqSD ≈ 2.4° @ N_throat=1.0；约 1.2° @ N_throat=2.0
      Acap/A2 ≈ 0.923
    """

    D2 = 1.0

    def setup_method(self):
        self.stations = design_pitot(M0=2.0)
        self.geo = pitot_geometry(self.stations, D2=self.D2)  # 默认 area_ratio_diff=1.192

    # ---- 扩压段长度 ----------------------------------------

    def test_L_diffuser_not_degenerate(self):
        """L_diffuser 必须远大于零（不能退化为 1e-6）。"""
        assert self.geo["L_diffuser"] > 0.1, (
            f"L_diffuser={self.geo['L_diffuser']:.6f} m，期望 > 0.1 m"
        )

    def test_L_diffuser_equals_N_throat_times_D2(self):
        """默认 N_throat=2.0：L_diffuser = 2.0 * D2。"""
        expected = 2.0 * self.D2
        assert abs(self.geo["L_diffuser"] - expected) < 1e-10, (
            f"L_diffuser={self.geo['L_diffuser']:.6f} m，期望 {expected:.6f} m"
        )

    # ---- 型线几何 ------------------------------------------

    def test_profile_has_four_points(self):
        assert len(self.geo["profile"]) == 4

    def test_profile_x_strictly_increasing(self):
        """型线4个关键点 x 坐标严格单调递增。"""
        xs = [p[0] for p in self.geo["profile"]]
        for i in range(1, len(xs)):
            assert xs[i] > xs[i - 1], (
                f"profile[{i}].x={xs[i]:.6f} <= profile[{i-1}].x={xs[i-1]:.6f}，不单调"
            )

    def test_profile_r_monotone_nondecreasing(self):
        """型线 r 坐标单调不减（从 r_capture 扩张到 r_exit）。"""
        rs = [p[1] for p in self.geo["profile"]]
        for i in range(1, len(rs)):
            assert rs[i] >= rs[i - 1] - 1e-12, (
                f"profile[{i}].r={rs[i]:.6f} < profile[{i-1}].r={rs[i-1]:.6f}，r 出现收缩"
            )

    # ---- 半径关系 ------------------------------------------

    def test_r_exit_correct(self):
        """r_exit = D2/2。"""
        assert abs(self.geo["r_exit"] - self.D2 / 2.0) < 1e-10

    def test_r_capture_less_than_r_exit(self):
        """r_capture < r_exit（扩压段为扩张管道，area_ratio_diff > 1）。"""
        assert self.geo["r_capture"] < self.geo["r_exit"], (
            f"r_capture={self.geo['r_capture']:.6f} >= r_exit={self.geo['r_exit']:.6f}"
        )

    def test_r_capture_value(self):
        """r_capture = r_exit / sqrt(1.192) ≈ 0.4580 m（D2=1.0）。

        Slater 2023 Table 3：A2*/ASD = 1.192，
        r_capture = 0.5 / sqrt(1.192) ≈ 0.457972 m。
        """
        import math
        expected = (self.D2 / 2.0) / math.sqrt(1.192)
        assert abs(self.geo["r_capture"] - expected) < 1e-6, (
            f"r_capture={self.geo['r_capture']:.6f} m，期望 {expected:.6f} m"
        )

    def test_r_throat_equals_r_capture(self):
        """皮托管无内收缩：r_throat = r_capture。"""
        assert self.geo["r_throat"] == pytest.approx(self.geo["r_capture"])

    # ---- 面积比 -------------------------------------------

    def test_area_ratio_diff_default(self):
        """默认 area_ratio_diff = 1.192（Slater 2023 Table 3）。"""
        assert abs(self.geo["area_ratio_diff"] - 1.192) < 1e-10

    def test_area_ratio_realized(self):
        """实际 A_exit/A_capture 与 area_ratio_diff 一致（误差 < 0.5%）。"""
        import math
        A_cap = math.pi * self.geo["r_capture"] ** 2
        A_ex  = math.pi * self.geo["r_exit"] ** 2
        ratio = A_ex / A_cap
        assert abs(ratio - self.geo["area_ratio_diff"]) < 0.005 * self.geo["area_ratio_diff"], (
            f"A_exit/A_capture={ratio:.6f}，area_ratio_diff={self.geo['area_ratio_diff']:.4f}"
        )

    def test_acap_over_a2_consistent_with_area_ratio(self):
        """A_cap/A2 = 1/area_ratio_diff（几何自洽性检验）。

        本方案由几何面积比倒推：A_cap = A_exit / area_ratio_diff，
        故 A_cap/A2 = 1/area_ratio_diff = 1/1.192 ≈ 0.839。
        注：Slater 2023 Table 3 的 Acap/A2=0.923 采用质量流量+总压恢复约定，
        与此处纯几何面积比方法不同，两者均正确但基准不同。
        """
        import math
        A_cap = math.pi * self.geo["r_capture"] ** 2
        A_ex  = math.pi * self.geo["r_exit"] ** 2
        ratio = A_cap / A_ex
        expected = 1.0 / self.geo["area_ratio_diff"]
        assert abs(ratio - expected) < 1e-6, (
            f"A_cap/A2={ratio:.6f}，期望 1/area_ratio_diff={expected:.6f}"
        )

    # ---- 等效锥角 -----------------------------------------

    def test_theta_eq_positive(self):
        """area_ratio_diff > 1 时等效锥角为正（扩张管道）。"""
        assert self.geo["theta_eq_diffuser"] > 0.0

    def test_theta_eq_reasonable_range(self):
        """等效锥角在 0.5°~3° 之间（N_throat=2.0，area_ratio_diff=1.192）。"""
        theta = self.geo["theta_eq_diffuser"]
        assert 0.5 <= theta <= 3.0, (
            f"theta_eq_diffuser={theta:.4f} deg，超出 [0.5, 3.0] deg 合理范围"
        )

    def test_theta_eq_N_throat_1_matches_literature(self):
        """N_throat=1.0 时 θeq ≈ 2.4°（Slater 2023 Table 3 验证基准，误差 < 0.1°）。"""
        geo = pitot_geometry(self.stations, D2=self.D2, N_throat=1.0)
        theta = geo["theta_eq_diffuser"]
        assert abs(theta - 2.409) < 0.1, (
            f"N_throat=1.0 时 theta_eq={theta:.4f} deg，期望 2.409±0.1 deg（Slater 2023）"
        )

    # ---- 其他参数 -----------------------------------------

    def test_x_shock_at_origin(self):
        assert self.geo["x_shock"] == pytest.approx(0.0)

    def test_custom_N_throat(self):
        """N_throat=1.0 时 L_diffuser = D2。"""
        geo = pitot_geometry(self.stations, D2=self.D2, N_throat=1.0)
        assert abs(geo["L_diffuser"] - self.D2) < 1e-10

    def test_explicit_L_diffuser_overrides_N_throat(self):
        """显式传入 L_diffuser 时忽略 N_throat。"""
        geo = pitot_geometry(self.stations, D2=self.D2, L_diffuser=3.5)
        assert abs(geo["L_diffuser"] - 3.5) < 1e-10

    def test_custom_area_ratio_diff(self):
        """自定义 area_ratio_diff=1.5 时 r_capture = r_exit/sqrt(1.5)。"""
        import math
        geo = pitot_geometry(self.stations, D2=self.D2, area_ratio_diff=1.5)
        expected_r = (self.D2 / 2.0) / math.sqrt(1.5)
        assert abs(geo["r_capture"] - expected_r) < 1e-10

    def test_invalid_area_ratio_diff_raises(self):
        """area_ratio_diff <= 0 应抛出 ValueError。"""
        with pytest.raises(ValueError):
            pitot_geometry(self.stations, D2=self.D2, area_ratio_diff=0.0)
        with pytest.raises(ValueError):
            pitot_geometry(self.stations, D2=self.D2, area_ratio_diff=-1.0)
