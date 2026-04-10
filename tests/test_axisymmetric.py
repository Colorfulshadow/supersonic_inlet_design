"""
tests/test_axisymmetric.py
===========================
轴对称锥形激波进气道气动设计单元测试。

设计工况（物理自洽参数）：
    M₀=2.0，δ_c=22.0°，M_EX=1.30

    T-M 结果：β_c≈39.60°，M_cone≈1.51（> M_EX=1.30，等熵段为真正压缩）
    验证基准：
      - total_pressure_recovery() ≥ 0.958（σ ≈ 0.963）
      - stEX.M = 1.30 ± 0.01
      - M_cone > M_EX（True）

注意：原 δ_c=31.37° 设计给出 M_cone=1.21 < M_EX=1.30（等熵膨胀，物理错误），
已在 design_axisymmetric() 中触发 ValueError。
"""

import pytest

from core.flow_stations import InletFlowStations
from inlets.axisymmetric.aero_design import design_axisymmetric
from inlets.axisymmetric.geometry import axisymmetric_geometry

# ------------------------------------------------------------------
# 设计点参数（物理自洽）
# ------------------------------------------------------------------
M0_DESIGN = 2.0
DELTA_C_DESIGN = 22.0   # M_cone ≈ 1.51 > M_EX=1.30，等熵段为压缩
M_EX_DESIGN = 1.30


@pytest.fixture(scope="module")
def stations_design():
    """标准设计工况结果（复用，避免重复 T-M 求解）。"""
    return design_axisymmetric(
        M0=M0_DESIGN,
        delta_c_deg=DELTA_C_DESIGN,
        M_EX=M_EX_DESIGN,
    )


# ------------------------------------------------------------------
# 1. 返回类型
# ------------------------------------------------------------------

class TestReturnType:
    def test_returns_inlet_flow_stations(self, stations_design):
        assert isinstance(stations_design, InletFlowStations)

    def test_all_required_stations_set(self, stations_design):
        assert stations_design.st0  is not None
        assert stations_design.stEX is not None
        assert stations_design.stNS is not None
        assert stations_design.st1  is not None
        assert stations_design.st2  is not None

    def test_extra_dict_present(self, stations_design):
        assert hasattr(stations_design, "extra")
        extra = stations_design.extra
        for key in ("delta_c_deg", "beta_c_deg", "M_cone", "sigma_cone", "sigma_isentropic"):
            assert key in extra, f"extra 字典缺少键 '{key}'"


# ------------------------------------------------------------------
# 2. 验证基准（物理自洽参数）
# ------------------------------------------------------------------

class TestVerificationBenchmarks:
    def test_total_pressure_recovery_ge_0958(self, stations_design):
        """δ_c=22°，M_EX=1.30：σ ≈ 0.963 ≥ 0.958（Slater 2023 Table 3 目标）。"""
        sigma = stations_design.total_pressure_recovery()
        assert sigma >= 0.958, f"总压恢复 σ={sigma:.6f} < 0.958"

    def test_stEX_mach_equals_M_EX(self, stations_design):
        assert stations_design.stEX.M == pytest.approx(M_EX_DESIGN, abs=0.01)

    def test_delta_c_deg_matches_input(self, stations_design):
        assert stations_design.extra["delta_c_deg"] == pytest.approx(DELTA_C_DESIGN, abs=0.1)

    def test_M_cone_gt_M_EX(self, stations_design):
        """M_cone > M_EX：等熵段为真正的压缩（物理自洽核心条件）。"""
        assert stations_design.extra["M_cone"] > M_EX_DESIGN


# ------------------------------------------------------------------
# 3. 物理约束
# ------------------------------------------------------------------

class TestPhysicalConstraints:
    def test_sigma_cone_in_valid_range(self, stations_design):
        sigma_cone = stations_design.extra["sigma_cone"]
        assert 0.90 <= sigma_cone <= 1.0, (
            f"锥形激波总压比 σ_cone={sigma_cone:.6f} 不在 [0.90, 1.0] 范围内"
        )

    def test_beta_c_greater_than_delta_c(self, stations_design):
        beta_c = stations_design.extra["beta_c_deg"]
        delta_c = stations_design.extra["delta_c_deg"]
        assert beta_c > delta_c, (
            f"激波角 β_c={beta_c:.4f}° 应大于锥角 δ_c={delta_c:.4f}°"
        )

    def test_sigma_isentropic_equals_1(self, stations_design):
        assert stations_design.extra["sigma_isentropic"] == pytest.approx(1.0)

    def test_st0_pt_normalized(self, stations_design):
        assert stations_design.st0.p_t == pytest.approx(1.0)

    def test_total_temp_invariant(self, stations_design):
        """正激波绝热，总温不变。"""
        assert stations_design.stNS.T_t == pytest.approx(1.0)
        assert stations_design.st2.T_t  == pytest.approx(1.0)

    def test_pt_chain_decreasing(self, stations_design):
        """总压沿流向单调递减。"""
        pts = [
            stations_design.st0.p_t,
            stations_design.stEX.p_t,
            stations_design.stNS.p_t,
        ]
        for i in range(len(pts) - 1):
            assert pts[i] > pts[i + 1], (
                f"总压在站位 {i} → {i+1} 未递减：{pts[i]:.6f} ≤ {pts[i+1]:.6f}"
            )

    def test_st1_equals_stNS(self, stations_design):
        assert stations_design.st1.M   == pytest.approx(stations_design.stNS.M)
        assert stations_design.st1.p_t == pytest.approx(stations_design.stNS.p_t)

    def test_st2_equals_st1(self, stations_design):
        assert stations_design.st2.M   == pytest.approx(stations_design.st1.M)
        assert stations_design.st2.p_t == pytest.approx(stations_design.st1.p_t)


# ------------------------------------------------------------------
# 4. 错误处理
# ------------------------------------------------------------------

class TestErrorHandling:
    def test_M0_le_1_raises_ValueError(self):
        with pytest.raises(ValueError, match="M0"):
            design_axisymmetric(M0=1.0, delta_c_deg=20.0)

    def test_M0_subsonic_raises_ValueError(self):
        with pytest.raises(ValueError):
            design_axisymmetric(M0=0.8, delta_c_deg=20.0)

    def test_M_cone_lt_M_EX_raises_ValueError(self):
        """δ_c=31.37°，M0=2.0 → M_cone=1.21 < M_EX=1.30，应抛出 ValueError。"""
        with pytest.raises(ValueError, match="物理矛盾"):
            design_axisymmetric(M0=2.0, delta_c_deg=31.37, M_EX=1.30)

    def test_M_cone_lt_M_EX_error_message_informative(self):
        """错误信息应包含 M_cone 和 M_EX 的值。"""
        with pytest.raises(ValueError) as exc_info:
            design_axisymmetric(M0=2.0, delta_c_deg=31.37, M_EX=1.30)
        msg = str(exc_info.value)
        assert "M_cone" in msg
        assert "M_EX" in msg


# ------------------------------------------------------------------
# 5. M_cone 合理性
# ------------------------------------------------------------------

class TestMcone:
    def test_M_cone_supersonic(self, stations_design):
        """锥面马赫数应大于 1（超声速锥形流）。"""
        assert stations_design.extra["M_cone"] > 1.0

    def test_M_cone_less_than_M0(self, stations_design):
        """锥面马赫数须小于来流马赫数（激波减速）。"""
        assert stations_design.extra["M_cone"] < M0_DESIGN


# ------------------------------------------------------------------
# 6. 自动优化锥角（delta_c_deg=None）
# ------------------------------------------------------------------

class TestAutoOptimization:
    def test_auto_delta_c_returns_valid_result(self):
        stations = design_axisymmetric(M0=M0_DESIGN, delta_c_deg=None, M_EX=M_EX_DESIGN)
        assert isinstance(stations, InletFlowStations)
        assert stations.total_pressure_recovery() > 0.90
        assert stations.extra["delta_c_deg"] > 0
        assert stations.extra["M_cone"] >= M_EX_DESIGN

    def test_auto_delta_c_satisfies_physical_constraint(self):
        """自动搜索结果必须满足 M_cone ≥ M_EX。"""
        stations = design_axisymmetric(M0=M0_DESIGN, delta_c_deg=None, M_EX=M_EX_DESIGN)
        assert stations.extra["M_cone"] >= M_EX_DESIGN


# ------------------------------------------------------------------
# 7. 几何测试（axisymmetric_geometry）
# ------------------------------------------------------------------

GEO_D2 = 1.37  # 测试用出口直径（m）；须 >= 2*r_cowl ≈ 1.249m，此处取 1.37m（约 10% 扩张量）


@pytest.fixture(scope="module")
def geo_design(stations_design):
    """标准设计工况几何结果（复用 stations_design 以避免重复 T-M 求解）。"""
    return axisymmetric_geometry(stations_design, D2=GEO_D2)


class TestGeometryBasicShape:
    def test_r_exit_gt_r_cowl(self, geo_design):
        """
        出口半径大于 cowl 唇口半径（r_exit > r_cowl），cowl 向外扩张。

        r_cowl 以 D2_ref=1.0m 为参考尺度（≈0.624m），与输入 D2 无关；
        GEO_D2=1.37m 使 r_exit=0.685m > r_cowl≈0.624m（约 10% 扩张量）。
        """
        assert geo_design["r_exit"] > geo_design["r_cowl"]

    def test_r_cb_tip_zero(self, geo_design):
        """中心锥尖端半径为 0（尖锥）。"""
        assert geo_design["r_cb_tip"] == pytest.approx(0.0)

    def test_r_cb_base_lt_r_cowl(self, geo_design):
        """中心锥底部半径小于 cowl 半径（有环形间隙）。"""
        assert geo_design["r_cb_base"] < geo_design["r_cowl"]

    def test_r_cb_base_positive(self, geo_design):
        """中心锥底部半径为正（锥体有实体）。"""
        assert geo_design["r_cb_base"] > 0.0

    def test_r_throat_positive(self, geo_design):
        """环形喉道径向高度为正。"""
        assert geo_design["r_throat"] > 0.0

    def test_r_throat_equals_cowl_minus_base(self, geo_design):
        """r_throat = r_cowl - r_cb_base。"""
        assert geo_design["r_throat"] == pytest.approx(
            geo_design["r_cowl"] - geo_design["r_cb_base"], rel=1e-6
        )

    def test_r_exit_equals_D2_half(self, geo_design):
        """出口半径 = D2 / 2。"""
        assert geo_design["r_exit"] == pytest.approx(GEO_D2 / 2.0)


class TestGeometryAxialPositions:
    def test_x_cone_tip_lt_x_cowl(self, geo_design):
        """锥尖在 cowl 唇口上游（x_cone_tip < x_cowl = 0）。"""
        assert geo_design["x_cone_tip"] < geo_design["x_cowl"]

    def test_x_cowl_zero(self, geo_design):
        """cowl 唇口定义为坐标原点（x_cowl = 0）。"""
        assert geo_design["x_cowl"] == pytest.approx(0.0)

    def test_x_diffuser_end_positive(self, geo_design):
        """扩压段末端在 cowl 唇口下游（x_diffuser_end > 0）。"""
        assert geo_design["x_diffuser_end"] > 0.0


class TestGeometryProfiles:
    def test_profile_cb_monotone_x(self, geo_design):
        """中心锥母线 x 坐标单调递增。"""
        xs = [p[0] for p in geo_design["profile_cb"]]
        assert all(xs[i] < xs[i + 1] for i in range(len(xs) - 1))

    def test_profile_cb_monotone_r(self, geo_design):
        """中心锥母线 r 坐标单调递增（从锥尖 r=0 到底部 r=r_cb_base）。"""
        rs = [p[1] for p in geo_design["profile_cb"]]
        assert all(rs[i] <= rs[i + 1] for i in range(len(rs) - 1))

    def test_profile_cb_tip_at_origin(self, geo_design):
        """中心锥母线首点 r = 0（尖锥顶点）。"""
        assert geo_design["profile_cb"][0][1] == pytest.approx(0.0)

    def test_profile_cb_base_matches(self, geo_design):
        """中心锥母线末点 r = r_cb_base，x = 0（唇口截面）。"""
        last = geo_design["profile_cb"][-1]
        assert last[0] == pytest.approx(0.0, abs=1e-9)
        assert last[1] == pytest.approx(geo_design["r_cb_base"], rel=1e-6)

    def test_profile_cowl_monotone_x(self, geo_design):
        """cowl 内壁母线 x 坐标单调递增。"""
        xs = [p[0] for p in geo_design["profile_cowl"]]
        assert all(xs[i] < xs[i + 1] for i in range(len(xs) - 1))

    def test_profile_cowl_monotone_r_increasing(self, geo_design):
        """
        cowl 内壁母线 r 坐标单调递增（r_exit > r_cowl，cowl 向外扩张）。

        r_cowl ≈ 0.624m（参考尺度），r_exit = GEO_D2/2 = 0.685m；
        cowl 从唇口向出口线性外扩约 10%，满足亚声速扩压正确方向。
        """
        rs = [p[1] for p in geo_design["profile_cowl"]]
        assert all(rs[i] <= rs[i + 1] for i in range(len(rs) - 1))

    def test_profile_cowl_starts_at_cowl_lip(self, geo_design):
        """cowl 内壁母线首点 = (x_cowl=0, r_cowl)。"""
        first = geo_design["profile_cowl"][0]
        assert first[0] == pytest.approx(0.0, abs=1e-9)
        assert first[1] == pytest.approx(geo_design["r_cowl"], rel=1e-6)

    def test_profile_cowl_ends_at_exit(self, geo_design):
        """cowl 内壁母线末点 = (x_diffuser_end, r_exit)。"""
        last = geo_design["profile_cowl"][-1]
        assert last[0] == pytest.approx(geo_design["x_diffuser_end"], rel=1e-6)
        assert last[1] == pytest.approx(geo_design["r_exit"], rel=1e-6)


class TestGeometryErrorHandling:
    def test_negative_D2_raises(self, stations_design):
        with pytest.raises(ValueError):
            axisymmetric_geometry(stations_design, D2=-1.0)

    def test_zero_D2_raises(self, stations_design):
        with pytest.raises(ValueError):
            axisymmetric_geometry(stations_design, D2=0.0)

    def test_D2_too_small_raises(self, stations_design):
        """D2=1.0m 使 r_exit=0.5m < r_cowl≈0.624m，应抛出 ValueError。"""
        with pytest.raises(ValueError, match="cowl 将向内收缩"):
            axisymmetric_geometry(stations_design, D2=1.0)
