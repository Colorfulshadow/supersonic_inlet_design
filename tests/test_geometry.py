"""
tests/test_geometry.py
======================
皮托管进气道几何关键点验证（第⑤步首批用例）。
"""

import pytest
from inlets.pitot.aero_design import design_pitot
from inlets.pitot.geometry import pitot_geometry


@pytest.fixture
def geo_m2():
    """M0=2.0，D2=1.0 m 的标准几何结果。"""
    st = design_pitot(M0=2.0)
    return pitot_geometry(st, D2=1.0)


class TestBasicDimensions:
    def test_r_exit_equals_half_D2(self, geo_m2):
        assert geo_m2["r_exit"] == pytest.approx(0.5)

    def test_r_capture_le_r_exit(self, geo_m2):
        """含总压恢复修正后，捕获半径 ≤ 出口半径（扩压段向外扩张）。"""
        assert geo_m2["r_capture"] <= geo_m2["r_exit"]

    def test_r_capture_range(self, geo_m2):
        """M0=2.0，D2=1.0 时捕获半径在合理范围（0.45~0.55 m）。
        由质量守恒 σ·A_capture·φ(M0)=A2·φ(M_NS) 得 r_capture≈0.500 m。
        """
        assert 0.45 <= geo_m2["r_capture"] <= 0.55

    def test_r_throat_equals_r_capture(self, geo_m2):
        """皮托管无内收缩，喉道半径 = 捕获半径。"""
        assert geo_m2["r_throat"] == pytest.approx(geo_m2["r_capture"])


class TestAxialCoordinates:
    def test_x_shock_is_zero(self, geo_m2):
        assert geo_m2["x_shock"] == pytest.approx(0.0)

    def test_x_diffuser_end_gt_x_shock(self, geo_m2):
        """扩压段末端在激波下游。"""
        assert geo_m2["x_diffuser_end"] > geo_m2["x_shock"]

    def test_custom_L_diffuser(self):
        """指定 L_diffuser 时，x_diffuser_end 应等于 L_diffuser。"""
        st = design_pitot(M0=2.0)
        geo = pitot_geometry(st, D2=1.0, L_diffuser=5.0)
        assert geo["x_diffuser_end"] == pytest.approx(5.0)

    def test_auto_L_diffuser_positive(self, geo_m2):
        """自动估算的扩压段长度应为正值。"""
        assert geo_m2["x_diffuser_end"] > 0.0


class TestProfile:
    def test_profile_at_least_4_points(self, geo_m2):
        assert len(geo_m2["profile"]) >= 4

    def test_profile_x_monotone_increasing(self, geo_m2):
        xs = [p[0] for p in geo_m2["profile"]]
        for i in range(1, len(xs)):
            assert xs[i] > xs[i - 1], (
                f"profile x 坐标在第 {i} 点不单调：{xs[i-1]:.4f} → {xs[i]:.4f}"
            )

    def test_profile_r_nonnegative(self, geo_m2):
        for x, r in geo_m2["profile"]:
            assert r >= 0.0, f"型线出现负半径 r={r:.4f} at x={x:.4f}"

    def test_profile_last_r_equals_r_exit(self, geo_m2):
        last_r = geo_m2["profile"][-1][1]
        assert last_r == pytest.approx(geo_m2["r_exit"])

    def test_profile_last_x_equals_x_diffuser_end(self, geo_m2):
        last_x = geo_m2["profile"][-1][0]
        assert last_x == pytest.approx(geo_m2["x_diffuser_end"])


class TestInvalidInput:
    def test_negative_D2_raises(self):
        st = design_pitot(M0=2.0)
        with pytest.raises(ValueError):
            pitot_geometry(st, D2=-1.0)

    def test_zero_D2_raises(self):
        st = design_pitot(M0=2.0)
        with pytest.raises(ValueError):
            pitot_geometry(st, D2=0.0)
