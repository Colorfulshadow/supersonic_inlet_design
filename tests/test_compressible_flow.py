"""
tests/test_compressible_flow.py
================================
pytest 测试：core/compressible_flow.py 数值验证。

运行方式：
    venv\\Scripts\\python.exe -m pytest tests/test_compressible_flow.py -v
"""

import math
import pytest
from core.compressible_flow import (
    shock_pt_ratio,
    shock_p_ratio,
    shock_T_ratio,
    M2_after_normal_shock,
    isentropic_T_ratio,
    isentropic_p_ratio,
    isentropic_M_from_pt_ratio,
    theta_from_beta_M,
    beta_from_theta_M,
    M2_after_oblique_shock,
    max_turning_angle,
)


# ---------------------------------------------------------------------------
# shock_pt_ratio — Slater 2023 基准值
# ---------------------------------------------------------------------------

class TestShockPtRatio:
    def test_normal_shock_M2(self):
        """皮托管正激波：M_n=2.0 → σ ≈ 0.7209"""
        assert abs(shock_pt_ratio(2.0) - 0.7209) < 0.0005

    def test_terminal_shock_2d(self):
        """二元终端正激波：M_n=1.40 → 0.9582"""
        assert abs(shock_pt_ratio(1.40) - 0.9582) < 0.0005

    def test_terminal_shock_axisym(self):
        """轴对称终端正激波：M_n=1.30 → 0.9794"""
        assert abs(shock_pt_ratio(1.30) - 0.9794) < 0.0005

    def test_no_shock_limit(self):
        """M_n→1 时总压比趋近于 1"""
        assert abs(shock_pt_ratio(1.0 + 1e-6) - 1.0) < 1e-4

    def test_decreases_with_Mn(self):
        """总压比随法向马赫数单调递减"""
        assert shock_pt_ratio(1.5) > shock_pt_ratio(2.0) > shock_pt_ratio(3.0)


# ---------------------------------------------------------------------------
# shock_p_ratio & shock_T_ratio — 物理一致性
# ---------------------------------------------------------------------------

class TestShockStaticRatios:
    def test_p_ratio_M2(self):
        """M_n=2.0 静压比：p2/p1 = 1 + 2γ/(γ+1)*(M²-1)"""
        expected = 1 + 2 * 1.4 / 2.4 * (4 - 1)
        assert abs(shock_p_ratio(2.0) - expected) < 1e-10

    def test_T_ratio_M1_limit(self):
        """M_n=1 时静温比 = 1（极限）"""
        assert abs(shock_T_ratio(1.0) - 1.0) < 1e-8

    def test_T_ratio_increases(self):
        """静温比随马赫数单调递增"""
        assert shock_T_ratio(2.0) > shock_T_ratio(1.5)

    def test_T_ratio_M2(self):
        """M_n=2.0 静温比解析值验证"""
        # T2/T1 = (p2/p1) * (2 + (γ-1)*M²) / (M²*(2γ-(γ-1)))
        p = shock_p_ratio(2.0)
        gamma = 1.4
        expected = p * (2 + 0.4 * 4) / (4 * (2.8 - 0.4))
        assert abs(shock_T_ratio(2.0) - expected) < 1e-10


# ---------------------------------------------------------------------------
# M2_after_normal_shock
# ---------------------------------------------------------------------------

class TestM2NormalShock:
    def test_M2_at_M1_2(self):
        """M1=2.0 → M2 ≈ 0.5774"""
        assert abs(M2_after_normal_shock(2.0) - 0.5774) < 0.0005

    def test_M2_subsonic(self):
        """正激波后必为亚声速"""
        for M1 in [1.1, 1.5, 2.0, 3.0, 5.0]:
            assert M2_after_normal_shock(M1) < 1.0

    def test_M2_limit_M1(self):
        """M1→1 时 M2→1"""
        assert abs(M2_after_normal_shock(1.0 + 1e-6) - 1.0) < 1e-4

    def test_M2_strong_shock(self):
        """强激波极限：M2 → sqrt((γ-1)/(2γ)) 当 M1→∞"""
        gamma = 1.4
        limit = math.sqrt((gamma - 1) / (2 * gamma))
        assert abs(M2_after_normal_shock(1e6) - limit) < 1e-4


# ---------------------------------------------------------------------------
# isentropic_T_ratio & isentropic_p_ratio
# ---------------------------------------------------------------------------

class TestIsentropic:
    def test_T_ratio_M0(self):
        assert abs(isentropic_T_ratio(0.0) - 1.0) < 1e-12

    def test_p_ratio_M0(self):
        assert abs(isentropic_p_ratio(0.0) - 1.0) < 1e-12

    def test_T_ratio_M1(self):
        """M=1：T0/T = 1 + (γ-1)/2 = 1.2（γ=1.4）"""
        assert abs(isentropic_T_ratio(1.0) - 1.2) < 1e-10

    def test_p_ratio_M1(self):
        """M=1：p0/p = 1.2^3.5 ≈ 1.8929"""
        expected = 1.2 ** (1.4 / 0.4)
        assert abs(isentropic_p_ratio(1.0) - expected) < 1e-8

    def test_isentropic_M_from_pt_ratio_roundtrip(self):
        """等熵总压比反算马赫数：round-trip 误差 < 1e-8"""
        for M in [0.5, 1.0, 1.5, 2.0, 3.0]:
            ratio = isentropic_p_ratio(M)
            M_back = isentropic_M_from_pt_ratio(ratio, M)
            assert abs(M_back - M) < 1e-8, f"M={M}: back={M_back}"


# ---------------------------------------------------------------------------
# theta_from_beta_M & beta_from_theta_M — round-trip
# ---------------------------------------------------------------------------

class TestObliqueShockAngles:
    def test_roundtrip_theta10_M2(self):
        """theta=10°, M=2.0：beta_from_theta_M → theta_from_beta_M 误差 < 1e-6°"""
        theta_deg = 10.0
        M = 2.0
        beta_deg = beta_from_theta_M(theta_deg, M)
        beta_rad = math.radians(beta_deg)
        theta_back_rad = theta_from_beta_M(beta_rad, M)
        theta_back_deg = math.degrees(theta_back_rad)
        assert abs(theta_back_deg - theta_deg) < 1e-6

    def test_roundtrip_various(self):
        """多组 (θ, M) round-trip 验证"""
        cases = [
            (5.0, 2.0),
            (15.0, 2.5),
            (20.0, 3.0),
            (17.34, 2.0),   # Slater 2023 基准楔角
        ]
        for theta_deg, M in cases:
            beta_deg = beta_from_theta_M(theta_deg, M)
            beta_rad = math.radians(beta_deg)
            theta_back = math.degrees(theta_from_beta_M(beta_rad, M))
            assert abs(theta_back - theta_deg) < 1e-6, (
                f"theta={theta_deg}, M={M}: back={theta_back}"
            )

    def test_beta_weak_solution(self):
        """弱激波角应小于正激波角（90°）"""
        beta = beta_from_theta_M(10.0, 2.0)
        assert beta < 90.0

    def test_beta_greater_than_mach_angle(self):
        """激波角应大于马赫角"""
        M = 2.0
        mu_deg = math.degrees(math.asin(1.0 / M))
        beta = beta_from_theta_M(10.0, M)
        assert beta > mu_deg


# ---------------------------------------------------------------------------
# M2_after_oblique_shock
# ---------------------------------------------------------------------------

class TestM2ObliqueShock:
    def test_consistency_with_theta_beta_M(self):
        """M2_after_oblique_shock 与 theta_from_beta_M 自洽：
        给定 θ=10°, M1=2.0，求 β，再求 M2，应超声速"""
        theta_deg = 10.0
        M1 = 2.0
        beta_deg = beta_from_theta_M(theta_deg, M1)
        beta_rad = math.radians(beta_deg)
        theta_rad = math.radians(theta_deg)
        M2 = M2_after_oblique_shock(M1, beta_rad, theta_rad)
        # 弱斜激波后仍为超声速
        assert M2 > 1.0

    def test_M2_reduces_to_normal_shock(self):
        """β=90° 时退化为正激波"""
        M1 = 2.0
        beta_rad = math.pi / 2
        theta_rad = 0.0   # 正激波时 θ=0
        M2_oblique = M2_after_oblique_shock(M1, beta_rad, theta_rad)
        M2_normal = M2_after_normal_shock(M1)
        assert abs(M2_oblique - M2_normal) < 1e-6


# ---------------------------------------------------------------------------
# max_turning_angle
# ---------------------------------------------------------------------------

class TestMaxTurningAngle:
    def test_M2_range(self):
        """M=2.0 最大转折角应在 22°~24° 之间"""
        theta_max = max_turning_angle(2.0)
        assert 22.0 < theta_max < 24.0, f"max_turning_angle(2.0) = {theta_max:.4f}°"

    def test_increases_with_M(self):
        """最大转折角随马赫数先增后趋于极限，M=3>M=2 时应更大"""
        assert max_turning_angle(3.0) > max_turning_angle(2.0)

    def test_positive(self):
        """最大转折角为正"""
        assert max_turning_angle(2.0) > 0
