"""
tests/test_taylor_maccoll.py
============================
pytest 测试：core/taylor_maccoll.py 数值验证。

参考基准
--------
与 NACA TN 1135 (Table B.1, M₀=2.0) 校对：
    δ_c = 20°  → β_c ≈ 38.2°,  M_cone ≈ 1.57
    δ_c = 25°  → β_c ≈ 42.5°,  M_cone ≈ 1.42

CLAUDE.md §4.4 对 δ_c=31.37° 给出的估算（β_c=50.79°, M_cone=1.55）
标注为"待 T-M 求解器验证"，实际 T-M 计算结果为：
    δ_c = 31.37° → β_c ≈ 49.77°, M_cone ≈ 1.21（与 NACA 趋势一致）

运行方式：
    venv\\Scripts\\python.exe -m pytest tests/test_taylor_maccoll.py -v
"""

import math
import pytest
from core.taylor_maccoll import (
    cone_shock_pt_ratio,
    solve_taylor_maccoll,
    taylor_maccoll_ode,
)


# ---------------------------------------------------------------------------
# ODE 右端项基本检查
# ---------------------------------------------------------------------------

class TestTaylorMaccollODE:
    def test_returns_two_values(self):
        """ODE 返回两个元素的列表/向量。"""
        result = taylor_maccoll_ode(math.radians(45), [0.5, -0.2])
        assert len(result) == 2

    def test_dVr_equals_Vtheta(self):
        """无旋条件：dV_r/dθ = V_θ。"""
        V_r, V_theta = 0.5, -0.25
        theta = math.radians(40)
        derivs = taylor_maccoll_ode(theta, [V_r, V_theta])
        assert abs(derivs[0] - V_theta) < 1e-12

    def test_cone_surface_degenerate(self):
        """锥面处 V_θ=0 时，dV_r/dθ = 0。"""
        V_r, V_theta = 0.47, 0.0
        theta = math.radians(31)
        derivs = taylor_maccoll_ode(theta, [V_r, V_theta])
        assert abs(derivs[0]) < 1e-12   # dV_r/dθ = V_theta = 0

    def test_subsonic_a_sq_positive(self):
        """速度幅值 < 1 时声速平方 > 0（物理约束）。"""
        V_r, V_theta = 0.4, -0.25   # V² = 0.16+0.0625 = 0.2225 < 1
        theta = math.radians(45)
        derivs = taylor_maccoll_ode(theta, [V_r, V_theta])
        # 只要不抛异常且数值有界，即为通过
        assert all(math.isfinite(d) for d in derivs)


# ---------------------------------------------------------------------------
# 物理极限验证
# ---------------------------------------------------------------------------

class TestPhysicalLimits:
    """验证 T-M 解在极限情形下的正确行为。"""

    def test_thin_cone_beta_near_mach_angle(self):
        """极薄锥（δ_c=2°）：β_c 应紧贴 Mach 角（30°）。"""
        result = solve_taylor_maccoll(2.0, 2.0)
        mu_deg = math.degrees(math.asin(1.0 / 2.0))   # 30°
        assert abs(result["beta_c_deg"] - mu_deg) < 0.1

    def test_thin_cone_M_cone_near_M0(self):
        """极薄锥（δ_c=2°）：锥面马赫数应接近来流 M₀=2.0。"""
        result = solve_taylor_maccoll(2.0, 2.0)
        assert abs(result["M_cone"] - 2.0) < 0.02

    def test_M_cone_decreases_with_delta_c(self):
        """锥面马赫数随锥半角单调递减（锥越粗，压缩越强）。"""
        results = [solve_taylor_maccoll(dc, 2.0) for dc in [5, 10, 15, 20, 25, 30]]
        M_cones = [r["M_cone"] for r in results]
        for i in range(len(M_cones) - 1):
            assert M_cones[i] > M_cones[i + 1]

    def test_beta_c_increases_with_delta_c(self):
        """激波角随锥半角单调递增。"""
        results = [solve_taylor_maccoll(dc, 2.0) for dc in [5, 10, 15, 20, 25, 30]]
        betas = [r["beta_c_deg"] for r in results]
        for i in range(len(betas) - 1):
            assert betas[i] < betas[i + 1]

    def test_beta_c_greater_than_delta_c(self):
        """激波角始终大于锥半角（激波在锥面外侧）。"""
        for dc in [5, 10, 20, 30, 31.37]:
            r = solve_taylor_maccoll(dc, 2.0)
            assert r["beta_c_deg"] > dc

    def test_M_cone_supersonic(self):
        """锥面流场为超声速（设计工况 M₀=2.0 下 M_cone > 1）。"""
        for dc in [5, 10, 20, 30, 31.37]:
            r = solve_taylor_maccoll(dc, 2.0)
            assert r["M_cone"] > 1.0


# ---------------------------------------------------------------------------
# NACA TN 1135 参考值校验（M₀=2.0）
# ---------------------------------------------------------------------------

class TestNACABenchmarks:
    """与 NACA TN 1135 Table B.1 对比，容差 ±0.5°（角度）/ ±0.05（马赫数）。"""

    def test_delta20_beta(self):
        """δ_c=20° → β_c ≈ 38.2°（NACA 1135）。"""
        r = solve_taylor_maccoll(20.0, 2.0)
        assert abs(r["beta_c_deg"] - 38.2) < 0.5

    def test_delta20_M_cone(self):
        """δ_c=20° → M_cone ≈ 1.57（NACA 1135）。"""
        r = solve_taylor_maccoll(20.0, 2.0)
        assert abs(r["M_cone"] - 1.57) < 0.05

    def test_delta25_beta(self):
        """δ_c=25° → β_c ≈ 42.5°（NACA 1135）。"""
        r = solve_taylor_maccoll(25.0, 2.0)
        assert abs(r["beta_c_deg"] - 42.5) < 0.5

    def test_delta25_M_cone(self):
        """δ_c=25° → M_cone ≈ 1.42（NACA 1135）。"""
        r = solve_taylor_maccoll(25.0, 2.0)
        assert abs(r["M_cone"] - 1.42) < 0.05


# ---------------------------------------------------------------------------
# 设计工况（δ_c=31.37°，M₀=2.0）— 主基准测试
# ---------------------------------------------------------------------------

class TestDesignCase:
    """
    设计工况 M₀=2.0，δ_c=31.37°（轴对称进气道设计点）。

    注意：CLAUDE.md §4.4 给出的估算值（β_c=50.79°, M_cone=1.55）标注
    "待 T-M 求解器验证"，实际 T-M 计算值（与 NACA 1135 趋势一致）为：
        β_c  ≈ 49.77°  （本测试容差 ±0.5°）
        M_cone ≈ 1.21  （本测试容差 ±0.05）
    """

    @pytest.fixture(scope="class")
    def result(self):
        return solve_taylor_maccoll(31.37, 2.0)

    def test_beta_c_value(self, result):
        """β_c ≈ 49.77° ±0.5°（T-M 计算值，与 NACA 1135 趋势一致）。"""
        assert abs(result["beta_c_deg"] - 49.77) < 0.5, (
            f"β_c = {result['beta_c_deg']:.4f}°（期望 49.77° ±0.5°）"
        )

    def test_M_cone_value(self, result):
        """M_cone ≈ 1.21 ±0.05（T-M 计算值）。"""
        assert abs(result["M_cone"] - 1.21) < 0.05, (
            f"M_cone = {result['M_cone']:.4f}（期望 1.21 ±0.05）"
        )

    def test_V_theta_cone_near_zero(self, result):
        """锥面无穿透条件：|V_θ(δ_c)| < 1e-4。"""
        assert abs(result["V_theta_cone"]) < 1e-4, (
            f"|V_theta_cone| = {abs(result['V_theta_cone']):.2e}"
        )

    def test_result_keys(self, result):
        """返回字典包含所有必要字段。"""
        assert "beta_c_deg" in result
        assert "M_cone" in result
        assert "V_r_cone" in result
        assert "V_theta_cone" in result

    def test_V_r_cone_positive(self, result):
        """锥面径向速度为正（流体向外流动）。"""
        assert result["V_r_cone"] > 0

    def test_beta_c_greater_than_delta_c(self, result):
        """激波角 > 锥半角。"""
        assert result["beta_c_deg"] > 31.37


# ---------------------------------------------------------------------------
# cone_shock_pt_ratio
# ---------------------------------------------------------------------------

class TestConeShockPtRatio:
    def test_weak_cone_high_sigma(self):
        """小锥角（δ_c=10°）的锥形激波总压比应接近 1（σ > 0.998）。"""
        r = solve_taylor_maccoll(10.0, 2.0)
        beta_c_rad = math.radians(r["beta_c_deg"])
        sigma = cone_shock_pt_ratio(2.0, beta_c_rad)
        assert sigma > 0.998, f"σ={sigma:.6f}（期望 > 0.998）"

    def test_moderate_cone_sigma(self):
        """δ_c=20° 的锥形激波总压比 > 0.98（量级合理性检查）。"""
        r = solve_taylor_maccoll(20.0, 2.0)
        beta_c_rad = math.radians(r["beta_c_deg"])
        sigma = cone_shock_pt_ratio(2.0, beta_c_rad)
        assert sigma > 0.98, f"σ={sigma:.6f}（期望 > 0.98）"

    def test_sigma_less_than_one(self):
        """总压比不超过 1（熵增原理）。"""
        r = solve_taylor_maccoll(31.37, 2.0)
        beta_c_rad = math.radians(r["beta_c_deg"])
        sigma = cone_shock_pt_ratio(2.0, beta_c_rad)
        assert sigma < 1.0

    def test_sigma_decreases_with_cone_angle(self):
        """更大的锥角带来更强的激波和更大的总压损失（σ 递减）。"""
        sigmas = []
        for dc in [10, 20, 30]:
            r = solve_taylor_maccoll(dc, 2.0)
            beta_rad = math.radians(r["beta_c_deg"])
            sigmas.append(cone_shock_pt_ratio(2.0, beta_rad))
        assert sigmas[0] > sigmas[1] > sigmas[2]

    def test_design_case_sigma_physical(self):
        """δ_c=31.37° 设计工况：总压比在物理合理范围（0.85 < σ < 0.95）。"""
        r = solve_taylor_maccoll(31.37, 2.0)
        beta_c_rad = math.radians(r["beta_c_deg"])
        sigma = cone_shock_pt_ratio(2.0, beta_c_rad)
        assert 0.85 < sigma < 0.95, f"σ={sigma:.6f}（期望 0.85~0.95）"

    def test_uses_shock_pt_ratio(self):
        """cone_shock_pt_ratio 等价于 shock_pt_ratio(M0*sin(beta_c))。"""
        from core.compressible_flow import shock_pt_ratio
        beta_c_rad = math.radians(45.0)
        M0 = 2.0
        M_n = M0 * math.sin(beta_c_rad)
        expected = shock_pt_ratio(M_n)
        result = cone_shock_pt_ratio(M0, beta_c_rad)
        assert abs(result - expected) < 1e-12
