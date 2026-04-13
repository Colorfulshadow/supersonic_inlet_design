"""
tests/test_atmosphere.py
========================
验证 core/atmosphere.py 的 ISA 大气模型和捕获面积计算。

验证基准来源：CLAUDE.md §4.7 / §5.4，ICAO 标准大气。
"""

import math
import pytest

from core.atmosphere import (
    ISAAtmosphere,
    capture_area,
    capture_area_from_total,
    isa_atmosphere,
)


# ---------------------------------------------------------------------------
# 海平面（H = 0 m）标准值
# ---------------------------------------------------------------------------

class TestSeaLevel:
    def test_temperature(self):
        T, *_ = isa_atmosphere(0.0)
        assert abs(T - 288.15) < 0.01, f"海平面温度 {T} K，期望 288.15 K"

    def test_pressure(self):
        _, p, *_ = isa_atmosphere(0.0)
        assert abs(p - 101325.0) < 1.0, f"海平面压力 {p} Pa，期望 101325 Pa"

    def test_density(self):
        _, _, rho, _ = isa_atmosphere(0.0)
        # ρ = 101325 / (287.05 × 288.15) = 1.2250 kg/m³（ICAO 标准值）
        assert abs(rho - 1.225) < 0.001, f"海平面密度 {rho} kg/m³，期望 ≈ 1.225"

    def test_speed_of_sound(self):
        *_, a = isa_atmosphere(0.0)
        # a = sqrt(1.4 × 287.05 × 288.15) ≈ 340.29 m/s
        assert abs(a - 340.29) < 0.05, f"海平面声速 {a} m/s，期望 ≈ 340.29 m/s"


# ---------------------------------------------------------------------------
# 对流层顶（H = 11 000 m）
# ---------------------------------------------------------------------------

class TestTropopause:
    def test_temperature(self):
        T, *_ = isa_atmosphere(11000.0)
        assert abs(T - 216.65) < 0.01, f"对流层顶温度 {T} K，期望 216.65 K"

    def test_pressure(self):
        _, p, *_ = isa_atmosphere(11000.0)
        # ICAO 标准值 22632.1 Pa，允许误差 ±10 Pa（对流层公式累积误差）
        assert abs(p - 22632.1) < 10.0, f"对流层顶压力 {p} Pa，期望 ≈ 22632 Pa"

    def test_temperature_continuity(self):
        """H=11000 m 左右两侧温度连续（≤ 0.01 K 差异）。"""
        eps = 1.0   # 1 m 偏移，足够检验连续性
        T_below, *_ = isa_atmosphere(11000.0 - eps)
        T_above, *_ = isa_atmosphere(11000.0 + eps)
        assert abs(T_below - T_above) < 0.1, (
            f"对流层顶温度不连续：下方 {T_below:.4f} K，上方 {T_above:.4f} K"
        )


# ---------------------------------------------------------------------------
# 平流层（H = 20 000 m）——设计工况，§5.4 验证基准
# ---------------------------------------------------------------------------

class TestStratosphere20km:
    H = 20000.0

    def test_temperature(self):
        T, *_ = isa_atmosphere(self.H)
        assert abs(T - 216.65) < 0.01, f"20 km 温度 {T} K，期望 216.65 K"

    def test_pressure(self):
        _, p, *_ = isa_atmosphere(self.H)
        assert abs(p - 5474.9) < 1.0, f"20 km 压力 {p:.2f} Pa，期望 5474.9 Pa（±1 Pa）"

    def test_density(self):
        _, _, rho, _ = isa_atmosphere(self.H)
        assert abs(rho - 0.08803) < 0.00005, (
            f"20 km 密度 {rho:.6f} kg/m³，期望 0.08803（±0.00005）"
        )

    def test_speed_of_sound(self):
        *_, a = isa_atmosphere(self.H)
        assert abs(a - 295.07) < 0.05, f"20 km 声速 {a:.4f} m/s，期望 295.07（±0.05）"


# ---------------------------------------------------------------------------
# capture_area 功能验证
# ---------------------------------------------------------------------------

class TestCaptureArea:
    def test_design_point(self):
        """M₀=2.0，H=20 km，ṁ=100 kg/s → A_cap ≈ 1.924 m²（±0.005 m²）。"""
        A = capture_area(100.0, 2.0, 20000.0)
        assert abs(A - 1.924) < 0.005, f"设计工况 A_cap = {A:.4f} m²，期望 1.924 m²"

    def test_decreases_with_mach(self):
        """同高度同质量流量下，M₀ 增大则 A_cap 减小（物理一致性）。"""
        A1 = capture_area(100.0, 1.5, 20000.0)
        A2 = capture_area(100.0, 2.0, 20000.0)
        A3 = capture_area(100.0, 2.5, 20000.0)
        assert A1 > A2 > A3, (
            f"A_cap 应随 M₀ 增大而减小，实际：M=1.5→{A1:.4f}, M=2.0→{A2:.4f}, M=2.5→{A3:.4f}"
        )

    def test_proportional_to_mdot(self):
        """A_cap 正比于质量流量（线性关系）。"""
        A50 = capture_area(50.0, 2.0, 20000.0)
        A100 = capture_area(100.0, 2.0, 20000.0)
        ratio = A100 / A50
        assert abs(ratio - 2.0) < 1e-9, f"A_cap 应与 ṁ 成正比，比值 {ratio:.10f}，期望 2.0"

    def test_positive(self):
        """A_cap 始终为正值。"""
        A = capture_area(100.0, 2.0, 20000.0)
        assert A > 0.0


# ---------------------------------------------------------------------------
# 附加物理一致性检验
# ---------------------------------------------------------------------------

class TestPhysicalConsistency:
    def test_temperature_decreases_in_troposphere(self):
        """对流层内温度随高度单调递减。"""
        heights = [0, 2000, 5000, 8000, 11000]
        temps = [isa_atmosphere(h)[0] for h in heights]
        for i in range(len(temps) - 1):
            assert temps[i] > temps[i + 1], (
                f"对流层温度应随高度减小：T({heights[i]})={temps[i]:.2f} K，"
                f"T({heights[i+1]})={temps[i+1]:.2f} K"
            )

    def test_temperature_constant_in_stratosphere(self):
        """平流层（11000–20000 m）温度恒为 216.65 K。"""
        for H in [12000, 15000, 18000, 20000]:
            T, *_ = isa_atmosphere(float(H))
            assert abs(T - 216.65) < 0.01, f"平流层 H={H} m 温度 {T} K，期望 216.65 K"

    def test_pressure_decreases_monotonically(self):
        """压力随高度单调递减（对流层 + 平流层）。"""
        heights = [0, 5000, 10000, 11000, 15000, 20000]
        pressures = [isa_atmosphere(float(h))[1] for h in heights]
        for i in range(len(pressures) - 1):
            assert pressures[i] > pressures[i + 1], (
                f"压力应随高度减小：p({heights[i]})={pressures[i]:.1f} Pa，"
                f"p({heights[i+1]})={pressures[i+1]:.1f} Pa"
            )

    def test_ideal_gas_law(self):
        """各高度均满足理想气体方程 ρ = p/(R·T)。"""
        R = 287.05
        for H in [0, 5000, 11000, 15000, 20000]:
            T, p, rho, _ = isa_atmosphere(float(H))
            rho_ideal = p / (R * T)
            assert abs(rho - rho_ideal) < 1e-9, (
                f"H={H} m 理想气体方程不满足：ρ={rho}, p/(RT)={rho_ideal}"
            )


# ---------------------------------------------------------------------------
# ISAAtmosphere 类测试
# ---------------------------------------------------------------------------

class TestISAAtmosphere:
    """验证 ISAAtmosphere 面向对象封装的正确性。"""

    def test_construction_20km(self):
        """H=20 km 时静态参数与函数式 API 一致。"""
        atm = ISAAtmosphere(20000.0)
        T, p, rho, a = isa_atmosphere(20000.0)
        assert atm.T_static == T
        assert atm.p_static == p
        assert atm.rho == rho
        assert atm.a == a

    def test_total_temperature_M2(self):
        """M=2.0, H=20 km：T0 = T*(1+0.2*4) = 216.65*1.8 = 389.97 K。"""
        atm = ISAAtmosphere(20000.0)
        T0 = atm.total_temperature(2.0)
        expected = 216.65 * 1.8
        assert abs(T0 - expected) < 0.05, f"T0={T0:.4f} K，期望 {expected:.4f} K"

    def test_total_pressure_ratio_M2(self):
        """M=2.0 时总压与静压之比 P0/p = (1+0.2*4)^3.5 ≈ 7.824。"""
        atm = ISAAtmosphere(20000.0)
        P0 = atm.total_pressure(2.0)
        ratio = P0 / atm.p_static
        expected = (1.0 + 0.2 * 4.0) ** 3.5   # = 1.8^3.5
        assert abs(ratio - expected) < 1e-6, (
            f"P0/p = {ratio:.6f}，期望 {expected:.6f}"
        )

    def test_total_pressure_ratio_numeric(self):
        """(1+0.2*4)^3.5 ≈ 7.824 的数值验证（CLAUDE.md 测试要求）。"""
        ratio = (1.0 + 0.2 * 4.0) ** 3.5
        assert abs(ratio - 7.824) < 0.005, f"比值 {ratio:.4f}，期望 ≈ 7.824"

    def test_velocity_M2(self):
        """速度 V = M * a，M=2.0, H=20 km → V ≈ 590.14 m/s。"""
        atm = ISAAtmosphere(20000.0)
        v = atm.velocity(2.0)
        assert abs(v - 590.14) < 0.5, f"速度 {v:.2f} m/s，期望 ≈ 590.14 m/s"

    def test_capture_area_design_point(self):
        """设计工况 A_cap ≈ 1.924 m²（与函数式 API 一致）。"""
        atm = ISAAtmosphere(20000.0)
        A = atm.capture_area(100.0, 2.0)
        assert abs(A - 1.924) < 0.005, f"A_cap = {A:.4f} m²，期望 1.924 m²"

    def test_capture_area_negative_mdot_raises(self):
        """m_dot ≤ 0 时应抛出 ValueError。"""
        atm = ISAAtmosphere(20000.0)
        with pytest.raises(ValueError):
            atm.capture_area(0.0, 2.0)
        with pytest.raises(ValueError):
            atm.capture_area(-10.0, 2.0)

    def test_repr_contains_height(self):
        """repr 字符串包含高度信息。"""
        atm = ISAAtmosphere(20000.0)
        assert "20000" in repr(atm)

    def test_gamma_propagated(self):
        """非标准 gamma 能被正确传入并影响计算。"""
        atm_14 = ISAAtmosphere(0.0, gamma=1.4)
        atm_16 = ISAAtmosphere(0.0, gamma=1.6)
        # 声速与 gamma 相关
        assert atm_14.a != atm_16.a


# ---------------------------------------------------------------------------
# capture_area_from_total 测试
# ---------------------------------------------------------------------------

class TestCaptureAreaFromTotal:
    """验证总条件捕获面积公式与静条件公式的等价性及边界行为。"""

    def test_equivalent_to_static_formula(self):
        """总条件与静条件公式结果应在数值误差内一致。"""
        H = 20000.0
        M0 = 2.0
        mdot = 100.0
        gamma = 1.4

        atm = ISAAtmosphere(H, gamma)
        A_static = capture_area(mdot, M0, H, gamma)

        T0 = atm.total_temperature(M0)
        P0 = atm.total_pressure(M0)
        A_total = capture_area_from_total(mdot, M0, T0, P0, gamma)

        assert abs(A_static - A_total) / A_static < 1e-9, (
            f"静条件公式 {A_static:.6f} m² vs 总条件公式 {A_total:.6f} m²，两者应等价"
        )

    def test_design_point_value(self):
        """M₀=2.0，H=20 km，ṁ=100 kg/s → A_cap ≈ 1.924 m²（±0.005 m²）。"""
        atm = ISAAtmosphere(20000.0)
        T0 = atm.total_temperature(2.0)
        P0 = atm.total_pressure(2.0)
        A = capture_area_from_total(100.0, 2.0, T0, P0)
        assert abs(A - 1.924) < 0.005, f"A_cap = {A:.4f} m²，期望 1.924 m²"

    def test_mdot_zero_raises(self):
        with pytest.raises(ValueError):
            capture_area_from_total(0.0, 2.0, 400.0, 50000.0)

    def test_mdot_negative_raises(self):
        with pytest.raises(ValueError):
            capture_area_from_total(-1.0, 2.0, 400.0, 50000.0)

    def test_M_zero_raises(self):
        with pytest.raises(ValueError):
            capture_area_from_total(100.0, 0.0, 400.0, 50000.0)

    def test_T0_zero_raises(self):
        with pytest.raises(ValueError):
            capture_area_from_total(100.0, 2.0, 0.0, 50000.0)

    def test_P0_zero_raises(self):
        with pytest.raises(ValueError):
            capture_area_from_total(100.0, 2.0, 400.0, 0.0)

    def test_proportional_to_mdot(self):
        """A_cap 正比于 ṁ。"""
        atm = ISAAtmosphere(20000.0)
        T0 = atm.total_temperature(2.0)
        P0 = atm.total_pressure(2.0)
        A50  = capture_area_from_total(50.0,  2.0, T0, P0)
        A100 = capture_area_from_total(100.0, 2.0, T0, P0)
        assert abs(A100 / A50 - 2.0) < 1e-9
