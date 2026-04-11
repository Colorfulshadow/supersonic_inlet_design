"""
tests/test_atmosphere.py
========================
验证 core/atmosphere.py 的 ISA 大气模型和捕获面积计算。

验证基准来源：CLAUDE.md §4.7 / §5.4，ICAO 标准大气。
"""

import math
import pytest

from core.atmosphere import isa_atmosphere, capture_area


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
