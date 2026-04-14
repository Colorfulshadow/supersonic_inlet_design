"""
tests/test_prandtl_meyer.py
============================
Prandtl-Meyer 函数验证（core/prandtl_meyer.py）。

验证基准（γ=1.4）
-----------------
- ν(1.0) = 0.0 rad
- ν(2.0) ≈ 0.4606 rad（26.38°）
- ν(1.4) ≈ 0.1569 rad（8.99°）
- M_from_pm_angle(ν(2.0)) ≈ 2.0（往返精度 < 1e-6）
- M_start=2.0, delta_theta=5° → M_final ≈ 1.82（NOT 1.75，经 PM 公式验证）
"""

import math

import numpy as np
import pytest

from core.prandtl_meyer import (
    M_from_pm_angle,
    isentropic_ramp_coords,
    prandtl_meyer_angle,
)


# ---------------------------------------------------------------------------
# prandtl_meyer_angle
# ---------------------------------------------------------------------------

class TestPrandtlMeyerAngle:
    """ν(M) 函数单元测试。"""

    def test_sonic_returns_zero(self):
        """M=1.0 时 ν=0。"""
        assert prandtl_meyer_angle(1.0) == pytest.approx(0.0)

    def test_subsonic_returns_zero(self):
        """M<1 时 ν=0（无 PM 膨胀）。"""
        assert prandtl_meyer_angle(0.5) == pytest.approx(0.0)
        assert prandtl_meyer_angle(0.0) == pytest.approx(0.0)

    def test_M2_benchmark(self):
        """ν(2.0) ≈ 26.38°（0.4606 rad），γ=1.4。"""
        nu = prandtl_meyer_angle(2.0, gamma=1.4)
        assert math.degrees(nu) == pytest.approx(26.38, abs=0.02)

    def test_M14_benchmark(self):
        """ν(1.4) ≈ 8.99°（0.1569 rad），γ=1.4。"""
        nu = prandtl_meyer_angle(1.4, gamma=1.4)
        assert math.degrees(nu) == pytest.approx(8.99, abs=0.02)

    def test_monotone_increasing(self):
        """ν(M) 关于 M>1 单调递增。"""
        Ms = [1.1, 1.5, 2.0, 3.0, 5.0]
        nus = [prandtl_meyer_angle(M) for M in Ms]
        for i in range(len(nus) - 1):
            assert nus[i] < nus[i + 1], f"ν(M={Ms[i]}) ≥ ν(M={Ms[i+1]})"

    def test_gamma_sensitivity(self):
        """改变 γ 对结果有影响（非 1.4 时结果应不同）。"""
        nu14 = prandtl_meyer_angle(2.0, gamma=1.4)
        nu13 = prandtl_meyer_angle(2.0, gamma=1.3)
        assert abs(nu14 - nu13) > 0.01

    def test_returns_float(self):
        """返回值应为 float。"""
        assert isinstance(prandtl_meyer_angle(2.0), float)


# ---------------------------------------------------------------------------
# M_from_pm_angle
# ---------------------------------------------------------------------------

class TestMFromPmAngle:
    """M_from_pm_angle 反算函数单元测试。"""

    def test_round_trip_M2(self):
        """往返精度：M_from_pm_angle(ν(2.0)) ≈ 2.0，误差 < 1e-6。"""
        nu = prandtl_meyer_angle(2.0)
        M_recovered = M_from_pm_angle(nu)
        assert abs(M_recovered - 2.0) < 1e-6

    def test_round_trip_M15(self):
        """往返精度：M_from_pm_angle(ν(1.5)) ≈ 1.5。"""
        nu = prandtl_meyer_angle(1.5)
        M_recovered = M_from_pm_angle(nu)
        assert abs(M_recovered - 1.5) < 1e-6

    def test_round_trip_M4(self):
        """往返精度：M_from_pm_angle(ν(4.0)) ≈ 4.0。"""
        nu = prandtl_meyer_angle(4.0)
        M_recovered = M_from_pm_angle(nu)
        assert abs(M_recovered - 4.0) < 1e-5

    def test_nu_zero_returns_one(self):
        """ν=0 时 M=1.0。"""
        M = M_from_pm_angle(0.0)
        assert M == pytest.approx(1.0, abs=1e-6)

    def test_negative_nu_raises(self):
        """ν<0 抛出 ValueError。"""
        with pytest.raises(ValueError, match="不能为负"):
            M_from_pm_angle(-0.1)

    def test_exceeds_max_raises(self):
        """ν > ν_max 抛出 ValueError。"""
        nu_max = (math.pi / 2) * (math.sqrt(2.4 / 0.4) - 1)
        with pytest.raises(ValueError):
            M_from_pm_angle(nu_max + 0.1)

    def test_returns_float(self):
        """返回 float。"""
        nu = prandtl_meyer_angle(2.0)
        assert isinstance(M_from_pm_angle(nu), float)


# ---------------------------------------------------------------------------
# isentropic_ramp_coords
# ---------------------------------------------------------------------------

class TestIsotropicRampCoords:
    """isentropic_ramp_coords 函数单元测试。"""

    def test_output_shapes(self):
        """xs, ys 形状均为 (n_steps+1,)。"""
        xs, ys, _, _ = isentropic_ramp_coords(2.0, 0.0, 5.0, n_steps=50)
        assert xs.shape == (51,)
        assert ys.shape == (51,)

    def test_origin_at_zero(self):
        """起始点 (xs[0], ys[0]) = (0, 0)。"""
        xs, ys, _, _ = isentropic_ramp_coords(2.0, 0.0, 5.0, n_steps=20)
        assert xs[0] == pytest.approx(0.0)
        assert ys[0] == pytest.approx(0.0)

    def test_M_final_correct_M2_5deg(self):
        """M_start=2.0, delta_theta=5° → M_final ≈ 1.82（不是 1.75）。

        物理验证：ν(2.0)≈26.38°，压缩 5° 后 ν=21.38°，对应 M≈1.82。
        """
        xs, ys, M_final, theta_final = isentropic_ramp_coords(2.0, 0.0, 5.0, n_steps=200)
        # 用 PM 公式直接验证期望值
        nu_start = prandtl_meyer_angle(2.0)
        nu_end   = nu_start - math.radians(5.0)
        M_expected = M_from_pm_angle(nu_end)
        assert abs(M_final - M_expected) < 0.005, (
            f"M_final={M_final:.4f}，期望≈{M_expected:.4f}"
        )
        # 确认 M_final ≈ 1.82（而非 1.75）
        assert 1.80 < M_final < 1.85, (
            f"M_final={M_final:.4f}，期望在 (1.80, 1.85) 内"
        )

    def test_theta_final_deg(self):
        """theta_final_deg = theta_start_deg + delta_theta_deg。"""
        _, _, _, theta_final = isentropic_ramp_coords(2.0, 10.0, 5.0, n_steps=50)
        assert theta_final == pytest.approx(15.0, abs=1e-10)

    def test_M_final_less_than_M_start(self):
        """等熵压缩后 M_final < M_start。"""
        _, _, M_final, _ = isentropic_ramp_coords(2.0, 0.0, 5.0)
        assert M_final < 2.0

    def test_ys_monotone_increasing_with_inclination(self):
        """theta_start_deg > 0 时 ys 单调递增。"""
        _, ys, _, _ = isentropic_ramp_coords(2.0, 5.0, 3.0, n_steps=50)
        for i in range(len(ys) - 1):
            assert ys[i + 1] >= ys[i], f"ys[{i+1}]={ys[i+1]:.4f} < ys[{i}]={ys[i]:.4f}"

    def test_xs_monotone_increasing(self):
        """xs 单调递增（theta < 90° 时）。"""
        xs, _, _, _ = isentropic_ramp_coords(2.0, 5.0, 3.0, n_steps=50)
        for i in range(len(xs) - 1):
            assert xs[i + 1] > xs[i]

    def test_arc_length_per_step(self):
        """每步归一化弧长 = 1（ds=1 per step）。"""
        xs, ys, _, _ = isentropic_ramp_coords(2.0, 10.0, 3.0, n_steps=10)
        for i in range(len(xs) - 1):
            ds = math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
            assert abs(ds - 1.0) < 1e-10, f"Step {i}: ds={ds:.6f}，期望 1.0"

    def test_different_n_steps_same_M_final(self):
        """不同 n_steps 应收敛到相同 M_final（步数越多越精确）。"""
        _, _, M_50, _  = isentropic_ramp_coords(2.0, 0.0, 5.0, n_steps=50)
        _, _, M_200, _ = isentropic_ramp_coords(2.0, 0.0, 5.0, n_steps=200)
        # 收敛性：两者误差 < 0.005
        assert abs(M_50 - M_200) < 0.005

    def test_invalid_M_start_raises(self):
        """M_start ≤ 1 抛出 ValueError。"""
        with pytest.raises(ValueError, match="M_start"):
            isentropic_ramp_coords(1.0, 0.0, 5.0)
        with pytest.raises(ValueError, match="M_start"):
            isentropic_ramp_coords(0.8, 0.0, 5.0)

    def test_invalid_delta_theta_raises(self):
        """delta_theta_deg ≤ 0 抛出 ValueError。"""
        with pytest.raises(ValueError, match="delta_theta_deg"):
            isentropic_ramp_coords(2.0, 0.0, 0.0)
        with pytest.raises(ValueError, match="delta_theta_deg"):
            isentropic_ramp_coords(2.0, 0.0, -1.0)

    def test_invalid_n_steps_raises(self):
        """n_steps < 1 抛出 ValueError。"""
        with pytest.raises(ValueError, match="n_steps"):
            isentropic_ramp_coords(2.0, 0.0, 5.0, n_steps=0)

    def test_gamma_propagation(self):
        """非标准 γ 应改变 M_final。"""
        _, _, M_14, _ = isentropic_ramp_coords(2.0, 0.0, 5.0, gamma=1.4)
        _, _, M_13, _ = isentropic_ramp_coords(2.0, 0.0, 5.0, gamma=1.3)
        assert abs(M_14 - M_13) > 0.01
