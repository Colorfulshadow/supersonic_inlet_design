"""
tests/test_pitot.py
===================
皮托管进气道气动设计验证，对应 CLAUDE.md §5.3 和 §四 数值基准。
"""

import pytest
from inlets.pitot.aero_design import design_pitot
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
