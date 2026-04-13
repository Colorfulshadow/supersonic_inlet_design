"""
tests/test_flow_stations.py
============================
pytest 测试：core/flow_stations.py

运行方式：
    venv\\Scripts\\python.exe -m pytest tests/test_flow_stations.py -v
"""

import pytest
from core.atmosphere import ISAAtmosphere
from core.flow_stations import FlowState, InletFlowStations


# ---------------------------------------------------------------------------
# 辅助：构造一组完整的 8 站位对象（皮托管式，M₀=2.0 示意数据）
# ---------------------------------------------------------------------------

def _make_full_stations() -> InletFlowStations:
    """构造完整 8 站位对象（数值仅用于测试，不代表精确气动计算）。

    取 p_t0 = 101325 Pa（标准大气压，代替真实高空总压），
    用逐级总压比模拟各段损失。
    """
    p_t0 = 101325.0
    T_t0 = 300.0   # 简化：各站总温相同（绝热进气道）

    # 各段总压比（均 ≤ 1）
    sigma_L0  = 1.000   # 前缘无损失
    sigma_EXL = 0.980   # 外部斜激波
    sigma_1EX = 0.990   # 唇口区
    sigma_SD1 = 1.000   # 亚声速无损失
    sigma_2SD = 1.000   # 扩压段无损失
    # 终端正激波
    sigma_NS  = 0.721   # M=2.0 正激波总压比

    p_tL  = p_t0  * sigma_L0
    p_tEX = p_tL  * sigma_EXL
    p_t1  = p_tEX * sigma_1EX
    p_tNS = p_t1  * sigma_NS
    p_tTH = p_tNS
    p_tSD = p_tNS
    p_t2  = p_tSD * sigma_2SD

    return InletFlowStations(
        st0  = FlowState(M=2.0,  p_t=p_t0,  T_t=T_t0,  label="0"),
        stL  = FlowState(M=2.0,  p_t=p_tL,  T_t=T_t0,  label="L"),
        stEX = FlowState(M=1.8,  p_t=p_tEX, T_t=T_t0,  label="EX"),
        stNS = FlowState(M=0.60, p_t=p_tNS, T_t=T_t0,  label="NS"),
        st1  = FlowState(M=1.4,  p_t=p_t1,  T_t=T_t0,  label="1"),
        stTH = FlowState(M=1.0,  p_t=p_tTH, T_t=T_t0,  label="TH"),
        stSD = FlowState(M=0.55, p_t=p_tSD, T_t=T_t0,  label="SD"),
        st2  = FlowState(M=0.40, p_t=p_t2,  T_t=T_t0,  label="2"),
    )


# ---------------------------------------------------------------------------
# FlowState 构造与验证
# ---------------------------------------------------------------------------

class TestFlowState:
    def test_basic_construction(self):
        st = FlowState(M=2.0, p_t=50000.0, T_t=250.0)
        assert st.M == 2.0
        assert st.p_t == 50000.0
        assert st.T_t == 250.0

    def test_optional_fields_default_none(self):
        st = FlowState(M=1.5, p_t=80000.0, T_t=280.0)
        assert st.p is None
        assert st.T is None
        assert st.label == ""

    def test_with_static_values(self):
        st = FlowState(M=1.0, p_t=100000.0, T_t=300.0, p=52828.0, T=250.0, label="TH")
        assert st.p == 52828.0
        assert st.T == 250.0
        assert st.label == "TH"

    def test_negative_M_raises(self):
        with pytest.raises(ValueError, match="马赫数"):
            FlowState(M=-1.0, p_t=100000.0, T_t=300.0)

    def test_zero_p_t_raises(self):
        with pytest.raises(ValueError, match="总压"):
            FlowState(M=1.0, p_t=0.0, T_t=300.0)

    def test_negative_p_t_raises(self):
        with pytest.raises(ValueError, match="总压"):
            FlowState(M=1.0, p_t=-100.0, T_t=300.0)

    def test_zero_T_t_raises(self):
        with pytest.raises(ValueError, match="总温"):
            FlowState(M=1.0, p_t=100000.0, T_t=0.0)

    def test_M_zero_allowed(self):
        """M=0（滞止状态）是物理合法的。"""
        st = FlowState(M=0.0, p_t=101325.0, T_t=288.15)
        assert st.M == 0.0


# ---------------------------------------------------------------------------
# InletFlowStations 构造
# ---------------------------------------------------------------------------

class TestInletFlowStationsConstruction:
    def test_empty_construction(self):
        """全为 None 的空对象应正常构造。"""
        ifs = InletFlowStations()
        assert ifs.st0 is None
        assert ifs.st2 is None

    def test_partial_construction(self):
        st = FlowState(M=2.0, p_t=101325.0, T_t=300.0)
        ifs = InletFlowStations(st0=st)
        assert ifs.st0 is st
        assert ifs.stEX is None

    def test_full_construction(self):
        ifs = _make_full_stations()
        assert ifs.st0 is not None
        assert ifs.st2 is not None
        for attr in ["st0", "stL", "stEX", "stNS", "st1", "stTH", "stSD", "st2"]:
            assert getattr(ifs, attr) is not None


# ---------------------------------------------------------------------------
# total_pressure_recovery
# ---------------------------------------------------------------------------

class TestTotalPressureRecovery:
    def test_correct_value(self):
        """完整 8 站位：σ = p_t2 / p_t0，数值正确。"""
        ifs = _make_full_stations()
        sigma = ifs.total_pressure_recovery()
        expected = ifs.st2.p_t / ifs.st0.p_t
        assert abs(sigma - expected) < 1e-12

    def test_known_pitot_sigma(self):
        """皮托管式：σ = 0.721 * 0.980 * 0.990 ≈ 0.699。"""
        ifs = _make_full_stations()
        sigma = ifs.total_pressure_recovery()
        expected = 1.0 * 0.980 * 0.990 * 0.721 * 1.0 * 1.0
        assert abs(sigma - expected) < 1e-10

    def test_st0_none_raises(self):
        """st0 为 None 时抛出 ValueError，错误信息含 'st0'。"""
        ifs = _make_full_stations()
        ifs.st0 = None
        with pytest.raises(ValueError, match="st0"):
            ifs.total_pressure_recovery()

    def test_st2_none_raises(self):
        """st2 为 None 时抛出 ValueError，错误信息含 'st2'。"""
        ifs = _make_full_stations()
        ifs.st2 = None
        with pytest.raises(ValueError, match="st2"):
            ifs.total_pressure_recovery()

    def test_both_none_raises(self):
        """st0 和 st2 均为 None 时也抛出 ValueError。"""
        ifs = InletFlowStations()
        with pytest.raises(ValueError):
            ifs.total_pressure_recovery()

    def test_sigma_lte_one(self):
        """进气道总压恢复系数不超过 1（熵增原理）。"""
        ifs = _make_full_stations()
        assert ifs.total_pressure_recovery() <= 1.0

    def test_no_loss_sigma_equals_one(self):
        """无任何损失时 σ = 1.0。"""
        p_t = 100000.0
        T_t = 300.0
        ifs = InletFlowStations(
            st0=FlowState(M=2.0, p_t=p_t, T_t=T_t),
            st2=FlowState(M=0.4, p_t=p_t, T_t=T_t),
        )
        assert abs(ifs.total_pressure_recovery() - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# recovery_chain
# ---------------------------------------------------------------------------

class TestRecoveryChain:
    def test_keys_present(self):
        """返回字典包含全部 5 个键。"""
        ifs = _make_full_stations()
        chain = ifs.recovery_chain()
        assert set(chain.keys()) == {"L/0", "EX/L", "1/EX", "SD/1", "2/SD"}

    def test_all_values_present_full(self):
        """完整站位：所有键值均为 float（非 None）。"""
        ifs = _make_full_stations()
        chain = ifs.recovery_chain()
        for key, val in chain.items():
            assert val is not None, f"chain['{key}'] 不应为 None"
            assert isinstance(val, float)

    def test_missing_intermediate_station(self):
        """中间站位缺失时，相关键返回 None，其他键正常。"""
        ifs = _make_full_stations()
        ifs.stEX = None   # 缺 stEX

        chain = ifs.recovery_chain()
        assert chain["L/0"]  is not None   # stL/st0 均有值
        assert chain["EX/L"] is None       # stEX 缺失
        assert chain["1/EX"] is None       # stEX 缺失（下游端）
        assert chain["SD/1"] is not None   # st1/stSD 均有值
        assert chain["2/SD"] is not None   # stSD/st2 均有值

    def test_chain_product_equals_total_recovery(self):
        """各段乘积应等于总压恢复系数。"""
        ifs = _make_full_stations()
        chain = ifs.recovery_chain()
        product = 1.0
        for val in chain.values():
            product *= val
        sigma = ifs.total_pressure_recovery()
        assert abs(product - sigma) < 1e-10

    def test_lossless_chain_all_ones(self):
        """无损失进气道：所有分项均为 1.0。"""
        p_t, T_t = 100000.0, 300.0

        def _st(M, label=""):
            return FlowState(M=M, p_t=p_t, T_t=T_t, label=label)

        ifs = InletFlowStations(
            st0=_st(2.0, "0"), stL=_st(2.0, "L"), stEX=_st(1.8, "EX"),
            stNS=_st(0.6, "NS"), st1=_st(1.4, "1"), stTH=_st(1.0, "TH"),
            stSD=_st(0.55, "SD"), st2=_st(0.4, "2"),
        )
        chain = ifs.recovery_chain()
        for key, val in chain.items():
            assert abs(val - 1.0) < 1e-12, f"chain['{key}']={val}（无损失应为 1.0）"

    def test_empty_stations_all_none(self):
        """全空对象：所有键值均为 None。"""
        chain = InletFlowStations().recovery_chain()
        for val in chain.values():
            assert val is None


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_returns_nonempty_string(self):
        """summary() 返回非空字符串。"""
        ifs = _make_full_stations()
        s = ifs.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_contains_M_keyword(self):
        """摘要字符串包含 'M' 字样（马赫数标识）。"""
        ifs = _make_full_stations()
        assert "M" in ifs.summary()

    def test_contains_sigma(self):
        """完整站位时摘要包含总压恢复系数信息。"""
        ifs = _make_full_stations()
        s = ifs.summary()
        assert "σ" in s or "sigma" in s.lower() or "总压恢复" in s

    def test_partial_stations_no_crash(self):
        """部分站位缺失时 summary() 不崩溃。"""
        ifs = InletFlowStations(
            st0=FlowState(M=2.0, p_t=101325.0, T_t=300.0, label="0"),
        )
        s = ifs.summary()
        assert isinstance(s, str)
        assert "M" in s

    def test_empty_stations_no_crash(self):
        """全空时 summary() 不崩溃，返回字符串。"""
        s = InletFlowStations().summary()
        assert isinstance(s, str)

    def test_all_labels_present(self):
        """摘要应包含全部 8 个站位标签（0/L/EX/NS/1/TH/SD/2）。"""
        ifs = _make_full_stations()
        s = ifs.summary()
        for tag in ["0", "L", "EX", "NS", "1", "TH", "SD", "2"]:
            assert tag in s, f"摘要中缺少站位标签 '{tag}'"


# ---------------------------------------------------------------------------
# attach_physical_conditions
# ---------------------------------------------------------------------------

def _make_pitot_normalized() -> InletFlowStations:
    """构造皮托管式归一化站位（p_t0=1.0, T_t0=1.0, sigma=0.7209）。"""
    sigma = 0.7209
    st0  = FlowState(M=2.0,    p_t=1.0,   T_t=1.0, label="0")
    stEX = FlowState(M=2.0,    p_t=1.0,   T_t=1.0, label="EX")
    stNS = FlowState(M=0.5774, p_t=sigma, T_t=1.0, label="NS")
    st1  = FlowState(M=0.5774, p_t=sigma, T_t=1.0, label="1")
    st2  = FlowState(M=0.5774, p_t=sigma, T_t=1.0, label="2")
    return InletFlowStations(st0=st0, stEX=stEX, stNS=stNS, st1=st1, st2=st2)


class TestAttachPhysicalConditions:
    """验证 InletFlowStations.attach_physical_conditions 的行为。"""

    H_KM = 20.0          # 设计飞行高度（km）
    H_M  = 20000.0       # 米
    M0   = 2.0
    MDOT = 100.0         # kg/s

    def _atm(self):
        return ISAAtmosphere(self.H_M)

    # ---- 正常工况 ----

    def test_st0_total_pressure_scaled(self):
        """st0.p_t 应等于真实来流总压（等熵关系）。"""
        ifs = _make_pitot_normalized()
        atm = self._atm()
        ifs.attach_physical_conditions(atm, self.M0, self.MDOT)

        P0_expected = atm.total_pressure(self.M0)
        assert abs(ifs.st0.p_t - P0_expected) < 1.0, (
            f"st0.p_t = {ifs.st0.p_t:.2f} Pa，期望 {P0_expected:.2f} Pa"
        )

    def test_st0_total_temperature_scaled(self):
        """st0.T_t 应等于真实来流总温（等熵关系）。"""
        ifs = _make_pitot_normalized()
        atm = self._atm()
        ifs.attach_physical_conditions(atm, self.M0, self.MDOT)

        T0_expected = atm.total_temperature(self.M0)
        assert abs(ifs.st0.T_t - T0_expected) < 0.1, (
            f"st0.T_t = {ifs.st0.T_t:.2f} K，期望 {T0_expected:.2f} K"
        )

    def test_st0_static_conditions_attached(self):
        """st0 应附加真实静温（T）、静压（p）。"""
        ifs = _make_pitot_normalized()
        atm = self._atm()
        ifs.attach_physical_conditions(atm, self.M0, self.MDOT)

        assert abs(ifs.st0.T - 216.65) < 0.01, f"st0.T = {ifs.st0.T}"
        assert abs(ifs.st0.p - 5474.9) < 2.0,  f"st0.p = {ifs.st0.p}"

    def test_st0_rho_and_v_attached(self):
        """st0 应附加密度 rho 和速度 v（动态属性）。"""
        ifs = _make_pitot_normalized()
        atm = self._atm()
        ifs.attach_physical_conditions(atm, self.M0, self.MDOT)

        assert hasattr(ifs.st0, "rho"), "st0 缺少 rho 属性"
        assert hasattr(ifs.st0, "v"),   "st0 缺少 v 属性"
        assert abs(ifs.st0.rho - 0.08803) < 0.00005
        assert abs(ifs.st0.v - 590.14)   < 0.5

    def test_capture_area_attached(self):
        """self.A_cap 应 ≈ 1.924 m²（±0.005 m²）。"""
        ifs = _make_pitot_normalized()
        atm = self._atm()
        ifs.attach_physical_conditions(atm, self.M0, self.MDOT)

        assert hasattr(ifs, "A_cap"), "InletFlowStations 缺少 A_cap 属性"
        assert abs(ifs.A_cap - 1.924) < 0.005, (
            f"A_cap = {ifs.A_cap:.4f} m²，期望 1.924 m²"
        )

    def test_sigma_preserved_after_scaling(self):
        """缩放不改变总压恢复系数 σ = p_t2 / p_t0。"""
        ifs = _make_pitot_normalized()
        sigma_before = ifs.total_pressure_recovery()   # 归一化下的 σ

        atm = self._atm()
        ifs.attach_physical_conditions(atm, self.M0, self.MDOT)
        sigma_after = ifs.total_pressure_recovery()    # 绝对量下的 σ

        assert abs(sigma_after - sigma_before) < 1e-9, (
            f"缩放前 σ={sigma_before:.8f}，缩放后 σ={sigma_after:.8f}，应相同"
        )

    def test_all_stations_pt_scaled(self):
        """所有非 None 站位的 p_t 均应按比例放大。"""
        ifs = _make_pitot_normalized()
        pt_before = {
            attr: getattr(ifs, attr).p_t
            for attr in ["st0", "stEX", "stNS", "st1", "st2"]
        }
        atm = self._atm()
        P0_abs = atm.total_pressure(self.M0)
        scale = P0_abs / 1.0   # p_t0_rel = 1.0

        ifs.attach_physical_conditions(atm, self.M0, self.MDOT)
        for attr, pt_orig in pt_before.items():
            pt_new = getattr(ifs, attr).p_t
            assert abs(pt_new - pt_orig * scale) < 1.0, (
                f"{attr}.p_t 缩放异常：{pt_orig:.6f} × {scale:.2f} ≠ {pt_new:.2f}"
            )

    # ---- 错误处理 ----

    def test_mdot_zero_raises(self):
        """m_dot=0（σ=0 语义）时应抛出 ValueError。"""
        ifs = _make_pitot_normalized()
        atm = self._atm()
        with pytest.raises(ValueError, match="m_dot"):
            ifs.attach_physical_conditions(atm, self.M0, 0.0)

    def test_mdot_negative_raises(self):
        """m_dot < 0 时应抛出 ValueError。"""
        ifs = _make_pitot_normalized()
        atm = self._atm()
        with pytest.raises(ValueError):
            ifs.attach_physical_conditions(atm, self.M0, -50.0)

    def test_st0_none_raises(self):
        """st0 为 None 时应抛出 ValueError。"""
        ifs = InletFlowStations()
        atm = self._atm()
        with pytest.raises(ValueError, match="st0"):
            ifs.attach_physical_conditions(atm, self.M0, self.MDOT)

    # ---- 物理量纲验证 ----

    def test_A_cap_units_m2(self):
        """A_cap 量纲应为 m²，数值在合理范围内（0.1~10 m²）。"""
        ifs = _make_pitot_normalized()
        atm = self._atm()
        ifs.attach_physical_conditions(atm, self.M0, self.MDOT)
        assert 0.1 < ifs.A_cap < 10.0, f"A_cap = {ifs.A_cap:.4f} m² 不在合理范围"

    def test_A_cap_slater2023_tolerance(self):
        """与 Slater 2023 基准偏差 < 2%。"""
        ifs = _make_pitot_normalized()
        atm = self._atm()
        ifs.attach_physical_conditions(atm, self.M0, self.MDOT)
        slater_ref = 1.924
        rel_err = abs(ifs.A_cap - slater_ref) / slater_ref
        assert rel_err < 0.02, (
            f"A_cap = {ifs.A_cap:.4f} m²，相对误差 {rel_err*100:.2f}% > 2%"
        )
