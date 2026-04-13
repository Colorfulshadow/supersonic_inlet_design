"""
core/flow_stations.py
=====================
8 站位流场框架数据类，对应 CLAUDE.md §六。

新增方法
--------
- ``attach_physical_conditions(atm, M0, m_dot)``
  为各站位附加真实物理绝对量（将归一化总压按大气条件缩放，并为 st0 附加
  真实静温、静压、密度、速度等字段）。

站位定义
--------
0   自由来流，M₀
L   进气道前缘局部来流
EX  外部超声速扩压末端（终端正激波上游）
NS  终端正激波下游
1   唇口入口截面
TH  喉道
SD  亚声速扩压起点
2   发动机面（进气道出口）

总压恢复链：
    σ = (p_tL/p_t0) · (p_tEX/p_tL) · (p_t1/p_tEX) · (p_tSD/p_t1) · (p_t2/p_tSD)
      = p_t2 / p_t0

禁止在本模块中导入 geometry/、gui/ 或 inlets/ 的任何内容。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    # 仅用于类型注解，避免运行时循环导入
    from core.atmosphere import ISAAtmosphere


@dataclass
class FlowState:
    """单个流场站位的热力学状态。

    Parameters
    ----------
    M : float
        马赫数（无量纲）。
    p_t : float
        总压（Pa）；各站位应使用一致的参考体系（绝对值或归一化均可，
        但同一 :class:`InletFlowStations` 实例内必须一致）。
    T_t : float
        总温（K）。
    p : float, optional
        静压（Pa）。若未给出，可通过等熵关系由 M、p_t 推算。
    T : float, optional
        静温（K）。若未给出，可通过等熵关系由 M、T_t 推算。
    label : str, optional
        站位标签，如 ``"0"``、``"EX"``、``"NS"`` 等，用于显示。
    """

    M: float
    p_t: float
    T_t: float
    p: Optional[float] = None
    T: Optional[float] = None
    label: str = ""

    def __post_init__(self) -> None:
        if self.M < 0:
            raise ValueError(f"马赫数不能为负：M={self.M}")
        if self.p_t <= 0:
            raise ValueError(f"总压必须为正：p_t={self.p_t}")
        if self.T_t <= 0:
            raise ValueError(f"总温必须为正：T_t={self.T_t}")


@dataclass
class InletFlowStations:
    """8 站位流场容器，对应 CLAUDE.md §六。

    各站位均为可选（默认 ``None``），在气动计算完成后逐步赋值。

    Attributes
    ----------
    st0  : FlowState | None  自由来流
    stL  : FlowState | None  进气道前缘局部来流
    stEX : FlowState | None  外部超声速扩压末端（NS 上游）
    stNS : FlowState | None  终端正激波下游
    st1  : FlowState | None  唇口入口截面
    stTH : FlowState | None  喉道
    stSD : FlowState | None  亚声速扩压起点
    st2  : FlowState | None  发动机面（进气道出口）
    """

    st0:  Optional[FlowState] = None
    stL:  Optional[FlowState] = None
    stEX: Optional[FlowState] = None
    stNS: Optional[FlowState] = None
    st1:  Optional[FlowState] = None
    stTH: Optional[FlowState] = None
    stSD: Optional[FlowState] = None
    st2:  Optional[FlowState] = None

    # ------------------------------------------------------------------
    # 总压恢复
    # ------------------------------------------------------------------

    def total_pressure_recovery(self) -> float:
        """总压恢复系数 σ = p_t2 / p_t0。

        Returns
        -------
        float
            σ ∈ (0, 1]。

        Raises
        ------
        ValueError
            若 ``st0`` 或 ``st2`` 为 ``None``。
        """
        missing = [name for name, st in (("st0", self.st0), ("st2", self.st2)) if st is None]
        if missing:
            raise ValueError(
                f"计算总压恢复系数需要站位 {missing}，但它们当前为 None。"
            )
        return self.st2.p_t / self.st0.p_t  # type: ignore[union-attr]

    def recovery_chain(self) -> dict[str, Optional[float]]:
        """各段分项总压恢复比。

        返回字典键（按流向顺序）：
            ``'L/0'``、``'EX/L'``、``'1/EX'``、``'SD/1'``、``'2/SD'``

        某段任一端为 ``None`` 时，对应键值为 ``None``；否则为
        下游总压 / 上游总压。

        Returns
        -------
        dict[str, float | None]
        """
        def _ratio(downstream: Optional[FlowState], upstream: Optional[FlowState]) -> Optional[float]:
            if downstream is None or upstream is None:
                return None
            return downstream.p_t / upstream.p_t

        return {
            "L/0":   _ratio(self.stL,  self.st0),
            "EX/L":  _ratio(self.stEX, self.stL),
            "1/EX":  _ratio(self.st1,  self.stEX),
            "SD/1":  _ratio(self.stSD, self.st1),
            "2/SD":  _ratio(self.st2,  self.stSD),
        }

    def summary(self) -> str:
        """格式化的站位参数摘要字符串，供打印和 GUI 显示。

        Returns
        -------
        str
            多行文本，每行一个站位，格式：
            ``<标签>  M=<M>  p_t=<p_t>  T_t=<T_t>``
        """
        _station_order = [
            ("0",  self.st0),
            ("L",  self.stL),
            ("EX", self.stEX),
            ("NS", self.stNS),
            ("1",  self.st1),
            ("TH", self.stTH),
            ("SD", self.stSD),
            ("2",  self.st2),
        ]
        lines = ["进气道 8 站位流场摘要", "-" * 52]
        for tag, st in _station_order:
            if st is None:
                lines.append(f"  {tag:3s}  (未赋值)")
            else:
                label_str = f"[{st.label}]" if st.label else ""
                p_str = f"p_t={st.p_t:.2f} Pa" if st.p_t >= 1.0 else f"p_t={st.p_t:.6f}"
                lines.append(
                    f"  {tag:3s}  {label_str:6s}  M={st.M:.4f}  {p_str}  T_t={st.T_t:.2f} K"
                )
        # 总压恢复（如 st0、st2 均有值）
        if self.st0 is not None and self.st2 is not None:
            sigma = self.total_pressure_recovery()
            lines.append("-" * 52)
            lines.append(f"  总压恢复系数 σ = {sigma:.6f}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 物理定尺：将归一化量转换为真实绝对量
    # ------------------------------------------------------------------

    def attach_physical_conditions(
        self,
        atm: "ISAAtmosphere",
        M0: float,
        m_dot: float,
    ) -> None:
        """为各站位附加真实物理绝对量。

        将当前各站位的归一化总压（以 st0.p_t 为基准的相对比值）缩放为
        基于 ISA 标准大气的绝对总压值，并为 st0 附加真实静温、静压、
        密度、速度等附加字段。

        Parameters
        ----------
        atm : ISAAtmosphere
            已初始化的大气对象（包含飞行高度处的静态参数）。
        M0 : float
            来流马赫数（与气动设计时一致）。
        m_dot : float
            质量流量（kg/s），用于计算真实捕获面积；必须 > 0。

        Raises
        ------
        ValueError
            - 若 ``m_dot ≤ 0``（零质量流量无物理意义）。
            - 若 ``st0`` 为 ``None``（基准站位缺失，无法缩放）。
            - 若 ``st0.p_t ≤ 0``（归一化基准为零，无法建立缩放关系）。

        Notes
        -----
        **缩放逻辑**：调用前各站位 ``p_t`` 通常以 ``st0.p_t = 1.0`` 归一化，
        ``p_t[i] / p_t[0]`` 表示该站位相对 st0 的总压比（= 链式总压恢复比）。
        本方法计算真实 ``p_t0_abs = p_static * (1+(γ-1)/2*M²)^(γ/(γ-1))``，
        然后对所有非 None 站位执行 ``p_t_abs = p_t_rel / p_t0_rel * p_t0_abs``。
        总温同理（``T_t = T_static * (1+(γ-1)/2*M²)``，各站等量缩放）。

        **附加属性**（动态挂载，不修改 dataclass 定义）：

        - ``st0.T``   → 真实静温（K）（覆盖原 Optional[float]）
        - ``st0.p``   → 真实静压（Pa）（覆盖原 Optional[float]）
        - ``st0.rho`` → 密度（kg/m³）
        - ``st0.v``   → 来流速度（m/s）
        - ``self.A_cap`` → 捕获面积（m²）
        """
        # ---- 参数校验 ----
        if m_dot <= 0.0:
            raise ValueError(
                f"质量流量 m_dot 必须 > 0，当前 m_dot={m_dot}。"
                "零或负质量流量在物理上无意义。"
            )
        if self.st0 is None:
            raise ValueError(
                "attach_physical_conditions 要求 st0 已赋值，"
                "请先完成气动设计再调用此方法。"
            )
        p_t0_rel: float = self.st0.p_t
        if p_t0_rel <= 0.0:
            raise ValueError(
                f"st0.p_t 必须 > 0，当前 {p_t0_rel}。归一化基准无效。"
            )

        # ---- 来流真实总条件（等熵）----
        gamma = atm.gamma
        T_t0_abs: float = atm.total_temperature(M0)   # K
        p_t0_abs: float = atm.total_pressure(M0)      # Pa

        # ---- 缩放因子 ----
        p_scale = p_t0_abs / p_t0_rel
        T_scale = T_t0_abs / self.st0.T_t   # 通常 st0.T_t=1.0，T_scale=T_t0_abs

        # ---- 更新所有非 None 站位的绝对总温总压 ----
        _all_stations = [
            "st0", "stL", "stEX", "stNS", "st1", "stTH", "stSD", "st2",
        ]
        for attr in _all_stations:
            st: Optional[FlowState] = getattr(self, attr)
            if st is not None:
                st.p_t = st.p_t * p_scale
                st.T_t = st.T_t * T_scale

        # ---- 为 st0 附加真实静条件及速度 ----
        self.st0.T = atm.T_static                     # 静温（K）
        self.st0.p = atm.p_static                     # 静压（Pa）
        self.st0.rho = atm.rho                        # type: ignore[attr-defined]
        self.st0.v   = atm.velocity(M0)               # type: ignore[attr-defined]

        # ---- 捕获面积 ----
        self.A_cap = atm.capture_area(m_dot, M0)      # type: ignore[attr-defined]
