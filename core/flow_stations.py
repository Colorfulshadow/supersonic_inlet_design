"""
core/flow_stations.py
=====================
8 站位流场框架数据类，对应 CLAUDE.md §六。

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

from dataclasses import dataclass, field
from typing import Optional


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
