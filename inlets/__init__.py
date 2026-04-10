"""
inlets/__init__.py
==================
工厂函数，通过名称获取进气道构型实例。
"""

from __future__ import annotations


def get_inlet(name: str):
    """获取指定构型的气动设计模块。

    Parameters
    ----------
    name : str
        构型名称，取值为 ``"pitot"``、``"external_2d"``、``"axisymmetric"``。

    Returns
    -------
    module
        对应构型的 aero_design 模块。

    Raises
    ------
    ValueError
        若 name 不在已知构型列表中。
    """
    if name == "pitot":
        from inlets.pitot import aero_design
        return aero_design
    elif name == "external_2d":
        from inlets.external_2d import aero_design
        return aero_design
    elif name == "axisymmetric":
        from inlets.axisymmetric import aero_design
        return aero_design
    else:
        raise ValueError(
            f"未知构型 '{name}'，有效值为 'pitot'、'external_2d'、'axisymmetric'。"
        )
