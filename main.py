"""
超声速进气道参数化设计工具 — 组会演示主界面
运行：venv/Scripts/python.exe main.py
"""

import time
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib
matplotlib.rcParams["font.family"] = "Microsoft YaHei"
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import ttkbootstrap as ttkb
from ttkbootstrap.constants import *

# ---------------------------------------------------------------------------
# 构型参数配置（数据驱动）
# 每条参数：(key, 标签, 单位, (min, max), 精度digits, 默认值)
# ---------------------------------------------------------------------------
INLET_CONFIGS = {
    "pitot": {
        "label": "皮托管（正激波）",
        "params": [
            ("M0", "来流马赫数",   "-",  (1.1, 5.0), 2, 2.0),
            ("D2", "出口直径",     "m",  (0.1, 5.0), 2, 1.0),
        ],
    },
    "external_2d": {
        "label": "二元外压式",
        "params": [
            ("M0",          "来流马赫数",           "-",   (1.1, 5.0),  2, 2.0),
            ("N_stages",    "斜激波级数",           "-",   (1,   4),    0, 3),
            ("M_EX",        "终端激波前马赫数",     "-",   (1.0, 2.5),  2, 1.4),
            ("D2",          "出口当量直径",         "m",   (0.1, 5.0),  2, 1.0),
            ("N_throat",    "喉道过渡系数",         "-",   (1.0, 5.0),  2, 2.0),
            ("sigma_diff",  "扩压段总压恢复",       "-",   (0.80, 1.0), 4, 0.95),
            ("H_diff",      "扩压偏置高度",         "m",   (0.0, 5.0),  3, 1.0),
            ("L_diff_extra","扩压段附加长度",       "m",   (0.0, 10.0), 2, 2.0),
            ("mode_clip",   "唇口形式(1=尖/2=圆)", "-",   (1, 2),      0, 1),
            ("deltheta_clip","唇口偏折角",          "°",   (0.0, 10.0), 1, 4.0),
        ],
    },
    "axisymmetric": {
        "label": "轴对称锥形激波",
        "params": [
            ("M0",          "来流马赫数",       "-",  (1.1, 5.0),  2, 2.0),
            ("delta_c_deg", "锥半角",           "°",  (5.0, 40.0), 1, 22.0),
            ("M_EX",        "终端激波前马赫数", "-",  (1.0, 2.5),  2, 1.3),
            ("D2",          "出口直径",         "m",  (0.1, 5.0),  2, 1.4),
            ("sigma_diff",  "扩压段总压恢复",   "-",  (0.80, 1.0), 4, 0.95),
        ],
    },
}

OUTPUT_FIELDS = [
    ("总压恢复系数 σ",        "sigma",   "-"),
    ("终端激波前马赫数 M_EX", "M_EX",   "-"),
    ("激波后马赫数 M_NS",     "M_NS",   "-"),
    ("来流马赫数 M₀",         "M0",     "-"),
]


class InletDesignApp(ttkb.Window):
    def __init__(self):
        super().__init__(title="超声速进气道参数化设计工具", themename="litera")
        self.geometry("1200x750")
        self.minsize(1200, 750)
        self.resizable(True, True)

        self.inlet_var = tk.StringVar(value="pitot")
        self.param_vars: dict[str, tk.Variable] = {}

        self._build_ui()
        self._on_inlet_change()  # 初始化参数面板

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ---- 标题栏 ----
        title_bar = ttkb.Frame(self, padding=(12, 6))
        title_bar.pack(side=TOP, fill=X)
        ttkb.Label(
            title_bar,
            text="超声速进气道参数化设计工具",
            font=("Microsoft YaHei", 16, "bold"),
            bootstyle="primary",
        ).pack(side=LEFT)

        # ---- 主体区域（左 + 右） ----
        body = ttkb.Frame(self)
        body.pack(fill=BOTH, expand=True, padx=8, pady=(0, 4))

        # 左侧参数面板（固定宽度）
        self.left_panel = ttkb.Frame(body, width=270, padding=8)
        self.left_panel.pack(side=LEFT, fill=Y)
        self.left_panel.pack_propagate(False)

        # 右侧容器
        right_panel = ttkb.Frame(body, padding=(8, 0, 0, 0))
        right_panel.pack(side=LEFT, fill=BOTH, expand=True)

        # 右上：性能结果表格
        result_lf = ttkb.LabelFrame(right_panel, text="性能结果")
        result_lf.pack(fill=X, pady=(0, 6))
        result_frame = ttkb.Frame(result_lf)
        result_frame.pack(fill=X, padx=8, pady=6)
        self._build_result_table(result_frame)

        # 右下：matplotlib 型线预览
        preview_lf = ttkb.LabelFrame(right_panel, text="进气道型线预览")
        preview_lf.pack(fill=BOTH, expand=True)
        preview_frame = ttkb.Frame(preview_lf)
        preview_frame.pack(fill=BOTH, expand=True, padx=8, pady=6)
        self._build_canvas(preview_frame)

        # ---- 底部控制栏 ----
        self._build_bottom_bar()

        # ---- 左侧内容 ----
        self._build_left_panel()

    def _build_left_panel(self):
        lp = self.left_panel

        # 构型选择
        sel_lf = ttkb.LabelFrame(lp, text="构型选择")
        sel_lf.pack(fill=X, pady=(0, 8))
        sel_frame = ttkb.Frame(sel_lf)
        sel_frame.pack(fill=X, padx=6, pady=4)
        choices = [(v["label"], k) for k, v in INLET_CONFIGS.items()]
        self.inlet_combo = ttkb.Combobox(
            sel_frame,
            textvariable=self.inlet_var,
            values=[v["label"] for v in INLET_CONFIGS.values()],
            state="readonly",
            width=22,
        )
        # 显示标签，内部用 key
        self._label_to_key = {v["label"]: k for k, v in INLET_CONFIGS.items()}
        self._key_to_label = {k: v["label"] for k, v in INLET_CONFIGS.items()}
        self.inlet_var.set(INLET_CONFIGS["pitot"]["label"])
        self.inlet_combo.pack(fill=X)
        self.inlet_combo.bind("<<ComboboxSelected>>", lambda _: self._on_inlet_change())

        # 参数输入区（动态刷新）
        param_lf = ttkb.LabelFrame(lp, text="设计参数")
        param_lf.pack(fill=BOTH, expand=True)
        self.param_frame = ttkb.Frame(param_lf)
        self.param_frame.pack(fill=BOTH, expand=True, padx=6, pady=4)

    def _build_result_table(self, parent):
        cols = ("参数", "数值", "单位")
        self.result_tree = ttkb.Treeview(
            parent, columns=cols, show="headings", height=5, bootstyle="primary"
        )
        for col, w in zip(cols, (240, 120, 60)):
            self.result_tree.heading(col, text=col)
            self.result_tree.column(col, width=w, anchor=CENTER if col != "参数" else W)
        self.result_tree.pack(fill=X)
        # 插入占位行
        for label, _, unit in OUTPUT_FIELDS:
            self.result_tree.insert("", END, values=(label, "—", unit))

    def _build_canvas(self, parent):
        self.fig, self.ax = plt.subplots(figsize=(7, 3.8), dpi=96)
        self.fig.patch.set_facecolor("#f8f9fa")
        self.ax.set_aspect("equal")
        self.ax.set_title("（点击「计算」后显示型线）", fontsize=10, color="gray")
        self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    def _build_bottom_bar(self):
        bar = ttkb.Frame(self, padding=(8, 4))
        bar.pack(side=BOTTOM, fill=X)

        self.status_var = tk.StringVar(value="就绪")
        ttkb.Label(bar, textvariable=self.status_var, bootstyle="secondary").pack(side=LEFT)

        ttkb.Button(bar, text="导出结果", bootstyle="outline-secondary",
                    command=self._export_results).pack(side=RIGHT, padx=4)
        ttkb.Button(bar, text="计算", bootstyle="success",
                    command=self._start_calculation).pack(side=RIGHT, padx=4)
        ttkb.Button(bar, text="重置", bootstyle="outline",
                    command=self._reset_params).pack(side=RIGHT, padx=4)

    # ------------------------------------------------------------------
    # 参数面板动态刷新
    # ------------------------------------------------------------------

    def _on_inlet_change(self):
        label = self.inlet_var.get()
        key = self._label_to_key.get(label, "pitot")
        self._current_key = key

        # 清空旧控件
        for w in self.param_frame.winfo_children():
            w.destroy()
        self.param_vars.clear()

        cfg = INLET_CONFIGS[key]["params"]
        for param_key, param_label, unit, (pmin, pmax), digits, default in cfg:
            row = ttkb.Frame(self.param_frame, padding=(0, 3))
            row.pack(fill=X)

            ttkb.Label(row, text=param_label, width=14, anchor=W).pack(side=LEFT)

            if digits == 0:
                var = tk.IntVar(value=int(default))
                entry = ttkb.Spinbox(
                    row, from_=pmin, to=pmax, textvariable=var, width=7,
                    increment=1, format="%d",
                )
            else:
                var = tk.DoubleVar(value=default)
                fmt = f"%.{digits}f"
                entry = ttkb.Spinbox(
                    row, from_=pmin, to=pmax, textvariable=var, width=7,
                    increment=10 ** (-digits), format=fmt,
                )
            entry.pack(side=LEFT, padx=(4, 2))

            hint = f"[{unit}] {pmin}~{pmax}" if unit != "-" else f"{pmin}~{pmax}"
            ttkb.Label(row, text=hint, bootstyle="secondary", font=("Microsoft YaHei", 8)).pack(side=LEFT)

            self.param_vars[param_key] = var

    def _reset_params(self):
        self._on_inlet_change()

    # ------------------------------------------------------------------
    # 计算
    # ------------------------------------------------------------------

    def _start_calculation(self):
        key = self._current_key
        params = {}
        try:
            for k, var in self.param_vars.items():
                params[k] = var.get()
        except Exception as e:
            messagebox.showerror("参数错误", str(e))
            return

        t0 = time.perf_counter()
        try:
            stations, geo = self._run_design(key, params)
        except Exception as e:
            messagebox.showerror("计算错误", str(e))
            self.status_var.set(f"计算失败：{e}")
            return

        elapsed = time.perf_counter() - t0
        self._show_results(stations, params)
        self._draw_profile(key, geo)
        self.status_var.set(f"计算完成，耗时 {elapsed*1000:.1f} ms")

    def _run_design(self, key, params):
        if key == "pitot":
            from inlets.pitot.aero_design import design_pitot
            from inlets.pitot.geometry import pitot_geometry
            stations = design_pitot(M0=params["M0"])
            geo = pitot_geometry(stations, D2=params["D2"])

        elif key == "external_2d":
            from inlets.external_2d.aero_design import design_external_2d, oswatitsch_angles
            from inlets.external_2d.geometry import external_2d_geometry
            stations = design_external_2d(
                M0=params["M0"],
                N_stages=int(params["N_stages"]),
                M_EX=params["M_EX"],
            )
            angles = oswatitsch_angles(params["M0"], int(params["N_stages"]), params["M_EX"])
            geo = external_2d_geometry(
                stations, angles, D2=params["D2"],
                N_throat=params["N_throat"],
                L_diff_extra=params["L_diff_extra"],
            )

        elif key == "axisymmetric":
            from inlets.axisymmetric.aero_design import design_axisymmetric
            from inlets.axisymmetric.geometry import axisymmetric_geometry
            stations = design_axisymmetric(
                M0=params["M0"],
                delta_c_deg=params["delta_c_deg"],
                M_EX=params["M_EX"],
            )
            geo = axisymmetric_geometry(stations, D2=params["D2"])

        else:
            raise ValueError(f"未知构型：{key}")

        return stations, geo

    def _show_results(self, stations, params):
        sigma = stations.total_pressure_recovery()
        M_EX = stations.stEX.M if stations.stEX else None
        M_NS = stations.stNS.M if stations.stNS else None
        M0 = stations.st0.M if stations.st0 else params.get("M0")

        values_map = {
            "sigma": f"{sigma:.6f}" if sigma is not None else "—",
            "M_EX":  f"{M_EX:.4f}" if M_EX is not None else "—",
            "M_NS":  f"{M_NS:.4f}" if M_NS is not None else "—",
            "M0":    f"{M0:.4f}"   if M0  is not None else "—",
        }

        # 清空并重新填充
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        for label, field, unit in OUTPUT_FIELDS:
            self.result_tree.insert("", END, values=(label, values_map.get(field, "—"), unit))

        self._last_results = values_map

    # ------------------------------------------------------------------
    # 型线绘图（容错：几何异常不影响气动结果显示）
    # ------------------------------------------------------------------

    def _draw_profile(self, key, geo):
        self.ax.clear()
        self.ax.set_facecolor("#f8f9fa")

        try:
            if key == "pitot":
                self._plot_pitot(geo)
            elif key == "external_2d":
                self._plot_ext2d(geo)
            elif key == "axisymmetric":
                self._plot_axisymmetric(geo)
        except Exception:
            self.ax.set_title("几何绘图失败（气动结果仍有效）", color="red", fontsize=10)

        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y / r (m)")
        self.ax.grid(True, linestyle="--", alpha=0.4)
        self.fig.tight_layout()
        self.canvas.draw()

    def _plot_pitot(self, geo):
        profile = geo.get("profile", [])
        if profile:
            xs, ys = zip(*profile)
            self.ax.plot(xs, ys, "b-", linewidth=2, label="型线")
            # 镜像下壁
            self.ax.plot(xs, [-y for y in ys], "b-", linewidth=2)
        ns = geo.get("normal_shock_points")
        if ns:
            (x1, y1), (x2, y2) = ns
            self.ax.plot([x1, x2], [y1, y2],
                         color="red", linewidth=2, linestyle="-",
                         label="终端正激波")
        else:
            self.ax.axvline(x=0, color="red", linestyle="--", linewidth=1, label="唇口")
        self.ax.set_title("皮托管进气道型线")
        self.ax.legend(fontsize=9)

    def _plot_ext2d(self, geo):
        upper = geo.get("profile_upper", [])
        lower = geo.get("profile_lower", [])
        ramp  = geo.get("ramp_points",   [])
        shocks = geo.get("shock_points", [])

        if upper:
            xs, ys = zip(*upper)
            self.ax.plot(xs, ys, "b-", linewidth=2, label="上壁（cowl）")
        if lower:
            xs, ys = zip(*lower)
            self.ax.plot(xs, ys, "k-", linewidth=2, label="下壁（斜面）")
        for seg in shocks:
            if len(seg) == 2:
                (x1, y1), (x2, y2) = seg
                self.ax.plot([x1, x2], [y1, y2], "r--", linewidth=1.5, alpha=0.7)
        if ramp:
            xs, ys = zip(*ramp)
            self.ax.plot(xs, ys, "go", markersize=5)
        ns = geo.get("normal_shock_points")
        if ns:
            (x1, y1), (x2, y2) = ns
            self.ax.plot([x1, x2], [y1, y2],
                         color="red", linewidth=2, linestyle="-",
                         label="终端正激波")

        self.ax.set_title("二元外压式进气道型线")
        self.ax.legend(fontsize=9)

    def _plot_axisymmetric(self, geo):
        cb    = geo.get("profile_cb",   [])
        cowl  = geo.get("profile_cowl", [])

        if cb:
            xs, ys = zip(*cb)
            self.ax.plot(xs, ys, "k-", linewidth=2, label="中心锥")
            self.ax.plot(xs, [-y for y in ys], "k-", linewidth=2)
        if cowl:
            xs, ys = zip(*cowl)
            self.ax.plot(xs, ys, "b-", linewidth=2, label="外罩（cowl）")
            self.ax.plot(xs, [-y for y in ys], "b-", linewidth=2)

        self.ax.axhline(y=0, color="gray", linestyle=":", linewidth=1)
        ns = geo.get("normal_shock_points")
        if ns:
            (x1, y1), (x2, y2) = ns
            # 上半（径向正方向）
            self.ax.plot([x1, x2], [y1, y2],
                         color="red", linewidth=2, linestyle="-",
                         label="终端正激波")
            # 下半（镜像，轴对称）
            self.ax.plot([x1, x2], [-y1, -y2],
                         color="red", linewidth=2, linestyle="-")

        self.ax.set_title("轴对称锥形激波进气道型线")
        self.ax.legend(fontsize=9)

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------

    def _export_results(self):
        if not hasattr(self, "_last_results"):
            messagebox.showinfo("提示", "请先执行计算。")
            return
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
            title="导出结果",
        )
        if not path:
            return
        lines = [f"超声速进气道设计结果 — {self._key_to_label.get(self._current_key, '')}"]
        lines.append("-" * 40)
        for label, field, unit in OUTPUT_FIELDS:
            lines.append(f"{label:25s}  {self._last_results.get(field, '—'):>12}  {unit}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        messagebox.showinfo("导出成功", f"结果已保存至：\n{path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = InletDesignApp()
    app.mainloop()
