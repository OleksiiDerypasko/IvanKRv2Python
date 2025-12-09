"""
Віджет для відображення графіків процесу мінімізації в компактному темному стилі.

Показує один графік за раз у вигляді каруселі:
    - поверхня f(x1, x2) + траєкторія;
    - графік f(k);
    - контурні лінії + траєкторія.

Публічні методи залишаються сумісними з попередньою версією:
    show_placeholder(), plot_fk(...), plot_contour_trajectory(...)
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QStackedWidget,
    QComboBox,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from core.iteration_result import IterationResult
from .styles import MARGIN, SPACING, apply_card_style, PALETTE

_CANVAS_BG = PALETTE.surface_alt
_ACCENT = PALETTE.accent
_TEXT = PALETTE.text_main
_MUTED = PALETTE.text_muted


class PlotPage:
    def __init__(self, figure: Figure, canvas: FigureCanvas, axes):
        self.figure = figure
        self.canvas = canvas
        self.axes = axes


class PlotView(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("plotView")
        self.pages_order = ["surface", "fk", "contour"]
        self.pages: dict[str, PlotPage] = {}
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        layout.setSpacing(SPACING)

        apply_card_style(self)

        # Навігація між графіками
        nav = QHBoxLayout()
        nav.setSpacing(SPACING)
        nav.setContentsMargins(0, 0, 0, 0)

        nav.addWidget(QLabel("Графік:", self))

        self.combo_mode = QComboBox(self)
        self.combo_mode.addItems([
            "Поверхня f(x₁, x₂)",
            "Графік f(k)",
            "Рівні та траєкторія",
        ])
        self.combo_mode.currentIndexChanged.connect(self._on_combo_changed)
        nav.addWidget(self.combo_mode, stretch=1)

        self.btn_prev = QPushButton("◀")
        self.btn_next = QPushButton("▶")
        for btn in (self.btn_prev, self.btn_next):
            btn.setFixedWidth(34)
        self.btn_prev.clicked.connect(self._on_prev)
        self.btn_next.clicked.connect(self._on_next)

        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)

        layout.addLayout(nav)

        # Стек полотен
        self.stacked = QStackedWidget(self)
        layout.addWidget(self.stacked, stretch=1)

        self._create_pages()
        self.show_placeholder()

    def _create_pages(self) -> None:
        self.pages["surface"] = self._create_page(projection="3d")
        self.pages["fk"] = self._create_page()
        self.pages["contour"] = self._create_page()

        for key in self.pages_order:
            self.stacked.addWidget(self.pages[key].canvas)

    def _create_page(self, projection: Optional[str] = None) -> PlotPage:
        figure = Figure(facecolor=_CANVAS_BG)
        if projection == "3d":
            ax = figure.add_subplot(111, projection="3d")
        else:
            ax = figure.add_subplot(111)
        canvas = FigureCanvas(figure)
        canvas.setStyleSheet("background-color: transparent;")
        return PlotPage(figure, canvas, ax)

    # ------------------------------------------------------------------
    # Навігація
    # ------------------------------------------------------------------
    def _on_combo_changed(self, index: int) -> None:
        self.stacked.setCurrentIndex(index)

    def _on_prev(self) -> None:
        idx = (self.stacked.currentIndex() - 1) % len(self.pages_order)
        self.stacked.setCurrentIndex(idx)
        self.combo_mode.setCurrentIndex(idx)

    def _on_next(self) -> None:
        idx = (self.stacked.currentIndex() + 1) % len(self.pages_order)
        self.stacked.setCurrentIndex(idx)
        self.combo_mode.setCurrentIndex(idx)

    def _set_page(self, key: str) -> None:
        idx = self.pages_order.index(key)
        self.stacked.setCurrentIndex(idx)
        self.combo_mode.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # Стилізація
    # ------------------------------------------------------------------
    def _style_2d_axes(self, ax) -> None:
        ax.set_facecolor(_CANVAS_BG)
        ax.tick_params(colors=_MUTED, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(PALETTE.border)
            spine.set_linewidth(0.8)
        ax.grid(True, color=PALETTE.border, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.title.set_color(_TEXT)
        ax.xaxis.label.set_color(_TEXT)
        ax.yaxis.label.set_color(_TEXT)

    def _style_3d_axes(self, ax) -> None:
        ax.set_facecolor(_CANVAS_BG)
        ax.tick_params(colors=_MUTED, labelsize=8)
        ax.xaxis.label.set_color(_TEXT)
        ax.yaxis.label.set_color(_TEXT)
        ax.zaxis.label.set_color(_TEXT)
        ax.title.set_color(_TEXT)

    def _redraw(self, key: str) -> None:
        page = self.pages[key]
        page.figure.tight_layout()
        page.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Публічні методи
    # ------------------------------------------------------------------
    def show_placeholder(self) -> None:
        messages = {
            "surface": "Графік поверхні з'явиться після запуску",
            "fk": "Графік f(k) з'явиться після запуску",
            "contour": "Контурні лінії з'являться після запуску",
        }
        for key, msg in messages.items():
            page = self.pages[key]
            ax = page.axes
            ax.clear()
            if key == "surface":
                self._style_3d_axes(ax)
                ax.text(0.5, 0.5, 0.5, msg, ha="center", va="center", color=_MUTED)
            else:
                self._style_2d_axes(ax)
                ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color=_MUTED)
            page.canvas.draw_idle()

    def plot_fk(self, iterations: List[IterationResult]) -> None:
        if not iterations:
            self.show_placeholder()
            return

        ax = self.pages["fk"].axes
        ax.clear()
        self._style_2d_axes(ax)

        ks = [it.index for it in iterations]
        fs = [float(it.f) for it in iterations]

        ax.plot(ks, fs, marker="o", linestyle="-", linewidth=1.5, markersize=4, color=_ACCENT)
        ax.set_xlabel("k (номер ітерації)")
        ax.set_ylabel("f(xₖ)")
        ax.set_title("Графік f(k)")

        self._redraw("fk")
        self._set_page("fk")

    def plot_contour_trajectory(
        self,
        func: Callable[[np.ndarray], float],
        iterations: List[IterationResult],
        levels: int = 18,
        padding: float = 0.5,
        grid_size: int = 120,
    ) -> None:
        contour_ax = self.pages["contour"].axes
        surface_ax = self.pages["surface"].axes
        contour_ax.clear()
        surface_ax.clear()

        if not iterations:
            self.show_placeholder()
            return

        xs = np.array([it.x for it in iterations], dtype=float)
        if xs.ndim != 2 or xs.shape[1] != 2:
            self._style_2d_axes(contour_ax)
            self._style_3d_axes(surface_ax)
            contour_ax.text(0.5, 0.5, "Contour доступний лише для задачі в R²", ha="center", va="center", transform=contour_ax.transAxes, color=_MUTED)
            surface_ax.text(0.5, 0.5, 0.5, "Поверхню можна показати лише для R²", ha="center", va="center", color=_MUTED)
            self._redraw("contour")
            self._redraw("surface")
            self._set_page("contour")
            return

        x1_min, x1_max = xs[:, 0].min(), xs[:, 0].max()
        x2_min, x2_max = xs[:, 1].min(), xs[:, 1].max()

        if abs(x1_max - x1_min) < 1e-9:
            x1_min -= 1.0
            x1_max += 1.0
        if abs(x2_max - x2_min) < 1e-9:
            x2_min -= 1.0
            x2_max += 1.0

        x1_vals = np.linspace(x1_min - padding, x1_max + padding, grid_size)
        x2_vals = np.linspace(x2_min - padding, x2_max + padding, grid_size)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)

        Z = np.zeros_like(X1)
        for i in range(grid_size):
            for j in range(grid_size):
                Z[i, j] = func(np.array([X1[i, j], X2[i, j]], dtype=float))

        x1_traj, x2_traj = xs[:, 0], xs[:, 1]

        # Contour plot
        self._style_2d_axes(contour_ax)
        contour = contour_ax.contour(X1, X2, Z, levels=levels, colors=PALETTE.text_muted, linewidths=0.8)
        contour_ax.contourf(X1, X2, Z, levels=levels, cmap="magma", alpha=0.45)

        contour_ax.plot(x1_traj, x2_traj, marker="o", linestyle="-", linewidth=1.2, markersize=4, color=_ACCENT)
        contour_ax.scatter(x1_traj[0], x2_traj[0], color=PALETTE.accent_alt, marker="s", s=50, zorder=5)
        contour_ax.scatter(x1_traj[-1], x2_traj[-1], color=PALETTE.accent, marker="*", s=120, zorder=6)

        contour_ax.set_xlabel("x₁")
        contour_ax.set_ylabel("x₂")
        contour_ax.set_title("Рівні функції та траєкторія")

        # Surface plot
        self._style_3d_axes(surface_ax)
        surface_ax.plot_surface(
            X1,
            X2,
            Z,
            rstride=2,
            cstride=2,
            cmap="magma",
            linewidth=0.2,
            antialiased=True,
            alpha=0.9,
        )

        z_traj = np.array([func(x) for x in xs], dtype=float)
        surface_ax.plot(x1_traj, x2_traj, z_traj, color=_ACCENT, marker="o", linewidth=2, markersize=5)
        surface_ax.set_xlabel("x₁")
        surface_ax.set_ylabel("x₂")
        surface_ax.set_zlabel("f(x₁, x₂)")
        surface_ax.set_title("Поверхня f(x₁, x₂) + траєкторія")

        self._redraw("contour")
        self._redraw("surface")
        self._set_page("contour")
