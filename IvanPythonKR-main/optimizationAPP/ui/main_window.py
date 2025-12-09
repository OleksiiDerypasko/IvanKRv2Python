"""
Головне вікно з акцентом на чистій темній темі та простому розміщенні:
    - зліва: компактна панель керування;
    - справа: карусель графіків над таблицею ітерацій.
"""

from __future__ import annotations
from typing import Optional

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QStatusBar,
    QLabel,
    QSplitter,
)

from core.iteration_result import IterationResult
from ui.control_panel import ControlPanelWidget, OptimizationConfig
from ui.table_view import IterationsTableWidget
from ui.plot_view import PlotView
from ui.dialogs import show_about
from ui.styles import apply_label_muted, MARGIN, SPACING


class MainWindow(QMainWindow):
    """
    Головне вікно GUI оптимізатора:
        - зліва: компактна панель керування;
        - справа: графіки (один за раз у каруселі) та таблиця ітерацій.
    """

    optimizationRequested = pyqtSignal(OptimizationConfig)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.setWindowTitle("Мінімізація функцій")
        self.resize(1400, 880)

        self._create_actions()
        self._create_menu()
        self._create_status_bar()
        self._create_content()
        self._connect_signals()

    # ----------------------------------------------------------------------
    # Menu + actions
    # ----------------------------------------------------------------------
    def _create_actions(self) -> None:
        self.action_exit = QAction("Вихід", self, shortcut="Ctrl+Q")
        self.action_about = QAction("Про програму", self)

    def _create_menu(self) -> None:
        menu = self.menuBar()
        menu.addMenu("Файл").addAction(self.action_exit)
        menu.addMenu("Довідка").addAction(self.action_about)

    def _create_status_bar(self) -> None:
        status = QStatusBar(self)
        self.setStatusBar(status)
        status.showMessage("Готово")

    # ----------------------------------------------------------------------
    # CONTENT LAYOUT
    # ----------------------------------------------------------------------
    def _create_content(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        root.setSpacing(SPACING)

        left_panel = self._build_left_panel()
        right_panel = self._build_right_panel()

        root.addWidget(left_panel, stretch=2)
        root.addWidget(right_panel, stretch=5)

        self.update_run_stats(None, None)

    # ------------------------------------------------------------------
    # LEFT CONTROL PANEL
    # ------------------------------------------------------------------
    def _build_left_panel(self) -> QWidget:
        widget = QWidget(self)
        widget.setMinimumWidth(360)

        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING)

        self.control_panel = ControlPanelWidget(widget)
        layout.addWidget(self.control_panel)
        layout.addStretch()

        return widget

    # ------------------------------------------------------------------
    # RIGHT PANEL – графіки + таблиця
    # ------------------------------------------------------------------
    def _build_right_panel(self) -> QWidget:
        widget = QWidget(self)
        widget.setMinimumWidth(760)

        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING)

        splitter = QSplitter(Qt.Orientation.Vertical, widget)
        splitter.setHandleWidth(6)

        # Верх: карусель графіків
        self.plot_view = PlotView(widget)
        splitter.addWidget(self.plot_view)

        # Низ: таблиця + статистика
        bottom = QWidget(widget)
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(SPACING)

        self.iterations_table = IterationsTableWidget(bottom)
        bottom_layout.addWidget(self.iterations_table)
        bottom_layout.addLayout(self._build_stats_row())

        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)

        return widget

    def _build_stats_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING)

        self.label_func_evals = QLabel("f evals: –")
        self.label_outer_iters = QLabel("outer iters: –")
        self.label_last_step = QLabel("step: –")

        for lbl in (self.label_func_evals, self.label_outer_iters, self.label_last_step):
            apply_label_muted(lbl)
            row.addWidget(lbl)

        row.addStretch()
        return row

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------
    def _connect_signals(self) -> None:
        self.action_exit.triggered.connect(self.close)
        self.action_about.triggered.connect(lambda: show_about(self))

        self.control_panel.exitRequested.connect(self.close)
        self.control_panel.clearRequested.connect(self._on_clear_requested)
        self.control_panel.runRequested.connect(self._on_run_requested)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    def _on_run_requested(self, cfg: OptimizationConfig) -> None:
        self.clear_results()
        self.statusBar().showMessage(
            f"Запуск: {cfg.function_key}, {cfg.method_key}, x0={cfg.x0.tolist()}"
        )
        self.optimizationRequested.emit(cfg)

    def _on_clear_requested(self) -> None:
        self.clear_results()
        self.statusBar().showMessage("Очищено")

    # ------------------------------------------------------------------
    # PUBLIC API (для app.py / движка)
    # ------------------------------------------------------------------
    def clear_results(self) -> None:
        """
        Очистити таблицю, скинути графік у плейсхолдер і статистику.
        """
        self.clear_iterations_table()
        self.plot_view.show_placeholder()
        self.update_run_stats(None, None)

    def clear_iterations_table(self) -> None:
        """
        Backward compatible метод для app.py.
        """
        if hasattr(self, "iterations_table"):
            self.iterations_table.clear_table()

    def add_iteration(
        self,
        iteration: IterationResult,
        step_value=None,
    ) -> None:
        """
        Додати ітерацію в таблицю.
        """
        self.iterations_table.add_iteration(iteration, step_value)

    def add_iteration_row(
        self,
        k,
        x,
        f_val,
        step_value=None,
    ) -> None:
        """
        Сумісний з попередніми версіями метод:
        відновлює IterationResult з примітивів.
        """
        it = IterationResult(
            index=int(k),
            x=np.array(x, dtype=float),
            f=float(f_val),
            step_norm=float(step_value or 0.0),
            meta={},
        )
        self.add_iteration(it, step_value)

    def update_fk_plot(self, iterations) -> None:
        """
        Оновити графік f(k) (викликається з движка).
        """
        self.plot_view.plot_fk(iterations)

    def update_contour_trajectory(self, func, iterations) -> None:
        """
        Оновити contour plot + траєкторію (викликається з движка).
        """
        self.plot_view.plot_contour_trajectory(func, iterations)

    def update_run_stats(
        self,
        func_evals: Optional[int],
        last_step: Optional[float],
        n_outer_iters: Optional[int] = None,
    ) -> None:
        """
        Оновити підписи під таблицею:
            - кількість викликів цільової функції;
            - кількість зовнішніх ітерацій;
            - фінальний крок пошуку.
        """
        self.label_func_evals.setText(
            f"f evals: {func_evals if func_evals is not None else '–'}"
        )
        self.label_outer_iters.setText(
            f"outer iters: {n_outer_iters if n_outer_iters is not None else '–'}"
        )
        self.label_last_step.setText(
            "step: –" if last_step is None else f"step: {last_step:.3e}"
        )


# ----------------------------------------------------------------------
# MAIN (ручний запуск)
# ----------------------------------------------------------------------
def main() -> None:
    import sys

    app = QApplication(sys.argv)

    from ui.styles import apply_app_style

    apply_app_style(app)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
