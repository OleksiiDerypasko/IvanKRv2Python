"""
control_panel.py

Панель керування для GUI:
    - вибір функції;
    - вибір методу;
    - вибір методу лінійного пошуку;
    - початкова точка x0 = (x1, x2);
    - eps (точність);
    - max_iter;
    - опція "Запустити всі методи";
    - кнопки: Запустити, Очистити, Вихід.

Видає назовні:
    - сигнал runRequested(OptimizationConfig)
    - сигнал clearRequested()
    - сигнал exitRequested()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QComboBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
)

from .styles import (
    MARGIN,
    SPACING,
    apply_groupbox_flat_style,
    apply_button_secondary,
)


# ---------------------------------------------------------------------------
# Конфігурація запуску оптимізації
# ---------------------------------------------------------------------------

@dataclass
class OptimizationConfig:
    function_key: str
    method_key: str
    x0: np.ndarray
    eps: float
    max_iter: int
    run_all_methods: bool = False
    # ключ методу лінійного пошуку:
    #   "default", "dichotomy", "interval_halving",
    #   "golden_section", "cubic4"
    line_search_key: str = "default"


# ---------------------------------------------------------------------------
# Віджет панелі керування
# ---------------------------------------------------------------------------

class ControlPanelWidget(QWidget):
    """
    Ліва панель керування оптимізацією.

    Сигнали:
        runRequested(OptimizationConfig)  – натиснуто "Запустити"
        clearRequested()                  – натиснуто "Очистити"
        exitRequested()                   – натиснуто "Вихід"
    """

    runRequested = pyqtSignal(OptimizationConfig)
    clearRequested = pyqtSignal()
    exitRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # Побудова UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.setObjectName("controlPanel")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        main_layout.setSpacing(SPACING)

        # ------------------------------------------------------------------
        # Блок 1. Цільова функція та стартова точка
        # ------------------------------------------------------------------
        self.problem_group = QGroupBox("Цільова функція та старт", self)
        apply_groupbox_flat_style(self.problem_group)

        problem_layout = QVBoxLayout(self.problem_group)
        problem_layout.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        problem_layout.setSpacing(SPACING)

        lbl_func = QLabel("Функція:", self.problem_group)
        lbl_func.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.combo_function = QComboBox(self.problem_group)
        self.combo_function.addItems(
            [
                "f₁(x₁, x₂) = (12 + x₁² + (1 + x₂²)/x₁² + ((x₁x₂)² + 100)/(x₁x₂)⁴) / 10",
                "f₂(x₁, x₂) = (x₁ − x₂)² + (x₁ + x₂ − 10)² / 9",
                "f₃(x₁, x₂) = 5·(x₂ − 4x₁³ + 3x₁)² + (x₁ + 1)²",
                "f₄(x₁, x₂) = 5·(x₂ − 4x₁³ + 3x₁)² + (x₁ − 1)²",
                "f₅(x₁, x₂) = 100·(x₂ − x₁³ + x₁)² + (x₁ − 1)²",
                "f₆(x₁, x₂) = (0.01·(x₁ − 3))² − (x₂ − x₁) + exp(20·(x₂ − x₁))",
                "f₇(x₁, x₂) = 100·(x₂ − x₁²)² + (1 − x₁)²   [Розенброк]",
                "f₈(x₁, x₂) = (x₁ − 4)² + (x₂ − 4)²",
            ]
        )

        x_row = QHBoxLayout()
        x_row.setSpacing(SPACING)

        lbl_x0 = QLabel("x₀:", self.problem_group)
        lbl_x0.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.input_x1 = QDoubleSpinBox(self.problem_group)
        self.input_x1.setRange(-1e6, 1e6)
        self.input_x1.setDecimals(6)
        self.input_x1.setValue(0.0)

        self.input_x2 = QDoubleSpinBox(self.problem_group)
        self.input_x2.setRange(-1e6, 1e6)
        self.input_x2.setDecimals(6)
        self.input_x2.setValue(0.0)

        x_row.addWidget(lbl_x0)
        x_row.addWidget(QLabel("x₁:", self.problem_group))
        x_row.addWidget(self.input_x1)
        x_row.addSpacing(SPACING)
        x_row.addWidget(QLabel("x₂:", self.problem_group))
        x_row.addWidget(self.input_x2)

        problem_layout.addWidget(lbl_func)
        problem_layout.addWidget(self.combo_function)
        problem_layout.addLayout(x_row)

        main_layout.addWidget(self.problem_group)

        # ------------------------------------------------------------------
        # Блок 2. Метод оптимізації
        # ------------------------------------------------------------------
        self.method_group = QGroupBox("Метод оптимізації", self)
        apply_groupbox_flat_style(self.method_group)

        method_layout = QVBoxLayout(self.method_group)
        method_layout.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        method_layout.setSpacing(SPACING)

        lbl_method = QLabel("Метод:", self.method_group)
        lbl_method.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.combo_method = QComboBox(self.method_group)
        self.combo_method.addItems(
            [
                "Метод Коші",
                "Метод Флетчера–Рівза",
                "Метод Полака–Ріб’єра",
                "Метод Ньютона",
                "Метод Нелдера–Міда",
                "Метод Хука–Дживса",
            ]
        )

        lbl_ls = QLabel("Line search:", self.method_group)
        lbl_ls.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.combo_line_search = QComboBox(self.method_group)
        self.combo_line_search.addItems(
            [
                "Адаптація кроку",
                "Дихотомія",
                "Розподіл інтервалу навпіл",
                "Золотий переріз",
                "Кубічна інтерполяція (4 точки)",
            ]
        )

        self.check_run_all = QCheckBox(
            "Запустити всі методи для обраної функції",
            self.method_group,
        )

        method_layout.addWidget(lbl_method)
        method_layout.addWidget(self.combo_method)
        method_layout.addWidget(lbl_ls)
        method_layout.addWidget(self.combo_line_search)
        method_layout.addWidget(self.check_run_all)

        main_layout.addWidget(self.method_group)

        # ------------------------------------------------------------------
        # Блок 3. Параметри точності та ітерацій
        # ------------------------------------------------------------------
        self.params_group = QGroupBox("Параметри точності та ітерацій", self)
        apply_groupbox_flat_style(self.params_group)

        params_layout = QVBoxLayout(self.params_group)
        params_layout.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        params_layout.setSpacing(SPACING)

        params_row = QHBoxLayout()
        params_row.setSpacing(SPACING)

        self.input_eps = QDoubleSpinBox(self.params_group)
        self.input_eps.setRange(1e-15, 1e3)
        self.input_eps.setDecimals(10)
        self.input_eps.setValue(1e-6)

        self.input_max_iter = QSpinBox(self.params_group)
        self.input_max_iter.setRange(1, 10000)
        self.input_max_iter.setValue(300)

        params_row.addWidget(QLabel("eps:", self.params_group))
        params_row.addWidget(self.input_eps)
        params_row.addSpacing(SPACING * 2)
        params_row.addWidget(QLabel("max_iter:", self.params_group))
        params_row.addWidget(self.input_max_iter)
        params_row.addStretch(1)

        params_layout.addLayout(params_row)

        main_layout.addWidget(self.params_group)

        # ------------------------------------------------------------------
        # Нижній ряд кнопок
        # ------------------------------------------------------------------
        buttons_row = QHBoxLayout()
        buttons_row.setContentsMargins(0, SPACING, 0, 0)
        buttons_row.setSpacing(SPACING)

        self.button_run = QPushButton("Запустити", self)
        self.button_clear = QPushButton("Очистити", self)
        self.button_exit = QPushButton("Вихід", self)

        apply_button_secondary(self.button_clear)
        apply_button_secondary(self.button_exit)

        buttons_row.addWidget(self.button_run)
        buttons_row.addWidget(self.button_clear)
        buttons_row.addStretch(1)
        buttons_row.addWidget(self.button_exit)

        main_layout.addLayout(buttons_row)
        main_layout.addStretch(1)

    # ------------------------------------------------------------------
    # Сигнали
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self.button_run.clicked.connect(self._on_run_clicked)
        self.button_clear.clicked.connect(self._on_clear_clicked)
        self.button_exit.clicked.connect(self._on_exit_clicked)

    # ------------------------------------------------------------------
    # Внутрішні хелпери
    # ------------------------------------------------------------------

    def _get_selected_function_key(self) -> str:
        """
        Повертає ключ функції (f1..f8) за індексом combobox.
        """
        index = self.combo_function.currentIndex()
        return f"f{index + 1}"

    def _get_selected_method_key(self) -> str:
        """
        Повертає ключ методу:
            "cauchy", "fletcher_reeves", "polak_ribiere",
            "newton", "nelder_mead", "hook_jeeves"
        """
        index = self.combo_method.currentIndex()
        keys = [
            "cauchy",
            "fletcher_reeves",
            "polak_ribiere",
            "newton",
            "nelder_mead",
            "hook_jeeves",
        ]
        return keys[index]

    def _get_selected_line_search_key(self) -> str:
        """
        Повертає ключ методу лінійного пошуку для core:
            "default", "dichotomy", "interval_halving",
            "golden_section", "cubic4"
        """
        index = self.combo_line_search.currentIndex()
        keys = [
            "default",
            "dichotomy",
            "interval_halving",
            "golden_section",
            "cubic4",
        ]
        return keys[index]

    def build_config(self) -> OptimizationConfig:
        """
        Зібрати OptimizationConfig з поточного стану контролів.
        """
        function_key = self._get_selected_function_key()
        method_key = self._get_selected_method_key()
        line_search_key = self._get_selected_line_search_key()

        x0 = np.array(
            [self.input_x1.value(), self.input_x2.value()],
            dtype=float,
        )

        cfg = OptimizationConfig(
            function_key=function_key,
            method_key=method_key,
            x0=x0,
            eps=float(self.input_eps.value()),
            max_iter=int(self.input_max_iter.value()),
            run_all_methods=self.check_run_all.isChecked(),
            line_search_key=line_search_key,
        )
        return cfg

    # ------------------------------------------------------------------
    # Обробники кнопок
    # ------------------------------------------------------------------

    def _on_run_clicked(self) -> None:
        cfg = self.build_config()
        self.runRequested.emit(cfg)

    def _on_clear_clicked(self) -> None:
        self.clearRequested.emit()

    def _on_exit_clicked(self) -> None:
        self.exitRequested.emit()
