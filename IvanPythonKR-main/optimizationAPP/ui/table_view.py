"""
table_view.py

Таблиця ітерацій процесу мінімізації.

Функціонал:
    - відображає послідовність IterationResult;
    - колонки:
        k, α(step), x1, x2, f(x);
    - хелпери:
        clear_table()
        add_iteration(iteration, step_value)
        populate(iterations, step_getter=...)
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
)

from core.iteration_result import IterationResult
from .styles import MARGIN, SPACING, apply_table_style, PALETTE


class IterationsTableWidget(QWidget):
    """
    Обгортка над QTableWidget, адаптована під відображення
    табличного процесу мінімізації.

    Колонки:
        0: k          – номер ітерації
        1: α(step)    – крок (step size / ||Δx|| / alpha)
        2: x1         – перша координата
        3: x2         – друга координата
        4: f(x)       – значення цільової функції
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._build_ui()

    # ------------------------------------------------------------------
    # Побудова UI (нова компоновка)
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        root.setSpacing(SPACING)

        # Верхній заголовок над таблицею
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(SPACING)

        title = QLabel("Ітерації оптимізації", self)
        title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        subtitle = QLabel("k, крок α, координати x₁, x₂ та значення f(x)", self)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        subtitle.setStyleSheet(
            f"color: {PALETTE.text_muted}; font-size: 9pt;"
        )

        header_row.addWidget(title)
        header_row.addStretch(1)
        header_row.addWidget(subtitle)

        root.addLayout(header_row)

        # Безпосередньо таблиця
        self.table = QTableWidget(self)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["k", "α", "x₁", "x₂", "f(x)"])

        # Базовий стиль з вашого styles.py
        apply_table_style(self.table)

        # Додаткові налаштування, щоб вигляд був іншим
        h_header = self.table.horizontalHeader()
        h_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # k
        h_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)  # α
        h_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)          # x1
        h_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)          # x2
        h_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)          # f(x)

        # Вертикальний хедер як індекси рядків (можна вимкнути, якщо не треба)
        self.table.verticalHeader().setVisible(False)

        # Виділяємо рядки, не дозволяємо редагування напряму
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        # Трохи зменшимо висоту рядків
        self.table.verticalHeader().setDefaultSectionSize(22)

        root.addWidget(self.table)

    # ------------------------------------------------------------------
    # Публічне API
    # ------------------------------------------------------------------

    def clear_table(self) -> None:
        """Очистити всі рядки таблиці."""
        self.table.setRowCount(0)

    def add_iteration(
        self,
        iteration: IterationResult,
        step_value: Optional[float] = None,
    ) -> None:
        """
        Додати один рядок у таблицю за IterationResult.

        Parameters
        ----------
        iteration : IterationResult
            Об'єкт з index, x, f, step_norm, meta.
        step_value : Optional[float]
            Значення кроку. Якщо None — буде використано iteration.step_norm.
        """
        row = self.table.rowCount()
        self.table.insertRow(row)

        if step_value is None:
            step_value = float(iteration.step_norm)

        x = iteration.x
        f_val = float(iteration.f)

        def _item(text: Any, align: Qt.AlignmentFlag) -> QTableWidgetItem:
            it = QTableWidgetItem(str(text))
            it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
            it.setTextAlignment(align | Qt.AlignmentFlag.AlignVCenter)
            return it

        # k — по центру
        self.table.setItem(row, 0, _item(iteration.index, Qt.AlignmentFlag.AlignHCenter))

        # α(step) — по центру, у науковому форматі
        self.table.setItem(row, 1, _item(f"{step_value:.3e}", Qt.AlignmentFlag.AlignHCenter))

        # x1, x2, f(x) — праве вирівнювання
        if x.size >= 1:
            self.table.setItem(row, 2, _item(f"{x[0]:.6f}", Qt.AlignmentFlag.AlignRight))
        else:
            self.table.setItem(row, 2, _item("—", Qt.AlignmentFlag.AlignHCenter))

        if x.size >= 2:
            self.table.setItem(row, 3, _item(f"{x[1]:.6f}", Qt.AlignmentFlag.AlignRight))
        else:
            self.table.setItem(row, 3, _item("—", Qt.AlignmentFlag.AlignHCenter))

        self.table.setItem(row, 4, _item(f"{f_val:.6e}", Qt.AlignmentFlag.AlignRight))

    def populate(
        self,
        iterations: Iterable[IterationResult],
        step_getter: Optional[Callable[[IterationResult], Optional[float]]] = None,
    ) -> None:
        """
        Повністю перезаповнити таблицю трасою ітерацій.

        Parameters
        ----------
        iterations : Iterable[IterationResult]
            Список/ітератор з IterationResult.
        step_getter : Optional[Callable[[IterationResult], Optional[float]]]
            Функція, яка повертає "крок" для кожної ітерації.
            Якщо None — крок береться з iteration.meta["alpha"]
            (якщо є), інакше з iteration.step_norm.
        """
        self.clear_table()

        if step_getter is None:
            def step_getter_default(it: IterationResult) -> Optional[float]:
                # Якщо метод зберігає alpha в meta — використовуємо його
                if "alpha" in (it.meta or {}):
                    return float(it.meta["alpha"])
                return float(it.step_norm)

            step_getter = step_getter_default

        for it in iterations:
            step_value = step_getter(it)
            self.add_iteration(it, step_value=step_value)
