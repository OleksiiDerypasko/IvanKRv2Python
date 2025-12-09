"""
ui/dialogs.py

Стандартні діалоги для GUI-застосунку:

    - show_error     – повідомлення про помилку
    - show_info      – інформаційне повідомлення
    - show_about     – вікно "Про програму"
    - show_summary   – діалог зі зведеною таблицею ResultsSummary
"""

from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QMessageBox,
    QDialog,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QDialogButtonBox,
    QHeaderView,   # <-- ДОДАНО
    QFrame,
)

from core.results_summary import ResultsSummary
from .styles import MARGIN, SPACING, apply_label_muted, apply_table_style


# ---------------------------------------------------------------------------
# Простi діалоги: помилка / інформація / about
# ---------------------------------------------------------------------------


def show_error(parent: Optional[QWidget], message: str, title: str = "Помилка") -> None:
    """
    Показати діалог помилки з червоною іконкою.
    """
    dlg = QMessageBox(parent)
    dlg.setIcon(QMessageBox.Icon.Critical)
    dlg.setWindowTitle(title)
    dlg.setText(message)
    dlg.setStandardButtons(QMessageBox.StandardButton.Ok)
    dlg.exec()


def show_info(parent: Optional[QWidget], title: str, message: str) -> None:
    """
    Показати інформаційний діалог (з синьою "i").
    """
    dlg = QMessageBox(parent)
    dlg.setIcon(QMessageBox.Icon.Information)
    dlg.setWindowTitle(title)
    dlg.setText(message)
    dlg.setStandardButtons(QMessageBox.StandardButton.Ok)
    dlg.exec()


def show_about(parent: Optional[QWidget]) -> None:
    """
    Показати діалог "Про програму".
    """
    dlg = AboutDialog(parent)
    dlg.exec()


def _humanize_stop_reason(code: str | None) -> str:
    """
    Перетворити машинний код причини зупинки на людське пояснення.
    """
    if not code:
        return "Невідомо"

    mapping = {
        "step_norm": "Крок став досить малим (критерій норми кроку)",
        "f_change": "Зміна значення функції стала малою",
        "grad_norm": "Норма градієнта стала малою",
        "max_iter": "Досягнуто граничної кількості ітерацій",
        "nan": "Обчислення зупинено через некоректні значення (NaN/∞)",
        "user_stop": "Зупинено користувачем",
    }

    # Якщо код невідомий — все одно щось показати
    return mapping.get(code, f"Інша причина ({code})")


# ---------------------------------------------------------------------------
# Діалог зі зведеною таблицею ResultsSummary
# ---------------------------------------------------------------------------


class SummaryDialog(QDialog):
    """
    Діалог, що показує зведену таблицю результатів для всіх методів
    (ResultsSummary).
    """

    def __init__(self, parent: Optional[QWidget], summary: ResultsSummary) -> None:
        super().__init__(parent)
        self.summary = summary

        self.setWindowTitle("Зведена таблиця результатів")
        self.setModal(True)
        self.resize(800, 400)

        self._build_ui()
        self._populate()

    # ------------------------- UI -------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        layout.setSpacing(SPACING)

        self.label_title = QLabel(
            "Результати мінімізації для всіх методів\n"
            "(обрана функція та однакові стартові умови)",
            self,
        )
        self.label_title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.label_title.setWordWrap(True)

        self.table = QTableWidget(self)
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(
            [
                "Метод",
                "f*",
                "x*",
                "Ітерацій",
                "Виклики f",
                "Виклики grad",
                "Виклики hess",
                "Причина зупинки",
            ]
        )
        apply_table_style(self.table)

        # --- ГОЛОВНЕ: дозволь користувачу розширювати колонки мишкою ---
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(False)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok,
            orientation=Qt.Orientation.Horizontal,
            parent=self,
        )
        buttons.accepted.connect(self.accept)

        layout.addWidget(self.label_title)
        layout.addWidget(self.table)
        layout.addWidget(buttons)

    # --------------------- Заповнення даних ---------------------

    def _populate(self) -> None:
        """
        Заповнити таблицю даними з ResultsSummary.

        Очікується, що summary.as_rows() повертає список словників:
            {
                "method_name": str,
                "f_star": float,
                "x_star": np.ndarray | list | tuple,
                "n_iter": int,
                "func_evals": int,
                "grad_evals": int,
                "hess_evals": int,
                "stopped_by": str,
            }
        """
        rows: list[dict[str, Any]] = self.summary.as_rows()
        self.table.setRowCount(len(rows))

        for row_idx, row in enumerate(rows):
            method_name = row.get("method_name", row.get("method", ""))
            f_star = row.get("f_star", None)
            x_star = row.get("x_star", None)
            n_iter = row.get("n_iter", None)
            func_evals = row.get("func_evals", None)
            grad_evals = row.get("grad_evals", None)
            hess_evals = row.get("hess_evals", None)
            stopped_by = row.get("stopped_by", "")

            # x* форматуємо як [x1, x2, ...]
            if x_star is None:
                x_star_str = ""
            else:
                try:
                    xs = list(x_star)
                    x_star_str = "[" + ", ".join(f"{v:.4f}" for v in xs) + "]"
                except Exception:
                    x_star_str = str(x_star)

            def _item(val: Any) -> QTableWidgetItem:
                it = QTableWidgetItem(str(val))
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                return it

            self.table.setItem(row_idx, 0, _item(method_name))

            if f_star is None:
                self.table.setItem(row_idx, 1, _item(""))
            else:
                self.table.setItem(row_idx, 1, _item(f"{float(f_star):.6e}"))

            self.table.setItem(row_idx, 2, _item(x_star_str))
            self.table.setItem(
                row_idx, 3, _item("" if n_iter is None else n_iter)
            )
            self.table.setItem(
                row_idx, 4, _item("" if func_evals is None else func_evals)
            )
            self.table.setItem(
                row_idx, 5, _item("" if grad_evals is None else grad_evals)
            )
            self.table.setItem(
                row_idx, 6, _item("" if hess_evals is None else hess_evals)
            )
            stopped_text = _humanize_stop_reason(stopped_by)
            self.table.setItem(row_idx, 7, _item(stopped_text))


def show_summary(parent: Optional[QWidget], summary: ResultsSummary) -> None:
    """
    Зручна обгортка для показу SummaryDialog.
    """
    dlg = SummaryDialog(parent, summary)
    dlg.exec()


def show_error_dialog(parent: Optional[QWidget], title: str, message: str) -> None:
    """
    Сумісна з app.py обгортка навколо show_error
    з сигнатурою (parent, title, message).
    """
    show_error(parent, message=message, title=title)


class AboutDialog(QDialog):
    """
    Розширене вікно "Про програму" з колонкою, що займає всю висоту діалогу.
    """

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Про програму")
        self.setModal(True)

        if parent is not None:
            self.resize(int(parent.width() * 0.55), parent.height())
        else:
            self.resize(760, 640)

        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        root.setSpacing(SPACING)

        column = QFrame(self)
        column.setObjectName("aboutColumn")
        column_layout = QVBoxLayout(column)
        column_layout.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        column_layout.setSpacing(SPACING)

        title = QLabel("<b>Мінімізація багатовимірних унімодальних функцій</b>", column)
        title.setWordWrap(True)

        subtitle = QLabel("Навчальний застосунок для дослідження методів оптимізації.", column)
        subtitle.setWordWrap(True)
        apply_label_muted(subtitle)

        separator = QFrame(column)
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)

        description = QLabel(
            (
                "<p>Програма демонструє роботу методів мінімізації функцій двох змінних. "
                "Візуалізація включає таблицю ітерацій, графік f(k) та траєкторію на "
                "рівневих лініях цільової функції.</p>"
            ),
            column,
        )
        description.setWordWrap(True)

        methods = QLabel(
            """
            <p><b>Реалізовані методи:</b></p>
            <ul>
                <li>Метод Коші (градієнтний спуск)</li>
                <li>Метод Флетчера–Рівза</li>
                <li>Метод Полака–Ріб’єра</li>
                <li>Метод Ньютона</li>
                <li>Метод Нелдера–Міда</li>
                <li>Метод Хука–Дживса</li>
            </ul>
            """,
            column,
        )
        methods.setWordWrap(True)

        footer = QLabel(
            "<p><b>Версія:</b> 1.0.0<br/>ЛНУ ім. Івана Франка</p>",
            column,
        )
        footer.setWordWrap(True)
        apply_label_muted(footer)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok,
            orientation=Qt.Orientation.Horizontal,
            parent=self,
        )
        buttons.accepted.connect(self.accept)

        column_layout.addWidget(title)
        column_layout.addWidget(subtitle)
        column_layout.addWidget(separator)
        column_layout.addWidget(description)
        column_layout.addWidget(methods)
        column_layout.addStretch(1)
        column_layout.addWidget(footer)
        column_layout.addWidget(buttons, alignment=Qt.AlignmentFlag.AlignRight)

        root.addWidget(column, stretch=1)
        root.addStretch(1)
