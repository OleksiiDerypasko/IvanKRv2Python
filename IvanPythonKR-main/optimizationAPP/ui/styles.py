"""
Лаконічна темна тема для застосунку оптимізації.

Основні принципи:
    - темні фони, мінімум рамок;
    - один акцентний колір для дій та виділення;
    - спокійні вторинні відтінки для тексту й обводок.
"""

from __future__ import annotations
from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QTableWidget,
    QHeaderView,
    QGroupBox,
    QPushButton,
    QLabel,
)

MARGIN = 12
SPACING = 10
RADIUS = 6

FONT_FAMILY = "Montserrat"
FONT_SIZE = 10

@dataclass(frozen=True)
class AppPalette:
    background: str = "#0f1115"
    surface: str = "#171a1f"
    surface_alt: str = "#1f242b"

    text_main: str = "#e7ebf2"
    text_muted: str = "#9aa4b5"
    text_inverse: str = "#0f1115"

    accent: str = "#5fb3f7"  # спокійний блакитний акцент
    accent_alt: str = "#7dcfff"

    border: str = "#2a3039"
    border_soft: str = "#1c2027"

PALETTE = AppPalette()


# ---------------------------------------------------------------------------
# GLOBAL APP STYLESHEET (Qt Compatible)
# ---------------------------------------------------------------------------

def build_app_stylesheet() -> str:
    p = PALETTE

    return f"""
    QWidget {{
        background-color: {p.background};
        color: {p.text_main};
        font-family: "{FONT_FAMILY}";
        font-size: {FONT_SIZE}pt;
    }}

    QMainWindow {{
        background-color: {p.background};
    }}

    /* Panels */
    QGroupBox {{
        background-color: {p.surface};
        border: 1px solid {p.border_soft};
        border-radius: {RADIUS}px;
        margin-top: 14px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
        color: {p.accent};
        font-weight: 600;
    }}

    /* Menu */
    QMenuBar {{
        background-color: {p.surface};
        border-bottom: 1px solid {p.border};
    }}
    QMenuBar::item:selected {{
        background-color: {p.accent};
        color: {p.text_inverse};
        border-radius: 4px;
    }}

    QMenu {{
        background-color: {p.surface_alt};
        border: 1px solid {p.border};
    }}
    QMenu::item:selected {{
        background-color: {p.accent};
        color: {p.text_inverse};
    }}

    /* Status bar */
    QStatusBar {{
        background-color: {p.surface};
        color: {p.text_muted};
        border-top: 1px solid {p.border};
    }}

    /* Buttons */
    QPushButton {{
        background-color: {p.accent};
        color: {p.text_inverse};
        border-radius: {RADIUS}px;
        padding: 7px 14px;
        border: 1px solid {p.accent};
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: {p.accent_alt};
        border-color: {p.accent_alt};
    }}
    QPushButton:pressed {{
        background-color: {p.accent};
        border-color: {p.accent};
    }}

    /* Inputs */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {p.surface};
        border: 1px solid {p.border};
        border-radius: {RADIUS}px;
        padding: 6px 8px;
        color: {p.text_main};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border: 1px solid {p.accent};
    }}

    QLabel#functionPreview {{
        color: {p.text_muted};
        padding: 4px 0;
    }}

    QFrame#aboutColumn {{
        background-color: {p.surface};
        border: 1px solid {p.border_soft};
        border-radius: {RADIUS}px;
    }}

    /* Checkboxes */
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border-radius: 3px;
        border: 1px solid {p.border};
        background: {p.surface_alt};
    }}
    QCheckBox::indicator:checked {{
        background: {p.accent};
        border: 1px solid {p.accent};
    }}

    /* Tables */
    QTableWidget {{
        background-color: {p.surface};
        border: 1px solid {p.border};
        border-radius: {RADIUS}px;
        gridline-color: {p.border};
        alternate-background-color: {p.surface_alt};
        selection-background-color: {p.accent};
        selection-color: {p.text_inverse};
    }}

    QHeaderView::section {{
        background-color: {p.surface_alt};
        color: {p.text_main};
        padding: 6px;
        border: none;
        border-right: 1px solid {p.border};
        font-weight: 600;
    }}

    /* Tabs */
    QTabWidget::pane {{
        border: 1px solid {p.border};
        border-radius: {RADIUS}px;
        padding: 4px;
        background: {p.surface};
    }}
    QTabBar::tab {{
        padding: 6px 10px;
        border: 1px solid {p.border};
        border-bottom: none;
        background: {p.surface_alt};
        border-top-left-radius: {RADIUS}px;
        border-top-right-radius: {RADIUS}px;
        color: {p.text_muted};
    }}
    QTabBar::tab:selected {{
        background: {p.surface};
        color: {p.text_main};
        border-color: {p.accent};
    }}

    /* Scrollbars */
    QScrollBar:vertical {{
        background: transparent;
        width: 12px;
    }}
    QScrollBar::handle:vertical {{
        background: {p.surface_alt};
        border-radius: 6px;
        border: 1px solid {p.border};
    }}

    /* Tooltip */
    QToolTip {{
        background-color: {p.surface_alt};
        color: {p.text_main};
        border: 1px solid {p.border};
        padding: 6px;
        border-radius: 6px;
    }}
    """


# ---------------------------------------------------------------------------
# APPLY STYLE
# ---------------------------------------------------------------------------

def apply_app_style(app: QApplication) -> None:
    p = app.palette()

    p.setColor(QPalette.ColorRole.Window, QColor(PALETTE.background))
    p.setColor(QPalette.ColorRole.Base, QColor(PALETTE.surface))
    p.setColor(QPalette.ColorRole.Text, QColor(PALETTE.text_main))
    p.setColor(QPalette.ColorRole.Button, QColor(PALETTE.accent))

    app.setPalette(p)
    app.setFont(QFont(FONT_FAMILY, FONT_SIZE))
    app.setStyleSheet(build_app_stylesheet())


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def apply_groupbox_flat_style(group: QGroupBox):
    group.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)

def apply_table_style(table: QTableWidget):
    table.verticalHeader().setVisible(False)
    table.setAlternatingRowColors(True)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    table.horizontalHeader().setHighlightSections(False)
    table.horizontalHeader().setDefaultAlignment(
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
    )

def apply_button_secondary(btn: QPushButton):
    p = PALETTE
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {p.surface_alt};
            color: {p.text_main};
            border-radius: {RADIUS}px;
            padding: 7px 14px;
            border: 1px solid {p.border};
        }}
        QPushButton:hover {{
            border-color: {p.accent};
        }}
    """)

def apply_label_muted(lbl: QLabel):
    lbl.setStyleSheet(f"color: {PALETTE.text_muted};")

def apply_card_style(widget: QWidget):
    widget.setStyleSheet(f"""
        QWidget#{widget.objectName()} {{
            background-color: {PALETTE.surface};
            border-radius: {RADIUS}px;
            border: 1px solid {PALETTE.border};
        }}
    """)
