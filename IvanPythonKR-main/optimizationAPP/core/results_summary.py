"""
results_summary.py

Зведена таблиця результатів роботи різних методів оптимізації
для однієї обраної цільової функції.

Працює поверх об'єктів, які мають інтерфейс як OptimizationRunResult:
    - method_name
    - x_star
    - f_star
    - n_iter
    - func_evals
    - grad_evals
    - hess_evals
    - stopped_by
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ResultsSummary:
    """
    Зведення результатів роботи кількох методів оптимізації
    для однієї обраної цільової функції.

    Приклад використання:
        summary = ResultsSummary()
        summary.add_run(run_cauchy)
        summary.add_run(run_newton)
        rows = summary.as_rows()  # для GUI / pandas / CSV
    """
    runs: List[Any] = field(default_factory=list)

    def add_run(self, run: Any) -> None:
        """Додати результат одного методу до зведення."""
        self.runs.append(run)

    # ------------------------------------------------------------------
    # Перетворення в "табличний" вигляд
    # ------------------------------------------------------------------

    def as_rows(self) -> List[Dict[str, Any]]:
        """
        Повернути список dict-рядків, придатних для:
            - створення pandas.DataFrame,
            - виводу в GUI-таблицю,
            - експорту в CSV.

        Поля рядка:
            - method
            - x_star
            - f_star
            - n_iter
            - func_evals
            - grad_evals
            - hess_evals
            - stopped_by
        """
        rows: List[Dict[str, Any]] = []

        for run in self.runs:
            method_name = getattr(run, "method_name", "<unknown>")
            x_star = getattr(run, "x_star", None)
            f_star = getattr(run, "f_star", None)
            n_iter = getattr(run, "n_iter", None)
            func_evals = getattr(run, "func_evals", None)
            grad_evals = getattr(run, "grad_evals", None)
            hess_evals = getattr(run, "hess_evals", None)
            stopped_by = getattr(run, "stopped_by", None)

            if isinstance(x_star, np.ndarray):
                x_star_repr = x_star.tolist()
            else:
                x_star_repr = x_star

            rows.append(
                {
                    "method": method_name,
                    "x_star": x_star_repr,
                    "f_star": float(f_star) if f_star is not None else None,
                    "n_iter": int(n_iter) if n_iter is not None else None,
                    "func_evals": int(func_evals) if func_evals is not None else None,
                    "grad_evals": int(grad_evals) if grad_evals is not None else None,
                    "hess_evals": int(hess_evals) if hess_evals is not None else None,
                    "stopped_by": stopped_by,
                }
            )

        return rows

    # ------------------------------------------------------------------
    # Вибір "найкращого" методу
    # ------------------------------------------------------------------

    def best_by_f(self) -> Optional[Any]:
        """
        Повернути run з найменшим значенням f_star.
        Якщо список порожній або f_star не визначені — повертає None.
        """
        if not self.runs:
            return None

        best_run = None
        best_f = None

        for run in self.runs:
            f_star = getattr(run, "f_star", None)
            if f_star is None:
                continue
            f_val = float(f_star)
            if best_f is None or f_val < best_f:
                best_f = f_val
                best_run = run

        return best_run

    # ------------------------------------------------------------------
    # Опційно: повернути pandas.DataFrame
    # ------------------------------------------------------------------

    def to_dataframe(self):
        """
        Повернути pandas.DataFrame зі зведеною таблицею.

        Вимога: встановлений пакет pandas.
        """
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Для використання ResultsSummary.to_dataframe() "
                "потрібно встановити пакет 'pandas'."
            ) from exc

        return pd.DataFrame(self.as_rows())


__all__ = ["ResultsSummary"]
