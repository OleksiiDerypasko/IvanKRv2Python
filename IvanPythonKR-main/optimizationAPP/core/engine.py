"""
engine.py

Універсальний ітераційний двигун для запуску методів оптимізації (Optimizer).

Функціонал:
    - виконує цикл x_{k+1} = step(x_k) для довільного Optimizer;
    - формує трасу ітерацій (для таблиць і графіків);
    - рахує кількість викликів цільової функції, градієнта та Гессіана;
    - фіксує причину зупинки (step_norm, зміна функції, внутрішній критерій методу, max_iter);
    - підтримує callback для оновлення GUI / логів на кожній ітерації.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import numpy as np

from .optimizer_base import Optimizer, StepResult
from .functions import ArrayLike
from .iteration_result import IterationResult


@dataclass
class OptimizationRunResult:
    """
    Підсумок одного запуску оптимізації.

    Атрибути:
        method_name   - назва методу (Optimizer.name).
        iterations    - список IterationResult (трасa процесу).
        x_star        - знайдена точка мінімуму (остання в трасі).
        f_star        - значення f(x_star).
        n_iter        - кількість виконаних ітерацій (без урахування k=0).
        func_evals    - кількість викликів цільової функції.
        grad_evals    - кількість викликів градієнта.
        hess_evals    - кількість викликів Гессіана.
        stopped_by    - причина зупинки ("step_norm", "f_change", "max_iter", "method:...").
    """
    method_name: str
    iterations: List[IterationResult]
    x_star: np.ndarray
    f_star: float
    n_iter: int
    func_evals: int
    grad_evals: int
    hess_evals: int
    stopped_by: str


# Тип callback'а для GUI/логів
IterationCallback = Callable[[IterationResult], None]


class OptimizationEngine:
    """
    Движок, який керує ітераційним процесом для заданого Optimizer.

    Налаштування за замовчуванням (можуть бути переозначені у run()):
        tol_step  : поріг для норми кроку (default: 1e-6)
        tol_f     : поріг для зміни функції |f_{k+1} - f_k| (default: 1e-9)
        max_iter  : максимальна кількість ітерацій (default: 200)
    """

    def __init__(
        self,
        tol_step: float = 1e-6,
        tol_f: float = 1e-9,
        max_iter: int = 200,
    ) -> None:
        self.tol_step_default = tol_step
        self.tol_f_default = tol_f
        self.max_iter_default = max_iter

    def run(
        self,
        optimizer: Optimizer,
        x0: ArrayLike,
        max_iter: Optional[int] = None,
        tol_step: Optional[float] = None,
        tol_f: Optional[float] = None,
        callback: Optional[IterationCallback] = None,
    ) -> OptimizationRunResult:
        """
        Запустити процес оптимізації.
        """
        x0 = np.asarray(x0, dtype=float)

        # Параметри зупинки
        max_iter = max_iter if max_iter is not None else self.max_iter_default
        tol_step = tol_step if tol_step is not None else self.tol_step_default
        tol_f = tol_f if tol_f is not None else self.tol_f_default

        # Скидаємо стан методу та ініціалізуємо
        optimizer.reset()
        optimizer.initialize(x0)

        iterations: List[IterationResult] = []

        # Початкова точка (k = 0)
        f0 = optimizer.eval_f(x0)
        rec0 = IterationResult(
            index=0,
            x=x0.copy(),
            f=f0,
            step_norm=0.0,
            meta={"initial": True},
        )
        iterations.append(rec0)
        if callback is not None:
            callback(rec0)

        x_k = x0.copy()
        f_k = f0
        stopped_by: str = "max_iter"  # значення за замовчуванням, якщо вийдемо по циклу

        # Основний ітераційний цикл
        for k in range(1, max_iter + 1):
            # Один крок методу
            step_res: StepResult = optimizer.step(x_k)
            x_next = step_res.x_new
            f_next = step_res.f_new
            step_norm = float(step_res.step_norm)
            meta = dict(step_res.meta or {})
            meta.setdefault("iteration_internal", k)

            # Нова ітерація для трасування
            rec = IterationResult(
                index=k,
                x=x_next.copy(),
                f=f_next,
                step_norm=step_norm,
                meta=meta,
            )
            iterations.append(rec)

            if callback is not None:
                callback(rec)

            # Чи метод сам попросив зупинити процес?
            method_stopped = meta.get("stopped_by")

            if method_stopped is not None:
                stopped_by = f"method:{method_stopped}"
                break

            # ---- Налаштування толерансів залежно від типу методу ----
            # Для методів типу Nelder–Mead, які передають simplex_diameter,
            # використовуємо його замість step_norm і ігноруємо tol_f.
            has_simplex_diameter = "simplex_diameter" in meta

            effective_step_norm = step_norm
            if has_simplex_diameter:
                effective_step_norm = float(meta["simplex_diameter"])

            # Критерій зупинки по "кроці"
            if effective_step_norm < tol_step:
                stopped_by = "step_norm"
                break

            # Критерій по зміні функції:
            #   - застосовуємо тільки для методів БЕЗ simplex_diameter,
            #     щоб не рвати процес Nelder–Mead на ранніх стадіях.
            if (not has_simplex_diameter) and abs(f_next - f_k) < tol_f:
                stopped_by = "f_change"
                break

            # Переходимо до наступної ітерації
            x_k = x_next
            f_k = f_next
        else:
            # Якщо цикл не break — вийшли по max_iter
            stopped_by = "max_iter"

        # Результати
        last_rec = iterations[-1]

        result = OptimizationRunResult(
            method_name=optimizer.name,
            iterations=iterations,
            x_star=last_rec.x.copy(),
            f_star=float(last_rec.f),
            n_iter=len(iterations) - 1,  # без k=0
            func_evals=optimizer.func_evals,
            grad_evals=optimizer.grad_evals,
            hess_evals=optimizer.hess_evals,
            stopped_by=stopped_by,
        )

        return result


__all__ = [
    "IterationResult",
    "OptimizationRunResult",
    "OptimizationEngine",
]
