"""
hook_jeeves.py

Реалізація методу Хука–Дживса як стратегії Optimizer.

Ідея (pattern search):
    - Є базова точка x_B (base point).
    - Виконуємо exploratory search навколо x_B (по кожній координаті ±step):
          отримуємо x_E (exploratory point).
    - Якщо x_E покращує значення функції:
          робимо pattern move: x_P = x_E + (x_E - x_B).
          Якщо x_P ще краще — нова база x_B = x_P,
          інакше база x_B = x_E.
    - Якщо exploratory не покращує x_B:
          зменшуємо крок step = step * reduction_factor.

Один виклик step() = один повний цикл:
    Base → Exploratory → Pattern/ReduceStep.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .optimizer_base import Optimizer, StepResult
from .functions import ScalarFunction, VectorFunction, MatrixFunction, ArrayLike


class HookJeevesMethod(Optimizer):
    """
    Метод Хука–Дживса (pattern search, direct search).

    Особливості:
        - не використовує градієнт і Гессіан (тільки значення функції);
        - метод веде внутрішню базову точку (base_point) і крок step_size;
        - один step() = одна ітерація з exploratory та pattern move.

    Налаштування (options):
        initial_step      : початковий крок step_size (default: 1.0)
        reduction_factor  : множник для зменшення кроку (0 < r < 1, default: 0.5)
        min_step          : мінімально допустимий крок (default: 1e-6)
        max_exploratory   : максимум циклів exploratory на ітерацію (для безпеки, default: None)
    """

    requires_gradient: bool = False
    requires_hessian: bool = False

    def __init__(
        self,
        func: ScalarFunction,
        grad: Optional[VectorFunction] = None,   # ігнорується
        hess: Optional[MatrixFunction] = None,   # ігнорується
        options: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            func=func,
            grad=None,
            hess=None,
            options=options,
            name=name or "Hooke–Jeeves (pattern search)",
        )

    # ------------------------------------------------------------------
    # Ініціалізація
    # ------------------------------------------------------------------

    def initialize(self, x0: ArrayLike) -> None:
        """
        Ініціалізувати базову точку та крок пошуку.
        """
        super().initialize(x0)

        x0 = np.asarray(x0, dtype=float)
        initial_step: float = float(self.options.get("initial_step", 1.0))

        self.state["base_point"] = x0
        self.state["f_base"] = self.eval_f(x0)
        self.state["step_size"] = initial_step
        self.state["iteration"] = 0

    def _ensure_initialized(self, x_k: np.ndarray) -> None:
        """
        Якщо алгоритм ще не ініціалізовано — ініціалізуємо навколо x_k.
        """
        if "base_point" not in self.state or "f_base" not in self.state:
            self.initialize(x_k)

    # ------------------------------------------------------------------
    # Exploratory search
    # ------------------------------------------------------------------

    def _exploratory_search(
        self,
        x_start: np.ndarray,
        f_start: float,
        step: float,
        max_exploratory: Optional[int] = None,
    ) -> Tuple[np.ndarray, float, bool, int]:
        """
        Exploratory search навколо точки x_start з кроком step.

        Для кожної координати i:
            1) пробуємо x + step по осі i;
            2) якщо покращує — приймаємо;
               інакше пробуємо x - step по осі i;
            3) якщо покращує — приймаємо;
               інакше залишаємо як було.

        Повертає:
            x_new           - знайдена точка
            f_new           - f(x_new)
            improved        - чи стало краще, ніж у x_start
            evals           - кількість додаткових оцінок функції
        """
        x = x_start.copy()
        f_x = f_start
        n = x.size
        improved = False
        evals = 0

        for i in range(n):
            if max_exploratory is not None and evals >= max_exploratory:
                break

            # Пробуємо +step
            x_try = x.copy()
            x_try[i] += step
            f_try = self.eval_f(x_try)
            evals += 1

            if f_try < f_x:
                x = x_try
                f_x = f_try
                improved = True
                continue

            # Пробуємо -step
            if max_exploratory is not None and evals >= max_exploratory:
                break

            x_try = x.copy()
            x_try[i] -= step
            f_try = self.eval_f(x_try)
            evals += 1

            if f_try < f_x:
                x = x_try
                f_x = f_try
                improved = True

        return x, f_x, improved, evals

    # ------------------------------------------------------------------
    # Один крок методу Хука–Дживса
    # ------------------------------------------------------------------

    def _step_impl(self, x_k: np.ndarray) -> StepResult:
        """
        Один крок методу Хука–Дживса.

        Примітка:
            x_k передається ззовні (движком), але алгоритм фактично працює
            з внутрішнім base_point. Якщо initialize() ще не викликано,
            використовуємо x_k як початкову базу.
        """
        self._ensure_initialized(x_k)

        reduction_factor: float = float(self.options.get("reduction_factor", 0.5))
        min_step: float = float(self.options.get("min_step", 1e-6))
        max_exploratory_opt = self.options.get("max_exploratory", None)
        max_exploratory: Optional[int] = (
            int(max_exploratory_opt) if max_exploratory_opt is not None else None
        )

        base_point: np.ndarray = self.state["base_point"]
        f_base: float = float(self.state["f_base"])
        step_size: float = float(self.state["step_size"])
        iteration: int = int(self.state.get("iteration", 0))

        # Якщо крок вже дуже малий — вважаємо, що ми майже зійшлися
        if step_size < min_step:
            self.state["iteration"] = iteration + 1
            return StepResult(
                x_new=base_point.copy(),
                f_new=f_base,
                step_norm=0.0,
                meta={
                    "step_size": step_size,
                    "f_base": f_base,
                    "iteration": iteration,
                    "stopped_by": "step_below_min",
                    "exploratory_evals": 0,
                    "pattern_applied": False,
                    "step_type": "none",
                },
            )

        # -------------------------------
        # 1. Exploratory search навколо base_point
        # -------------------------------
        x_expl, f_expl, improved, evals = self._exploratory_search(
            x_start=base_point,
            f_start=f_base,
            step=step_size,
            max_exploratory=max_exploratory,
        )

        if not improved:
            # Немає покращення — зменшуємо крок
            new_step_size = step_size * reduction_factor
            self.state["step_size"] = new_step_size
            self.state["iteration"] = iteration + 1

            meta = {
                "step_size": new_step_size,
                "f_base": f_base,
                "iteration": iteration,
                "stopped_by": None,
                "exploratory_evals": evals,
                "pattern_applied": False,
                "step_type": "reduce_step",
            }

            return StepResult(
                x_new=base_point.copy(),
                f_new=f_base,
                step_norm=0.0,
                meta=meta,
            )

        # -------------------------------
        # 2. Pattern move (якщо exploratory покращив базову точку)
        # -------------------------------
        pattern = x_expl - base_point
        x_pattern = x_expl + pattern
        f_pattern = self.eval_f(x_pattern)
        evals += 1

        if f_pattern < f_expl:
            # Приймаємо pattern move
            new_base = x_pattern
            f_new = f_pattern
            step_type = "pattern"
            pattern_applied = True
        else:
            # Тільки exploratory
            new_base = x_expl
            f_new = f_expl
            step_type = "exploratory_only"
            pattern_applied = False

        step_norm = float(np.linalg.norm(new_base - base_point, ord=2))

        # Оновлюємо стан
        self.state["base_point"] = new_base
        self.state["f_base"] = f_new
        self.state["iteration"] = iteration + 1

        meta = {
            "step_size": step_size,
            "f_base_prev": f_base,
            "f_base_new": f_new,
            "iteration": iteration,
            "exploratory_evals": evals,
            "pattern_vector": pattern,
            "pattern_applied": pattern_applied,
            "step_type": step_type,
        }

        return StepResult(
            x_new=new_base,
            f_new=f_new,
            step_norm=step_norm,
            meta=meta,
        )


__all__ = [
    "HookJeevesMethod",
]
