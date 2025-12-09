"""
polak_ribiere.py

Реалізація методу Полака–Ріб’єра (градієнтний метод спряжених напрямків)
як стратегії Optimizer.

Ідея:
    p_0 = -g_0
    p_k = -g_k + β_k * p_{k-1},

    класична формула Полака–Ріб’єра:
        β_k^PR = (g_k^T (g_k - g_{k-1})) / (g_{k-1}^T g_{k-1})

    Часто використовують варіант із "обрізкою":
        β_k = max(0, β_k^PR)

    x_{k+1} = x_k + α_k * p_k,
    де α_k підбирається за допомогою процедури одномірного пошуку
    (line search) з умовою Арміхо або іншими 1D-методами.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .optimizer_base import Optimizer, StepResult
from .functions import ScalarFunction, VectorFunction, MatrixFunction
from .line_search import (
    line_search_1d,
    LINE_SEARCH_ARMIJO,
    LINE_SEARCH_DICHOTOMY,
    LINE_SEARCH_INTERVAL_HALVING,
    LINE_SEARCH_GOLDEN_SECTION,
    LINE_SEARCH_STEP_ADAPTATION,
    LINE_SEARCH_CUBIC_4POINT,
)


class PolakRibiereMethod(Optimizer):
    """
    Метод Полака–Ріб’єра (спряжені градієнти).

    Особливості:
        - використовує попередній напрямок p_{k-1};
        - коефіцієнт
              β_k^PR = (g_k^T (g_k - g_{k-1})) / (g_{k-1}^T g_{k-1});
        - часто беруть β_k = max(0, β_k^PR), щоб уникати нестійких напрямків;
        - при першій ітерації або при збоях повертається до градієнтного
          спуску (метод Коші).

    Налаштування (options):
        alpha0                  : початковий крок (default: 1.0)
        tau                     : множник для зменшення кроку (0 < tau < 1, default: 0.5)
        c1                      : константа Арміхо (default: 1e-4)
        max_backtracking        : максимальна кількість кроків backtracking (default: 20)
        grad_tol                : поріг норми градієнта (default: 1e-8)
        min_alpha               : мінімально допустимий крок (default: 1e-12)
        restart_on_non_descent  : якщо True, робити рестарт (p_k = -g_k),
                                  коли напрямок не є напрямком спуску (p_k^T g_k >= 0)
                                  (default: True)
        beta_positive           : якщо True, використовувати β_k = max(0, β_k^PR)
                                  (default: True)

        # Налаштування line search (аналогічно Коші / Флетчеру–Рівзу):
        line_search             : назва методу лінійного пошуку:
                                  "armijo_backtracking" (default),
                                  "golden_section", "dichotomy",
                                  "interval_halving", "step_adaptation",
                                  "cubic_4point"
        line_search_tol         : точність по α для інтервальних методів (default: 1e-6)
        line_search_max_iter    : max ітерацій 1D-пошуку (default: max_backtracking)
        line_search_max_step    : права межа інтервалу [0, max_step] для методів,
                                  що потребують інтервалу (default: alpha0 або 1.0)
        line_search_options     : dict з додатковими параметрами для line search
                                  (передається в line_search_1d(..., options=...))

        fallback_alpha          : фіксований крок для fallback, якщо line search провалився
                                  (default: 1e-3, але не менше за min_alpha)
    """

    requires_gradient: bool = True
    requires_hessian: bool = False

    def __init__(
        self,
        func: ScalarFunction,
        grad: Optional[VectorFunction] = None,
        hess: Optional[MatrixFunction] = None,
        options: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            func=func,
            grad=grad,
            hess=hess,
            options=options,
            name=name or "Polak–Ribiere (conjugate gradients)",
        )

    def initialize(self, x0) -> None:
        """
        Ініціалізація стану для першої ітерації.
        """
        super().initialize(x0)
        self.state["g_prev"] = None
        self.state["p_prev"] = None
        self.state["iteration"] = 0

    def _step_impl(self, x_k: np.ndarray) -> StepResult:
        """
        Один крок методу Полака–Ріб’єра із точки x_k.
        """
        # --- Базові налаштування (як раніше) -------------------------------
        alpha0: float = float(self.options.get("alpha0", 1.0))
        tau: float = float(self.options.get("tau", 0.5))
        c1: float = float(self.options.get("c1", 1e-4))
        max_bt: int = int(self.options.get("max_backtracking", 20))
        grad_tol: float = float(self.options.get("grad_tol", 1e-8))
        min_alpha: float = float(self.options.get("min_alpha", 1e-12))
        restart_on_non_descent: bool = bool(
            self.options.get("restart_on_non_descent", True)
        )
        beta_positive: bool = bool(
            self.options.get("beta_positive", True)
        )

        # --- Налаштування line search --------------------------------------
        line_search_method: str = str(
            self.options.get("line_search", LINE_SEARCH_ARMIJO)
        )

        line_search_tol: float = float(self.options.get("line_search_tol", 1e-6))
        line_search_max_iter: int = int(
            self.options.get("line_search_max_iter", max_bt)
        )

        ls_max_step_default = alpha0 if alpha0 > 0.0 else 1.0
        line_search_max_step: float = float(
            self.options.get("line_search_max_step", ls_max_step_default)
        )
        if line_search_max_step <= 0.0:
            line_search_max_step = ls_max_step_default

        base_ls_options: Dict[str, Any] = dict(
            self.options.get("line_search_options", {})
        )

        fallback_alpha: float = float(
            self.options.get("fallback_alpha", 1e-3)
        )
        if fallback_alpha <= 0.0:
            fallback_alpha = 1e-3
        fallback_alpha = max(fallback_alpha, min_alpha)

        # --- Поточне значення функції та градієнта -------------------------
        f_k = self.eval_f(x_k)
        g_k = self.eval_grad(x_k)
        grad_norm = float(np.linalg.norm(g_k, ord=2))

        iteration = int(self.state.get("iteration", 0))

        # Якщо градієнт дуже малий — біля стаціонарної точки
        if grad_norm < grad_tol:
            self.state["iteration"] = iteration + 1
            return StepResult(
                x_new=x_k.copy(),
                f_new=f_k,
                step_norm=0.0,
                meta={
                    "alpha": 0.0,
                    "grad": g_k,
                    "grad_norm": grad_norm,
                    "beta": 0.0,
                    "line_search_method": None,
                    "line_search_iterations": 0,
                    "line_search_evals": 0,
                    "line_search_steps": 0,
                    "line_search_failed": False,
                    "stopped_by": "small_gradient",
                    "iteration": iteration,
                },
            )

        g_prev = self.state.get("g_prev", None)
        p_prev = self.state.get("p_prev", None)

        # --- Обчислення напрямку p_k (логіка Полака–Ріб’єра не чіпаємо) ---
        if g_prev is None or p_prev is None:
            # Перша ітерація — чистий градієнтний спуск
            beta = 0.0
            p_k = -g_k
            restarted = True
        else:
            # Класичний β_k Полака–Ріб’єра:
            # β_k = (g_k^T (g_k - g_{k-1})) / (g_{k-1}^T g_{k-1})
            diff = g_k - g_prev
            num = float(np.dot(g_k, diff))
            den = float(np.dot(g_prev, g_prev))

            if den <= 1e-20:
                beta = 0.0
            else:
                beta = num / den

            if beta_positive:
                beta = max(0.0, beta)

            p_k = -g_k + beta * p_prev
            restarted = False

            # Перевірка напрямку спуску
            if restart_on_non_descent and float(np.dot(p_k, g_k)) >= 0.0:
                p_k = -g_k
                beta = 0.0
                restarted = True

        # --- Одномірний пошук уздовж p_k ----------------------------------
        directional_derivative = float(np.dot(g_k, p_k))  # ∇f(x_k)^T p_k

        def phi(alpha: float) -> float:
            return self.eval_f(x_k + alpha * p_k)

        line_search_failed = False
        ls_result = None
        ls_iterations = 0
        ls_evals = 0

        try:
            if line_search_method == LINE_SEARCH_ARMIJO:
                # Armijo backtracking через line_search_1d
                ls_options = dict(base_ls_options)
                ls_options.update(
                    {
                        "f0": f_k,
                        "directional_derivative": directional_derivative,
                        "alpha0": alpha0,
                        "tau": tau,
                        "c1": c1,
                        "max_backtracking": max_bt,
                        "min_alpha": min_alpha,
                    }
                )

                ls_result = line_search_1d(
                    phi=phi,
                    a=0.0,
                    b=line_search_max_step,
                    method=LINE_SEARCH_ARMIJO,
                    tol=line_search_tol,
                    max_iter=max_bt,
                    options=ls_options,
                )
            else:
                # Інтервальні методи: [0, line_search_max_step]
                a = 0.0
                b = line_search_max_step
                if b <= a:
                    raise ValueError("Некоректний інтервал для line search: [a, b]")

                ls_options = dict(base_ls_options)
                ls_options.setdefault("alpha0", 0.0)

                ls_result = line_search_1d(
                    phi=phi,
                    a=a,
                    b=b,
                    method=line_search_method,
                    tol=line_search_tol,
                    max_iter=line_search_max_iter,
                    options=ls_options,
                )

            if (
                ls_result is None
                or not np.isfinite(ls_result.alpha)
                or not np.isfinite(ls_result.phi_value)
            ):
                line_search_failed = True
            else:
                if line_search_method != LINE_SEARCH_ARMIJO:
                    if ls_result.alpha < 0.0 or ls_result.alpha > line_search_max_step * 1.001:
                        line_search_failed = True

        except Exception:
            ls_result = None
            line_search_failed = True

        # --- Обчислюємо наступну точку, враховуючи fallback ----------------
        if not line_search_failed and ls_result is not None:
            alpha = float(ls_result.alpha)
            f_new = float(ls_result.phi_value)
            x_new = x_k + alpha * p_k
            ls_iterations = int(ls_result.iterations)
            ls_evals = int(ls_result.func_evals)
        else:
            # Fallback: маленький фіксований крок
            alpha = fallback_alpha
            x_new = x_k + alpha * p_k
            f_new = self.eval_f(x_new)
            ls_iterations = 0
            ls_evals = 0

        step_norm = float(np.linalg.norm(alpha * p_k, ord=2))

        # Оновлюємо стан для наступної ітерації
        self.state["g_prev"] = g_k
        self.state["p_prev"] = p_k
        self.state["iteration"] = iteration + 1

        meta = {
            "alpha": alpha,
            "grad": g_k,
            "grad_norm": grad_norm,
            "beta": beta,
            "f_prev": f_k,
            "restarted": restarted,
            "iteration": iteration,
            "line_search_method": line_search_method,
            "line_search_iterations": ls_iterations,
            "line_search_evals": ls_evals,
            # для сумісності зі старим полем:
            "line_search_steps": ls_iterations,
            "line_search_failed": line_search_failed,
        }

        if ls_result is not None and hasattr(ls_result, "meta"):
            meta["line_search_meta"] = ls_result.meta

        return StepResult(
            x_new=x_new,
            f_new=f_new,
            step_norm=step_norm,
            meta=meta,
        )


__all__ = [
    "PolakRibiereMethod",
]
