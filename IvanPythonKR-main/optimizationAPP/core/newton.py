"""
newton.py

Реалізація методу Ньютона як стратегії Optimizer.

Ідея:
    x_{k+1} = x_k + α_k * p_k,
    де p_k розв'язує систему:
        H_k * p_k = -g_k,
    H_k = ∇²f(x_k), g_k = ∇f(x_k).

    Для глобальної збіжності:
        - використовуємо регуляризацію Гессіана (H_k + λ I),
        - лінійний пошук (backtracking line search) з умовою Арміхо
          або інші методи одномірного пошуку з core/line_search.py.
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


class NewtonMethod(Optimizer):
    """
    Метод Ньютона для мінімізації багатовимірних функцій.

    Особливості:
        - використовує градієнт та Гессіан;
        - напрямок p_k визначається з H_k p_k = -g_k;
        - при виродженому/несиметричному/негативно визначеному Гессіані
          використовується регуляризація (H_k + λ I);
        - для глобальної збіжності застосовується line search:
          за замовчуванням Armijo backtracking через core/line_search.py;
        - у разі проблем із Гессіаном можливий fallback до градієнтного напрямку.

    Налаштування (options):
        alpha0             : початковий крок α_0 (default: 1.0)
        tau                : множник для зменшення кроку (0 < tau < 1, default: 0.5)
        c1                 : константа Арміхо (default: 1e-4)
        max_backtracking   : максимальна кількість кроків backtracking (default: 20)
        grad_tol           : поріг норми градієнта (default: 1e-8)
        min_alpha          : мінімально допустимий крок (default: 1e-12)
        reg_lambda         : початковий множник λ для регуляризації Гессіана (default: 1e-6)
        max_reg_scale      : максимальний масштаб для λ (default: 1e6)
        use_line_search    : чи використовувати line search (default: True)
        restart_on_non_descent :
                             якщо True, при p_k^T g_k >= 0 беремо p_k = -g_k (default: True)

        # Додаткові налаштування line search (нове):
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
    requires_hessian: bool = True

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
            name=name or "Newton method",
        )

    def initialize(self, x0) -> None:
        """
        Ініціалізувати стан перед запуском ітерацій Ньютона.
        """
        super().initialize(x0)
        self.state["iteration"] = 0

    def _compute_newton_direction(
        self,
        g_k: np.ndarray,
        H_k: np.ndarray,
        reg_lambda: float,
        max_reg_scale: float,
    ) -> tuple[np.ndarray, float, bool]:
        """
        Обчислити напрямок Ньютона p_k із регуляризацією.

        Повертає:
            p_k          - знайдений напрямок
            used_lambda  - фактичне λ, яке було використано
            fallback     - True, якщо довелося робити fallback (напр. через сингулярність)
        """
        n = len(g_k)
        I = np.eye(n, dtype=float)

        lam = reg_lambda
        fallback = False

        # Симетризація Гессіана на всяк випадок
        H_sym = 0.5 * (H_k + H_k.T)

        while lam <= reg_lambda * max_reg_scale:
            try:
                H_reg = H_sym + lam * I
                # Спробуємо розв'язати H_reg p = -g
                p_k = -np.linalg.solve(H_reg, g_k)
                return p_k, lam, fallback
            except np.linalg.LinAlgError:
                # Матриця погано обумовлена або вироджена — збільшуємо λ
                lam *= 10.0

        # Якщо дійшли сюди — регуляризація не допомогла
        # fallback: напрямок градієнтного спуску
        fallback = True
        p_k = -g_k.copy()
        return p_k, lam, fallback

    def _step_impl(self, x_k: np.ndarray) -> StepResult:
        """
        Один крок методу Ньютона з точки x_k.
        """
        # --- Базові налаштування (як раніше) -------------------------------
        alpha0: float = float(self.options.get("alpha0", 1.0))
        tau: float = float(self.options.get("tau", 0.5))
        c1: float = float(self.options.get("c1", 1e-4))
        max_bt: int = int(self.options.get("max_backtracking", 20))
        grad_tol: float = float(self.options.get("grad_tol", 1e-8))
        min_alpha: float = float(self.options.get("min_alpha", 1e-12))
        reg_lambda: float = float(self.options.get("reg_lambda", 1e-6))
        max_reg_scale: float = float(self.options.get("max_reg_scale", 1e6))
        use_line_search: bool = bool(self.options.get("use_line_search", True))
        restart_on_non_descent: bool = bool(
            self.options.get("restart_on_non_descent", True)
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

        iteration = int(self.state.get("iteration", 0))

        # Обчислюємо f, ∇f, ∇²f у поточній точці
        f_k = self.eval_f(x_k)
        g_k = self.eval_grad(x_k)
        grad_norm = float(np.linalg.norm(g_k, ord=2))

        # Критерій зупинки по градієнту
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
                    "lambda": 0.0,
                    "line_search_steps": 0,
                    "stopped_by": "small_gradient",
                    "iteration": iteration,
                    "used_line_search": use_line_search,
                    "fallback_to_grad": False,
                    "line_search_method": None,
                    "line_search_iterations": 0,
                    "line_search_evals": 0,
                    "line_search_failed": False,
                },
            )

        H_k = self.eval_hess(x_k)

        # Обчислюємо напрямок Ньютона з регуляризацією
        p_k, used_lambda, fallback_to_grad = self._compute_newton_direction(
            g_k=g_k,
            H_k=H_k,
            reg_lambda=reg_lambda,
            max_reg_scale=max_reg_scale,
        )

        # Перевірка напрямку спуску
        directional_derivative = float(np.dot(g_k, p_k))  # ∇f(x_k)^T p_k

        if restart_on_non_descent and directional_derivative >= 0.0:
            # Якщо напрямок не є напрямком спуску, робимо fallback до градієнтного
            p_k = -g_k
            directional_derivative = float(np.dot(g_k, p_k))
            fallback_to_grad = True

        # Якщо line search вимкнено — пробуємо α = 1.0 (стара поведінка)
        if not use_line_search:
            alpha = 1.0
            x_new = x_k + alpha * p_k
            f_new = self.eval_f(x_new)
            step_norm = float(np.linalg.norm(x_new - x_k, ord=2))

            self.state["iteration"] = iteration + 1
            meta = {
                "alpha": alpha,
                "grad": g_k,
                "grad_norm": grad_norm,
                "lambda": used_lambda,
                "line_search_steps": 0,
                "f_prev": f_k,
                "iteration": iteration,
                "used_line_search": False,
                "fallback_to_grad": fallback_to_grad,
                "line_search_method": None,
                "line_search_iterations": 0,
                "line_search_evals": 0,
                "line_search_failed": False,
            }

            return StepResult(
                x_new=x_new,
                f_new=f_new,
                step_norm=step_norm,
                meta=meta,
            )

        # --- Лінійний пошук через core/line_search.py ----------------------
        def phi(alpha: float) -> float:
            return self.eval_f(x_k + alpha * p_k)

        ls_result = None
        line_search_failed = False
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
                    raise ValueError(
                        "NewtonMethod: некоректний інтервал для line search [a, b]."
                    )

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

        # --- Вибір α з урахуванням fallback --------------------------------
        if not line_search_failed and ls_result is not None:
            alpha = float(ls_result.alpha)
            f_new = float(ls_result.phi_value)
            x_new = x_k + alpha * p_k
            ls_iterations = int(ls_result.iterations)
            ls_evals = int(ls_result.func_evals)
        else:
            # Якщо line search зламався — маленький фіксований крок
            alpha = fallback_alpha
            x_new = x_k + alpha * p_k
            f_new = self.eval_f(x_new)
            ls_iterations = 0
            ls_evals = 0

        step_norm = float(np.linalg.norm(x_new - x_k, ord=2))

        # Оновлюємо лічильник ітерацій
        self.state["iteration"] = iteration + 1

        meta: Dict[str, Any] = {
            "alpha": alpha,
            "grad": g_k,
            "grad_norm": grad_norm,
            "lambda": used_lambda,
            "line_search_steps": ls_iterations,   # сумісність зі старим полем
            "f_prev": f_k,
            "iteration": iteration,
            "used_line_search": True,
            "fallback_to_grad": fallback_to_grad,
            "line_search_method": line_search_method,
            "line_search_iterations": ls_iterations,
            "line_search_evals": ls_evals,
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
    "NewtonMethod",
]
