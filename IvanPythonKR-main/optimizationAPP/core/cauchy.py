"""
cauchy.py

Реалізація методу Коші (градієнтний спуск) як стратегії Optimizer.

Ідея:
    x_{k+1} = x_k + α_k * p_k,
    де p_k = -∇f(x_k),
        α_k підбирається за допомогою line search з core.line_search
        (за замовчуванням — backtracking з умовою Арміхо, як було раніше).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .optimizer_base import Optimizer, StepResult
from .functions import ScalarFunction, VectorFunction, MatrixFunction, ArrayLike
from .line_search import (
    line_search_1d,
    LINE_SEARCH_ARMIJO,
)


class CauchyMethod(Optimizer):
    """
    Метод Коші (градієнтний спуск).

    Особливості:
        - рухаємося вздовж антинормалі градієнта: p_k = -∇f(x_k);
        - крок α_k обирається за допомогою процедури line search
          (за замовчуванням — backtracking з умовою Арміхо, як раніше);
        - використовує лише значення функції та градієнта.

    Налаштування (options):

        # Параметри перевірки малості градієнта
        grad_tol          : поріг норми градієнта, щоб вважати, що ми близько
                            до мінімуму (default: 1e-8)

        # Вибір та налаштування line search
        line_search       : рядок з ім'ям методу лінійного пошуку.
                            Якщо None / "armijo" / "backtracking",
                            використовується Armijo backtracking
                            (LINE_SEARCH_ARMIJO) — старий варіант за
                            замовчуванням.
        line_search_tol   : tol для одномірних методів (по α, default: 1e-6)
        line_search_max_iter : max_iter для одномірних методів (default: 50)
        line_search_a     : ліва межа інтервалу для α (default: 0.0)
        line_search_b     : права межа інтервалу для α (default: alpha0
                            при першому виклику)

        line_search_options : dict з додатковими параметрами для line search.
                              Для Armijo сюди/в options можна покласти:
            alpha0           : початковий крок (default: 1.0)
            tau              : множник для зменшення кроку (0 < tau < 1,
                               default: 0.5)
            c1               : константа Арміхо (default: 1e-4)
            max_backtracking : максимальна кількість кроків backtracking
                               (default: 20)
            min_alpha        : мінімально допустимий крок (default: 1e-12)

        Зворотна сумісність:
            Якщо line_search не вказано, то за замовчуванням використовується
            Armijo backtracking з тими ж параметрами, що й у старій версії
            CauchyMethod.
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
            name=name or "Cauchy (gradient descent)",
        )

    def _step_impl(self, x_k: np.ndarray) -> StepResult:
        """
        Один крок методу Коші із точки x_k.
        """
        # ----------------------------
        # Поточне значення f та градієнта
        # ----------------------------
        f_k = self.eval_f(x_k)
        g_k = self.eval_grad(x_k)
        grad_norm = float(np.linalg.norm(g_k, ord=2))

        grad_tol: float = float(self.options.get("grad_tol", 1e-8))

        # Якщо градієнт дуже малий — вважаємо, що ми біля стаціонарної точки
        if grad_norm < grad_tol:
            return StepResult(
                x_new=x_k.copy(),
                f_new=f_k,
                step_norm=0.0,
                meta={
                    "alpha": 0.0,
                    "grad": g_k,
                    "grad_norm": grad_norm,
                    "line_search_method": None,
                    "line_search_iterations": 0,
                    "line_search_steps": 0,
                    "line_search_evals": 0,
                    "stopped_by": "small_gradient",
                },
            )

        # Напрямок спуску
        p_k = -g_k

        # Напрямна похідна φ'(0) = ∇f(x_k)^T p_k = -||g_k||^2 < 0
        directional_derivative = float(np.dot(g_k, p_k))

        # ----------------------------
        # Налаштування line search
        # ----------------------------
        # Параметри Armijo / line search за замовчуванням
        alpha0: float = float(self.options.get("alpha0", 1.0))
        tau: float = float(self.options.get("tau", 0.5))
        c1: float = float(self.options.get("c1", 1e-4))
        max_bt: int = int(self.options.get("max_backtracking", 20))
        min_alpha: float = float(self.options.get("min_alpha", 1e-12))

        # Вибір методу line search
        ls_name = self.options.get("line_search", None) or self.options.get(
            "line_search_method", None
        )
        if ls_name is None or str(ls_name) in {"armijo", "backtracking", "armijo_backtracking"}:
            ls_method = LINE_SEARCH_ARMIJO
        else:
            ls_method = str(ls_name)

        # Інтервал для α (для більшості одномірних методів)
        alpha_min = float(self.options.get("line_search_a", 0.0))
        alpha_max = float(self.options.get("line_search_b", alpha0))

        if alpha_max <= alpha_min:
            # Невдале задання інтервалу – підстрахуємося
            alpha_min = 0.0
            alpha_max = alpha0 if alpha0 > 0 else 1.0

        ls_tol = float(self.options.get("line_search_tol", 1e-6))
        ls_max_iter = int(self.options.get("line_search_max_iter", max_bt))

        # Додаткові параметри для line search
        ls_options: Dict[str, Any] = dict(self.options.get("line_search_options", {}))

        if ls_method == LINE_SEARCH_ARMIJO:
            # Наповнюємо параметри Armijo значеннями з options,
            # зберігаючи стару поведінку за замовчуванням.
            ls_options.setdefault("alpha0", alpha0)
            ls_options.setdefault("tau", tau)
            ls_options.setdefault("c1", c1)
            ls_options.setdefault("max_backtracking", max_bt)
            ls_options.setdefault("min_alpha", min_alpha)

            # Значення φ(0) та напрямна похідна – специфічні для поточного кроку
            ls_options["f0"] = f_k
            ls_options["directional_derivative"] = directional_derivative
        else:
            # Для інших методів можемо передати alpha0 як hint
            ls_options.setdefault("alpha0", alpha0)

        # ----------------------------
        # Визначаємо φ(α) = f(x_k + α p_k)
        # ----------------------------
        def phi(alpha: float) -> float:
            x_candidate = x_k + float(alpha) * p_k
            return self.eval_f(x_candidate)

        # Запуск одномірного пошуку
        ls_result = line_search_1d(
            phi=phi,
            a=alpha_min,
            b=alpha_max,
            method=ls_method,
            tol=ls_tol,
            max_iter=ls_max_iter,
            options=ls_options,
        )

        alpha_star = float(ls_result.alpha)
        x_new = x_k + alpha_star * p_k
        f_new = float(ls_result.phi_value)
        step_norm = float(np.linalg.norm(x_new - x_k, ord=2))

        meta = {
            "alpha": alpha_star,
            "grad": g_k,
            "grad_norm": grad_norm,
            "f_prev": f_k,
            "line_search_method": ls_method,
            "line_search_iterations": ls_result.iterations,
            "line_search_steps": ls_result.iterations,  # для зворотної сумісності
            "line_search_evals": ls_result.func_evals,
            "line_search_meta": ls_result.meta,
        }

        return StepResult(
            x_new=x_new,
            f_new=f_new,
            step_norm=step_norm,
            meta=meta,
        )


__all__ = [
    "CauchyMethod",
]
