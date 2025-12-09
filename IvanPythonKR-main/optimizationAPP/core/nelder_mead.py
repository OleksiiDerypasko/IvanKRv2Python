"""
nelder_mead.py

Реалізація методу Нелдера–Міда як стратегії Optimizer.

Метод працює тільки зі значеннями функції f(x) (без градієнтів і Гессіана)
і оперує симплексом з (n + 1) вершин у n-вимірному просторі.

Основні кроки:
    1. Сортування вершин симплекса за значенням f.
    2. Обчислення центроїда всіх вершин, окрім найгіршої.
    3. Спроба відбиття (reflection).
    4. За потреби — розширення (expansion).
    5. Або контракт (contraction) / стиснення симплекса (shrink).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .optimizer_base import Optimizer, StepResult
from .functions import ScalarFunction, VectorFunction, MatrixFunction, ArrayLike


class NelderMeadMethod(Optimizer):
    """
    Метод Нелдера–Міда.

    Особливості:
        - не використовує градієнт чи Гессіан;
        - тримає всередині поточний симплекс;
        - один виклик step() = одна ітерація алгоритму над симплексом;
        - движок може оцінювати збіжність по розміру симплекса чи по step_norm.

    Налаштування (options):
        alpha                  : коефіцієнт відбиття (reflection), default: 1.0
        gamma                  : коефіцієнт розширення (expansion), default: 2.0
        rho                    : коефіцієнт контракту (contraction), default: 0.5
        sigma                  : коефіцієнт стиснення (shrink), default: 0.5
        initial_simplex_scale  : масштаб для побудови початкового симплекса (default: 0.05)
        grad_tol               : поріг для норми градієнта (не використовується, але для сумісності)
        min_simplex_diameter   : поріг для діаметра симплекса (можна використати в движку), default: 1e-8
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
            name=name or "Nelder–Mead simplex",
        )

    # ------------------------------------------------------------------
    # Ініціалізація симплекса
    # ------------------------------------------------------------------

    def initialize(self, x0: ArrayLike) -> None:
        """
        Побудова початкового симплекса навколо x0.
        """
        super().initialize(x0)

        x0 = np.asarray(x0, dtype=float)
        n = x0.size

        scale = float(self.options.get("initial_simplex_scale", 0.05))

        simplex = np.zeros((n + 1, n), dtype=float)
        simplex[0] = x0

        # Класичний варіант побудови симплекса:
        # для кожної координати додаємо маленьке зміщення
        for i in range(n):
            y = x0.copy()
            if y[i] != 0.0:
                y[i] = (1.0 + scale) * y[i]
            else:
                y[i] = scale
            simplex[i + 1] = y

        f_values = np.zeros(n + 1, dtype=float)
        for i in range(n + 1):
            f_values[i] = self.eval_f(simplex[i])

        self.state["simplex"] = simplex
        self.state["f_values"] = f_values
        self.state["iteration"] = 0

    # ------------------------------------------------------------------
    # Допоміжні функції
    # ------------------------------------------------------------------

    def _ensure_initialized(self, x_k: np.ndarray) -> None:
        """
        Якщо симплекс ще не ініціалізований, ініціалізуємо його навколо x_k.
        """
        if "simplex" not in self.state or "f_values" not in self.state:
            self.initialize(x_k)

    @staticmethod
    def _simplex_diameter(simplex: np.ndarray) -> float:
        """
        Оцінка "розміру" симплекса як максимальна відстань від кращої точки.
        """
        best = simplex[0]
        diffs = simplex - best
        dists = np.linalg.norm(diffs, axis=1)
        return float(np.max(dists))

    # ------------------------------------------------------------------
    # Один крок методу Нелдера–Міда
    # ------------------------------------------------------------------

    def _step_impl(self, x_k: np.ndarray) -> StepResult:
        """
        Один крок методу Нелдера–Міда.

        x_k тут практично не використовується, оскільки алгоритм працює
        з внутрішнім симплексом. Але при першому виклику ми можемо
        використати x_k як початкову точку для побудови симплекса.
        """
        self._ensure_initialized(x_k)

        alpha: float = float(self.options.get("alpha", 1.0))
        gamma: float = float(self.options.get("gamma", 2.0))
        rho: float = float(self.options.get("rho", 0.5))
        sigma: float = float(self.options.get("sigma", 0.5))

        simplex: np.ndarray = self.state["simplex"]
        f_values: np.ndarray = self.state["f_values"]
        iteration: int = int(self.state.get("iteration", 0))

        # Сортуємо вершини за значенням функції
        order = np.argsort(f_values)
        simplex = simplex[order]
        f_values = f_values[order]

        # Зберігаємо попередню найкращу точку (для step_norm)
        x_best_prev = simplex[0].copy()
        f_best_prev = float(f_values[0])

        n = simplex.shape[1]

        # Індекси:
        #   0           – найкраща точка
        #   1 .. n-1    – середні точки
        #   n           – найгірша точка
        x_best = simplex[0]
        x_worst = simplex[-1]

        f_best = float(f_values[0])
        f_worst = float(f_values[-1])
        f_second_worst = float(f_values[-2])

        # Центроїд усіх, окрім найгіршої точки
        centroid = np.mean(simplex[:-1], axis=0)

        # --------------------------------------------------------------
        # 1. Reflection (відбиття)
        # --------------------------------------------------------------
        x_reflect = centroid + alpha * (centroid - x_worst)
        f_reflect = self.eval_f(x_reflect)

        step_type = "reflection"
        used_point = x_reflect
        used_value = f_reflect

        if f_best <= f_reflect < f_second_worst:
            # Випадок "прийнятне відбиття"
            simplex[-1] = x_reflect
            f_values[-1] = f_reflect

        elif f_reflect < f_best:
            # ----------------------------------------------------------
            # 2. Expansion (розширення)
            # ----------------------------------------------------------
            x_expand = centroid + gamma * (x_reflect - centroid)
            f_expand = self.eval_f(x_expand)

            if f_expand < f_reflect:
                simplex[-1] = x_expand
                f_values[-1] = f_expand
                step_type = "expansion"
                used_point = x_expand
                used_value = f_expand
            else:
                simplex[-1] = x_reflect
                f_values[-1] = f_reflect
                step_type = "reflection(best)"
                used_point = x_reflect
                used_value = f_reflect

        else:
            # f_reflect >= f_second_worst
            # ----------------------------------------------------------
            # 3. Contraction (контракт)
            # ----------------------------------------------------------
            if f_reflect < f_worst:
                # Зовнішній контракт
                x_contract = centroid + rho * (x_reflect - centroid)
            else:
                # Внутрішній контракт
                x_contract = centroid - rho * (centroid - x_worst)

            f_contract = self.eval_f(x_contract)

            if f_contract < f_worst:
                simplex[-1] = x_contract
                f_values[-1] = f_contract
                step_type = "contraction"
                used_point = x_contract
                used_value = f_contract
            else:
                # ------------------------------------------------------
                # 4. Shrink (стиснення симплекса)
                # ------------------------------------------------------
                step_type = "shrink"
                for i in range(1, n + 1):
                    simplex[i] = x_best + sigma * (simplex[i] - x_best)
                    f_values[i] = self.eval_f(simplex[i])

                used_point = simplex[1]
                used_value = float(f_values[1])

        # Знову відсортуємо симплекс для визначення нової найкращої точки
        order = np.argsort(f_values)
        simplex = simplex[order]
        f_values = f_values[order]

        x_best_new = simplex[0].copy()
        f_best_new = float(f_values[0])

        # Оновлюємо стан
        self.state["simplex"] = simplex
        self.state["f_values"] = f_values
        self.state["iteration"] = iteration + 1

        step_norm = float(np.linalg.norm(x_best_new - x_best_prev, ord=2))
        diameter = self._simplex_diameter(simplex)

        meta = {
            "f_best_prev": f_best_prev,
            "f_best_new": f_best_new,
            "used_point": used_point,
            "used_value": used_value,
            "step_type": step_type,
            "iteration": iteration,
            "simplex_diameter": diameter,
        }

        return StepResult(
            x_new=x_best_new,
            f_new=f_best_new,
            step_norm=step_norm,
            meta=meta,
        )


__all__ = [
    "NelderMeadMethod",
]
