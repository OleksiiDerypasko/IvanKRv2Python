"""
optimizer_base.py

Базові класи та типи для реалізації методів оптимізації (Strategy).

Ідея:
    - Є абстрактний клас Optimizer, від якого наслідуються всі конкретні методи:
        * CauchyMethod
        * FletcherReevesMethod
        * PolakRibiereMethod
        * NewtonMethod
        * NelderMeadMethod
        * HookJeevesMethod
    - Кожен метод реалізує _step_impl(), а користувач/движок викликає step().

Формат:
    step(x_k: np.ndarray) -> StepResult
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .functions import (
    ArrayLike,
    ScalarFunction,
    VectorFunction,
    MatrixFunction,
    numerical_gradient,
    numerical_hessian,
)


# ---------------------------------------------------------------------------
# Результат одного кроку методу оптимізації
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """
    Результат одного кроку оптимізації.

    Атрибути:
        x_new     - нова точка пошуку (x_{k+1})
        f_new     - значення функції у новій точці f(x_{k+1})
        step_norm - норма кроку ||x_{k+1} - x_k||
        meta      - додаткова інформація (градієнт, напрямок, тощо)
    """
    x_new: np.ndarray
    f_new: float
    step_norm: float
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Базовий клас Optimizer (Strategy)
# ---------------------------------------------------------------------------

class Optimizer(ABC):
    """
    Абстрактний базовий клас для всіх методів оптимізації.

    Кожен конкретний метод:
        - наслідується від Optimizer;
        - реалізує _step_impl();
        - за потреби переозначає requires_gradient / requires_hessian.

    Використання:
        opt = CauchyMethod(func=..., grad=..., options={...})
        opt.reset()
        res = opt.step(x_k)  # StepResult
    """

    # Ці флаги можуть бути переозначені в дочірніх класах
    requires_gradient: bool = False
    requires_hessian: bool = False

    def __init__(
        self,
        func: ScalarFunction,
        grad: Optional[VectorFunction] = None,
        hess: Optional[MatrixFunction] = None,
        options: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        func : ScalarFunction
            Цільова функція f(x).
        grad : Optional[VectorFunction]
            Аналітичний градієнт ∇f(x), якщо відомий.
            Якщо None і requires_gradient = True, буде використано numerical_gradient.
        hess : Optional[MatrixFunction]
            Аналітичний Гессіан H(x), якщо відомий.
            Якщо None і requires_hessian = True, буде використано numerical_hessian.
        options : Optional[dict]
            Додаткові параметри методу (крок, критерії зупинки тощо).
        name : Optional[str]
            Людяна назва методу (для логів/таблиць).
        """
        self.func = func
        self._grad = grad
        self._hess = hess
        self.options: Dict[str, Any] = options or {}
        self.name: str = name or self.__class__.__name__

        # Лічильники викликів (можна використовувати в зведеній таблиці)
        self.func_evals: int = 0
        self.grad_evals: int = 0
        self.hess_evals: int = 0

        # Місце для внутрішнього стану (наприклад, попередній напрямок)
        self.state: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Сервісні методи для обчислення f, ∇f, H із підрахунком викликів
    # ------------------------------------------------------------------

    def eval_f(self, x: ArrayLike) -> float:
        """Обчислити f(x) та збільшити лічильник викликів функції."""
        self.func_evals += 1
        return float(self.func(np.asarray(x, dtype=float)))

    def eval_grad(self, x: ArrayLike) -> np.ndarray:
        """
        Обчислити ∇f(x):
            - якщо передано аналітичний grad, використати його;
            - інакше — чисельно через numerical_gradient.
        """
        self.grad_evals += 1
        x_arr = np.asarray(x, dtype=float)
        if self._grad is not None:
            return np.asarray(self._grad(x_arr), dtype=float)
        # fallback — чисельний градієнт
        return numerical_gradient(self.func, x_arr)

    def eval_hess(self, x: ArrayLike) -> np.ndarray:
        """
        Обчислити H(x):
            - якщо передано аналітичний hess, використати його;
            - інакше — чисельно через numerical_hessian.
        """
        self.hess_evals += 1
        x_arr = np.asarray(x, dtype=float)
        if self._hess is not None:
            return np.asarray(self._hess(x_arr), dtype=float)
        # fallback — чисельний Гессіан
        return numerical_hessian(self.func, x_arr)

    # ------------------------------------------------------------------
    # Життєвий цикл методу
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Скинути внутрішній стан та лічильники перед новим запуском оптимізації.
        Викликається движком перед першою ітерацією.
        """
        self.func_evals = 0
        self.grad_evals = 0
        self.hess_evals = 0
        self.state.clear()

    def initialize(self, x0: ArrayLike) -> None:
        """
        Ініціалізувати внутрішній стан для початкової точки x0 (опційно).
        За замовчуванням нічого не робить. Деякі методи (Ньютона, Хука–Дживса,
        Нелдера–Міда) можуть переозначити цей метод.
        """
        self.state["x0"] = np.asarray(x0, dtype=float)

    # ------------------------------------------------------------------
    # Головний публічний метод step()
    # ------------------------------------------------------------------

    def step(self, x_k: ArrayLike) -> StepResult:
        """
        Виконати один крок методу оптимізації з поточної точки x_k.

        Повертає:
            StepResult(x_new, f_new, step_norm, meta)
        """
        x_k_arr = np.asarray(x_k, dtype=float)
        result = self._step_impl(x_k_arr)

        if not isinstance(result, StepResult):
            raise TypeError(
                f"{self.__class__.__name__}._step_impl() "
                f"повинен повертати StepResult, отримано: {type(result)}"
            )

        return result

    # ------------------------------------------------------------------
    # Абстрактний метод, який реалізують конкретні стратегії
    # ------------------------------------------------------------------

    @abstractmethod
    def _step_impl(self, x_k: np.ndarray) -> StepResult:
        """
        Реалізація одного кроку методу оптимізації.

        Parameters
        ----------
        x_k : np.ndarray
            Поточна точка x_k.

        Returns
        -------
        StepResult
            Результат кроку: x_{k+1}, f(x_{k+1}), ||x_{k+1} - x_k||, meta.
        """
        raise NotImplementedError


__all__ = [
    "StepResult",
    "Optimizer",
]
