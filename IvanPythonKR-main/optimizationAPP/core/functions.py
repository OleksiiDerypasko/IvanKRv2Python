"""
functions.py

Модуль з визначенням тестових цільових функцій, їх градієнтів та Гессіанів.
Формат:
    - усі функції працюють з вектором x: numpy.ndarray форми (2,).
    - реалізовані:
        f1, ..., f8
        grad_f1, ..., grad_f8
        hess_f1, ..., hess_f8
    - є реєстр FUNCTIONS для зручного вибору функції в GUI/движку.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

ArrayLike = np.ndarray
ScalarFunction = Callable[[ArrayLike], float]
VectorFunction = Callable[[ArrayLike], ArrayLike]
MatrixFunction = Callable[[ArrayLike], ArrayLike]


# ---------------------------------------------------------------------------
# Чисельні похідні (центральні різниці)
# ---------------------------------------------------------------------------

def numerical_gradient(
    func: ScalarFunction,
    x: ArrayLike,
    h: float = 1e-6,
) -> ArrayLike:
    """
    Чисельний градієнт за центральною різницею.

    ∂f/∂x_i ≈ (f(x + h e_i) - f(x - h e_i)) / (2h)
    """
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x, dtype=float)

    for i in range(len(x)):
        x_fwd = x.copy()
        x_bwd = x.copy()
        x_fwd[i] += h
        x_bwd[i] -= h
        grad[i] = (func(x_fwd) - func(x_bwd)) / (2.0 * h)

    return grad


def numerical_hessian(
    func: ScalarFunction,
    x: ArrayLike,
    h: float = 1e-4,
) -> ArrayLike:
    """
    Чисельний Гессіан за центральною різницею.

    Діагональні елементи:
        ∂²f/∂x_i² ≈ (f(x+h e_i) - 2f(x) + f(x-h e_i)) / h²

    Позадіагональні елементи (i != j):
        ∂²f/∂x_i∂x_j ≈
            ( f(x_i+h, x_j+h) - f(x_i+h, x_j-h)
            - f(x_i-h, x_j+h) + f(x_i-h, x_j-h) ) / (4 h²)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    H = np.zeros((n, n), dtype=float)

    f_x = func(x)

    # Діагональ
    for i in range(n):
        x_fwd = x.copy()
        x_bwd = x.copy()
        x_fwd[i] += h
        x_bwd[i] -= h

        f_fwd = func(x_fwd)
        f_bwd = func(x_bwd)

        H[i, i] = (f_fwd - 2.0 * f_x + f_bwd) / (h ** 2)

    # Позадіагональні
    for i in range(n):
        for j in range(i + 1, n):
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()

            x_pp[i] += h; x_pp[j] += h
            x_pm[i] += h; x_pm[j] -= h
            x_mp[i] -= h; x_mp[j] += h
            x_mm[i] -= h; x_mm[j] -= h

            f_pp = func(x_pp)
            f_pm = func(x_pm)
            f_mp = func(x_mp)
            f_mm = func(x_mm)

            value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h ** 2)
            H[i, j] = H[j, i] = value

    return H


# ---------------------------------------------------------------------------
# Цільові функції f1–f8
# x = [x1, x2]
# ---------------------------------------------------------------------------

def f1(x: ArrayLike) -> float:
    """
    f1(x1, x2) = (12 + x1^2 + (1 + x2^2)/x1^2 + ((x1*x2)^2 + 100)/(x1*x2)^4) / 10
    """
    x1, x2 = np.asarray(x, dtype=float)
    x1x2 = x1 * x2

    if x1 == 0.0 or x1x2 == 0.0:
        # Формально функція не визначена, але щоб не падати:
        raise ValueError("f1: x1 і x1*x2 не повинні дорівнювати нулю (ділення на нуль).")

    term1 = 12.0
    term2 = x1 ** 2
    term3 = (1.0 + x2 ** 2) / (x1 ** 2)
    term4 = (x1x2 ** 2 + 100.0) / (x1x2 ** 4)

    return (term1 + term2 + term3 + term4) / 10.0


def f2(x: ArrayLike) -> float:
    """
    f2(x1, x2) = (x1 - x2)^2 + (x1 + x2 - 10)^2 / 9
    """
    x1, x2 = np.asarray(x, dtype=float)
    return (x1 - x2) ** 2 + (x1 + x2 - 10.0) ** 2 / 9.0


def f3(x: ArrayLike) -> float:
    """
    f3(x1, x2) = 5 * (x2 - 4*x1^3 + 3*x1)^2 + (x1 + 1)^2
    """
    x1, x2 = np.asarray(x, dtype=float)
    inner = x2 - 4.0 * x1 ** 3 + 3.0 * x1
    return 5.0 * inner ** 2 + (x1 + 1.0) ** 2


def f4(x: ArrayLike) -> float:
    """
    f4(x1, x2) = 5 * (x2 - 4*x1^3 + 3*x1)^2 + (x1 - 1)^2
    """
    x1, x2 = np.asarray(x, dtype=float)
    inner = x2 - 4.0 * x1 ** 3 + 3.0 * x1
    return 5.0 * inner ** 2 + (x1 - 1.0) ** 2


def f5(x: ArrayLike) -> float:
    """
    f5(x1, x2) = 100 * (x2 - x1^3 + x1)^2 + (x1 - 1)^2
    """
    x1, x2 = np.asarray(x, dtype=float)
    inner = x2 - x1 ** 3 + x1
    return 100.0 * inner ** 2 + (x1 - 1.0) ** 2


def f6(x: ArrayLike) -> float:
    """
    f6(x1, x2) = (0.01 * (x1 - 3))^2 - (x2 - x1) + exp(20 * (x2 - x1))
    """
    x1, x2 = np.asarray(x, dtype=float)
    term1 = (0.01 * (x1 - 3.0)) ** 2
    term2 = -(x2 - x1)
    term3 = np.exp(20.0 * (x2 - x1))
    return term1 + term2 + term3


def f7(x: ArrayLike) -> float:
    """
    f7(x1, x2) = 100 * (x2 - x1^2)^2 + (1 - x1)^2
    (модифікована функція Розенброка)
    """
    x1, x2 = np.asarray(x, dtype=float)
    return 100.0 * (x2 - x1 ** 2) ** 2 + (1.0 - x1) ** 2


def f8(x: ArrayLike) -> float:
    """
    f8(x1, x2) = (x1 - 4)^2 + (x2 - 4)^2
    (проста квадратична форма)
    """
    x1, x2 = np.asarray(x, dtype=float)
    return (x1 - 4.0) ** 2 + (x2 - 4.0) ** 2


# ---------------------------------------------------------------------------
# Обгортки для градієнтів та Гессіанів кожної функції
# (на основі чисельного диференціювання)
# ---------------------------------------------------------------------------

def grad_f1(x: ArrayLike) -> ArrayLike:
    return numerical_gradient(f1, x)


def grad_f2(x: ArrayLike) -> ArrayLike:
    return numerical_gradient(f2, x)


def grad_f3(x: ArrayLike) -> ArrayLike:
    return numerical_gradient(f3, x)


def grad_f4(x: ArrayLike) -> ArrayLike:
    return numerical_gradient(f4, x)


def grad_f5(x: ArrayLike) -> ArrayLike:
    return numerical_gradient(f5, x)


def grad_f6(x: ArrayLike) -> ArrayLike:
    return numerical_gradient(f6, x)


def grad_f7(x: ArrayLike) -> ArrayLike:
    return numerical_gradient(f7, x)


def grad_f8(x: ArrayLike) -> ArrayLike:
    return numerical_gradient(f8, x)


def hess_f1(x: ArrayLike) -> ArrayLike:
    return numerical_hessian(f1, x)


def hess_f2(x: ArrayLike) -> ArrayLike:
    return numerical_hessian(f2, x)


def hess_f3(x: ArrayLike) -> ArrayLike:
    return numerical_hessian(f3, x)


def hess_f4(x: ArrayLike) -> ArrayLike:
    return numerical_hessian(f4, x)


def hess_f5(x: ArrayLike) -> ArrayLike:
    return numerical_hessian(f5, x)


def hess_f6(x: ArrayLike) -> ArrayLike:
    return numerical_hessian(f6, x)


def hess_f7(x: ArrayLike) -> ArrayLike:
    return numerical_hessian(f7, x)


def hess_f8(x: ArrayLike) -> ArrayLike:
    return numerical_hessian(f8, x)


# ---------------------------------------------------------------------------
# Реєстр функцій для вибору в GUI / движку
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetFunction:
    key: str
    name: str
    func: ScalarFunction
    grad: VectorFunction
    hess: MatrixFunction


FUNCTIONS: Dict[str, TargetFunction] = {
    "f1": TargetFunction(
        key="f1",
        name="f1(x1, x2) = (12 + x1^2 + (1 + x2^2)/x1^2 + ((x1*x2)^2 + 100)/(x1*x2)^4) / 10",
        func=f1,
        grad=grad_f1,
        hess=hess_f1,
    ),
    "f2": TargetFunction(
        key="f2",
        name="f2(x1, x2) = (x1 - x2)^2 + (x1 + x2 - 10)^2 / 9",
        func=f2,
        grad=grad_f2,
        hess=hess_f2,
    ),
    "f3": TargetFunction(
        key="f3",
        name="f3(x1, x2) = 5 * (x2 - 4*x1^3 + 3*x1)^2 + (x1 + 1)^2",
        func=f3,
        grad=grad_f3,
        hess=hess_f3,
    ),
    "f4": TargetFunction(
        key="f4",
        name="f4(x1, x2) = 5 * (x2 - 4*x1^3 + 3*x1)^2 + (x1 - 1)^2",
        func=f4,
        grad=grad_f4,
        hess=hess_f4,
    ),
    "f5": TargetFunction(
        key="f5",
        name="f5(x1, x2) = 100 * (x2 - x1^3 + x1)^2 + (x1 - 1)^2",
        func=f5,
        grad=grad_f5,
        hess=hess_f5,
    ),
    "f6": TargetFunction(
        key="f6",
        name="f6(x1, x2) = (0.01*(x1-3))^2 - (x2 - x1) + exp(20*(x2 - x1))",
        func=f6,
        grad=grad_f6,
        hess=hess_f6,
    ),
    "f7": TargetFunction(
        key="f7",
        name="f7(x1, x2) = 100 * (x2 - x1^2)^2 + (1 - x1)^2",
        func=f7,
        grad=grad_f7,
        hess=hess_f7,
    ),
    "f8": TargetFunction(
        key="f8",
        name="f8(x1, x2) = (x1 - 4)^2 + (x2 - 4)^2",
        func=f8,
        grad=grad_f8,
        hess=hess_f8,
    ),
}

__all__ = [
    "ArrayLike",
    "ScalarFunction",
    "VectorFunction",
    "MatrixFunction",
    "numerical_gradient",
    "numerical_hessian",
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8",
    "grad_f1", "grad_f2", "grad_f3", "grad_f4",
    "grad_f5", "grad_f6", "grad_f7", "grad_f8",
    "hess_f1", "hess_f2", "hess_f3", "hess_f4",
    "hess_f5", "hess_f6", "hess_f7", "hess_f8",
    "TargetFunction",
    "FUNCTIONS",
]
