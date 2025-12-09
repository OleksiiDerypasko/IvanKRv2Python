"""
line_search.py

Модуль одномірного пошуку (line search) вздовж заданого напрямку.

Ідея:
    - Працюємо з допоміжною функцією φ(α) = f(x_k + α p_k),
      але в цьому модулі ми оперуємо абстрактною скалярною
      функцією одного аргументу φ: float -> float.
    - Конкретні методи оптимізації (Коші, Флетчера–Рівза,
      Полака–Ріб’єра, Ньютона) створюють φ(α) самі, використовуючи
      свої eval_f(...), щоб коректно вести лічильники викликів f(x).

Підтримувані методи лінійного пошуку:

    1) метод дихотомії;
    2) метод розподілу інтервалу навпіл (interval halving);
    3) метод золотого перерізу;
    4) метод адаптації кроку;
    5) метод кубічної інтерполяції із чотирма точками;
    6) Armijo backtracking (для градієнтних методів).

Єдиний публічний інтерфейс:
    - LineSearchResult        – результат 1D-пошуку;
    - line_search_1d(...)     – виклик конкретного методу;
    - константи LINE_SEARCH_* – імена методів для options у Optimizer-ах.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Callable, Dict, Optional

import numpy as np
import warnings

# Скалярова функція від одного аргументу α
Scalar1DFunction = Callable[[float], float]


# ---------------------------------------------------------------------------
# Константи / "enum" для типів методів лінійного пошуку
# ---------------------------------------------------------------------------

LINE_SEARCH_DEFAULT = "default"  # "за замовчуванням" (можна прив'язати до Armijo/адаптації кроку)
LINE_SEARCH_DICHOTOMY = "dichotomy"
LINE_SEARCH_INTERVAL_HALVING = "interval_halving"
LINE_SEARCH_GOLDEN_SECTION = "golden_section"
LINE_SEARCH_STEP_ADAPTATION = "step_adaptation"
LINE_SEARCH_CUBIC_4POINT = "cubic_4point"
LINE_SEARCH_ARMIJO = "armijo_backtracking"

# Для типізації (можна використовувати в сигнатурах)
LineSearchMethod = str


# ---------------------------------------------------------------------------
# Результат одномірного пошуку
# ---------------------------------------------------------------------------

@dataclass
class LineSearchResult:
    """
    Результат роботи процедури одномірного пошуку.

    Атрибути:
        alpha       - знайдене значення параметра кроку α*;
        phi_value   - значення φ(α*) у цій точці;
        iterations  - кількість ітерацій 1D-алгоритму;
        func_evals  - кількість викликів φ під час пошуку;
        meta        - довільна службова інформація
                      (кінцевий інтервал, історія α, тощо).
    """
    alpha: float
    phi_value: float
    iterations: int
    func_evals: int
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Публічний інтерфейс line search
# ---------------------------------------------------------------------------

def line_search_1d(
    phi: Scalar1DFunction,
    a: float,
    b: float,
    method: LineSearchMethod = LINE_SEARCH_DEFAULT,
    tol: float = 1e-6,
    max_iter: int = 100,
    options: Optional[Dict[str, Any]] = None,
) -> LineSearchResult:
    """
    Виконати одномірний пошук мінімуму функції φ(α) на відрізку [a, b].

    Parameters
    ----------
    phi : Callable[[float], float]
        Цільова скалярна функція одного аргументу α.
        Зазвичай φ(α) = f(x_k + α p_k), де f – багатовимірна функція.

    a, b : float
        Початковий інтервал пошуку [a, b], де a < b.
        Для деяких методів (Armijo backtracking) інтервал використовується
        лише для інформації в meta і не впливає на алгоритм.

    method : LineSearchMethod
        Обраний метод лінійного пошуку.
        Доступні варіанти (див. константи LINE_SEARCH_*):
            - "default"
            - "dichotomy"
            - "interval_halving"
            - "golden_section"
            - "step_adaptation"
            - "cubic_4point"
            - "armijo_backtracking"

    tol : float
        Бажана точність за α (розмір інтервалу / кроку).

    max_iter : int
        Максимальна кількість ітерацій одномірного алгоритму.
        Для складених методів (step_adaptation, cubic_4point, armijo_backtracking)
        це приблизна верхня межа для основного циклу; внутрішні підкроки
        можуть виконати ще кілька ітерацій (див. meta).

    options : Optional[Dict[str, Any]]
        Додаткові параметри для конкретних методів
        (наприклад, внутрішні коефіцієнти, початкові значення тощо).

    Returns
    -------
    LineSearchResult
        Структура з α*, φ(α*), кількістю ітерацій та додатковими meta-даними.

    Notes
    -----
    На даному етапі реалізовані методи:
        - дихотомії;
        - розподілу інтервалу навпіл;
        - золотого перерізу;
        - адаптації кроку (expansion + локальне уточнення);
        - кубічної інтерполяції з 4 точками;
        - Armijo backtracking (для градієнтних методів).
    """
    # Для Armijo backtracking інтервал [a, b] не критичний, тому перевірку
    # робимо лише для інших методів.
    if method != LINE_SEARCH_ARMIJO and a >= b:
        raise ValueError(
            "line_search_1d: ліва межа інтервалу повинна бути меншою за праву (a < b)."
        )

    options = options or {}

    if method == LINE_SEARCH_DICHOTOMY:
        return _line_search_dichotomy(phi, a, b, tol, max_iter, options)

    if method == LINE_SEARCH_INTERVAL_HALVING:
        return _line_search_interval_halving(phi, a, b, tol, max_iter, options)

    if method == LINE_SEARCH_GOLDEN_SECTION:
        return _line_search_golden_section(phi, a, b, tol, max_iter, options)

    if method == LINE_SEARCH_ARMIJO:
        return _line_search_armijo_backtracking(phi, a, b, tol, max_iter, options)

    if method == LINE_SEARCH_STEP_ADAPTATION:
        return _line_search_step_adaptation(phi, a, b, tol, max_iter, options)

    if method == LINE_SEARCH_CUBIC_4POINT:
        return _line_search_cubic_4point(phi, a, b, tol, max_iter, options)

    # "default" або будь-яке невідоме значення – поки що теж NotImplemented,
    # щоб явно показати, що алгоритм ще не налаштований.
    raise NotImplementedError(
        f"Line search method '{method}' ще не реалізований."
    )


# ---------------------------------------------------------------------------
# Внутрішні реалізації конкретних методів
# ---------------------------------------------------------------------------

def _line_search_dichotomy(
    phi: Scalar1DFunction,
    a: float,
    b: float,
    tol: float,
    max_iter: int,
    options: Dict[str, Any],
) -> LineSearchResult:
    """
    Метод дихотомії для пошуку мінімуму φ(α) на [a, b].

    Припущення:
        - φ(α) неперервна і (бажано) унімодальна на [a, b];
        - на кожній ітерації ми викидаємо частину інтервалу, в якій
          мінімум бути не може.

    Ідея:
        - на кожній ітерації беремо середину m = (a + b) / 2;
        - обираємо малий зсув δ > 0;
        - обчислюємо φ у точках m - δ та m + δ;
        - звужуємо інтервал залежно від того, в якій половині менше значення.

    Параметри з options:
        delta : float (optional)
            Малий зсув δ. Якщо не задано, береться δ = 0.25 * tol.
    """
    left = float(a)
    right = float(b)

    # δ – малий зсув від середини
    delta = float(options.get("delta", 0.25 * tol if tol > 0 else 1e-8))
    if delta <= 0.0:
        delta = 1e-8

    iterations = 0
    func_evals = 0

    while (right - left) > tol and iterations < max_iter:
        iterations += 1

        mid = 0.5 * (left + right)

        # Гарантуємо, що δ не більший за чверть поточного інтервалу
        max_delta = 0.25 * (right - left)
        d = min(delta, max_delta)

        x1 = mid - d
        x2 = mid + d

        f1 = phi(x1)
        f2 = phi(x2)
        func_evals += 2

        if f1 < f2:
            # Мінімум у лівій половині
            right = x2
        else:
            # Мінімум у правій половині
            left = x1

    alpha_star = 0.5 * (left + right)
    phi_star = phi(alpha_star)
    func_evals += 1

    meta = {
        "method": LINE_SEARCH_DICHOTOMY,
        "interval": (left, right),
        "stopped_by": "tol" if (right - left) <= tol else "max_iter",
        "delta": delta,
    }

    return LineSearchResult(
        alpha=alpha_star,
        phi_value=phi_star,
        iterations=iterations,
        func_evals=func_evals,
        meta=meta,
    )


def _line_search_interval_halving(
    phi: Scalar1DFunction,
    a: float,
    b: float,
    tol: float,
    max_iter: int,
    options: Dict[str, Any],
) -> LineSearchResult:
    """
    Метод розподілу інтервалу навпіл (interval halving) для φ(α).

    Припущення:
        - φ(α) унімодальна на [a, b];
        - значення функції скінченні та обчислювані в усіх точках інтервалу.

    Класична схема:
        - маємо інтервал [a, b] і його середину m;
        - додатково розглядаємо точки x1 = (a + m) / 2 та x2 = (m + b) / 2;
        - порівнюючи φ(x1), φ(m), φ(x2), обираємо новий інтервал довжиною (b - a) / 2,
          у якому гарантовано лежить мінімум (для унімодальної функції).

    На кожній ітерації:
        - 3 виклики φ (x1, m, x2);
        - інтервал скорочується рівно вдвічі.
    """
    left = float(a)
    right = float(b)

    iterations = 0
    func_evals = 0

    while (right - left) > tol and iterations < max_iter:
        iterations += 1

        mid = 0.5 * (left + right)
        x1 = 0.5 * (left + mid)
        x2 = 0.5 * (mid + right)

        f1 = phi(x1)
        fm = phi(mid)
        f2 = phi(x2)
        func_evals += 3

        if f1 < fm:
            # Мінімум у [left, mid]
            right = mid
        elif f2 < fm:
            # Мінімум у [mid, right]
            left = mid
        else:
            # Мінімум у середині, звужуємо до [x1, x2]
            left = x1
            right = x2

    alpha_star = 0.5 * (left + right)
    phi_star = phi(alpha_star)
    func_evals += 1

    meta = {
        "method": LINE_SEARCH_INTERVAL_HALVING,
        "interval": (left, right),
        "stopped_by": "tol" if (right - left) <= tol else "max_iter",
    }

    return LineSearchResult(
        alpha=alpha_star,
        phi_value=phi_star,
        iterations=iterations,
        func_evals=func_evals,
        meta=meta,
    )


def _line_search_golden_section(
    phi: Scalar1DFunction,
    a: float,
    b: float,
    tol: float,
    max_iter: int,
    options: Dict[str, Any],
) -> LineSearchResult:
    """
    Метод золотого перерізу для пошуку мінімуму φ(α) на [a, b].

    Припущення:
        - φ(α) неперервна та унімодальна на [a, b];
        - значення функції можна обчислити в будь-якій точці інтервалу.

    Ідея:
        - на кожній ітерації тримаємо два внутрішні пункти c і d:
              c = a + (1 - 1/φ) * (b - a),
              d = a + 1/φ * (b - a),
          де φ ≈ (1 + sqrt(5)) / 2 – число золотого перерізу;
        - порівнюємо φ(c) і φ(d) та звужуємо інтервал, зберігаючи одну
          з внутрішніх точок, щоб мінімізувати кількість викликів φ;
        - довжина інтервалу зменшується геометрично.
    """
    left = float(a)
    right = float(b)

    # Кон'югований коефіцієнт золотого перерізу
    inv_phi = (sqrt(5.0) - 1.0) / 2.0      # ≈ 0.618...
    inv_phi_sq = (3.0 - sqrt(5.0)) / 2.0   # ≈ 0.382..., 1/φ^2

    h = right - left
    if h <= tol:
        alpha_star = 0.5 * (left + right)
        phi_star = phi(alpha_star)
        return LineSearchResult(
            alpha=alpha_star,
            phi_value=phi_star,
            iterations=0,
            func_evals=1,
            meta={
                "method": LINE_SEARCH_GOLDEN_SECTION,
                "interval": (left, right),
                "stopped_by": "initial_tol",
            },
        )

    # Початкові внутрішні точки
    c = left + inv_phi_sq * h
    d = left + inv_phi * h

    fc = phi(c)
    fd = phi(d)
    func_evals = 2
    iterations = 0

    while h > tol and iterations < max_iter:
        iterations += 1

        if fc < fd:
            # Мінімум у [left, d]
            right = d
            d = c
            fd = fc
            h = right - left
            c = left + inv_phi_sq * h
            fc = phi(c)
            func_evals += 1
        else:
            # Мінімум у [c, right]
            left = c
            c = d
            fc = fd
            h = right - left
            d = left + inv_phi * h
            fd = phi(d)
            func_evals += 1

    alpha_star = 0.5 * (left + right)
    phi_star = phi(alpha_star)
    func_evals += 1

    meta = {
        "method": LINE_SEARCH_GOLDEN_SECTION,
        "interval": (left, right),
        "stopped_by": "tol" if h <= tol else "max_iter",
    }

    return LineSearchResult(
        alpha=alpha_star,
        phi_value=phi_star,
        iterations=iterations,
        func_evals=func_evals,
        meta=meta,
    )


def _line_search_armijo_backtracking(
    phi: Scalar1DFunction,
    a: float,
    b: float,
    tol: float,
    max_iter: int,
    options: Dict[str, Any],
) -> LineSearchResult:
    """
    Armijo backtracking line search для φ(α).

    Припущення:
        - φ(α) = f(x_k + α p_k) для деякої фіксованої точки x_k та напряму p_k;
        - напрямок p_k є напрямком спуску, тобто φ'(0) = ∇f(x_k)^T p_k < 0;
        - значення φ(α) можна обчислити для всіх α ≥ 0 в робочому інтервалі.

    Алгоритм:
        - стартуємо з α0 > 0;
        - поки не виконується умова Арміхо
              φ(α) <= φ(0) + c1 * α * φ'(0),
          множимо α на tau (0 < tau < 1);
        - якщо α стає меншим за min_alpha або перевищено ліміт кроків,
          зупиняємося з останнім кандидатом.

    Параметри з options:
        f0                    : значення φ(0) = f(x_k) (обов'язково);
        directional_derivative : значення φ'(0) = ∇f(x_k)^T p_k (обов'язково);
        alpha0                : початковий крок (default: 1.0);
        tau                   : множник для зменшення кроку, 0 < tau < 1
                                (default: 0.5);
        c1                    : константа Арміхо (default: 1e-4);
        max_backtracking      : максимальна кількість кроків backtracking
                                (default: max_iter);
        min_alpha             : мінімально допустимий крок (default: 1e-12).

    ПРИМІТКА:
        Інтервал [a, b] у цьому методі використовується лише для інформації
        в meta["interval"] і не впливає на сам алгоритм.
    """
    if "f0" not in options or "directional_derivative" not in options:
        raise ValueError(
            "Armijo backtracking потребує 'f0' та 'directional_derivative' "
            "у параметрі options."
        )

    f0 = float(options["f0"])
    dphi0 = float(options["directional_derivative"])

    alpha = float(options.get("alpha0", 1.0))
    tau = float(options.get("tau", 0.5))
    c1 = float(options.get("c1", 1e-4))
    max_bt = int(options.get("max_backtracking", max_iter))
    min_alpha = float(options.get("min_alpha", 1e-12))

    if max_bt <= 0:
        max_bt = max_iter if max_iter > 0 else 1

    n_steps_limit = max(1, min(max_iter, max_bt))

    iterations = 0
    func_evals = 0
    accepted = False
    reason = "max_backtracking"

    alpha_curr = alpha
    phi_curr = f0

    for _ in range(n_steps_limit):
        iterations += 1

        alpha_curr = alpha
        phi_curr = phi(alpha_curr)
        func_evals += 1

        if phi_curr <= f0 + c1 * alpha_curr * dphi0:
            accepted = True
            reason = "armijo"
            break

        alpha *= tau
        if alpha < min_alpha:
            reason = "min_alpha"
            break

    alpha_star = alpha_curr
    phi_star = phi_curr

    meta = {
        "method": LINE_SEARCH_ARMIJO,
        "interval": (a, b),
        "stopped_by": reason,
        "accepted": accepted,
        "f0": f0,
        "directional_derivative": dphi0,
        "alpha0": float(options.get("alpha0", 1.0)),
        "tau": tau,
        "c1": c1,
        "max_backtracking": max_bt,
        "min_alpha": min_alpha,
    }

    return LineSearchResult(
        alpha=alpha_star,
        phi_value=phi_star,
        iterations=iterations,
        func_evals=func_evals,
        meta=meta,
    )


def _line_search_step_adaptation(
    phi: Scalar1DFunction,
    a: float,
    b: float,
    tol: float,
    max_iter: int,
    options: Dict[str, Any],
) -> LineSearchResult:
    """
    Метод адаптації кроку (expansion + локальне уточнення).

    Припущення:
        - φ(α) неперервна на [a, b];
        - вздовж обраного напрямку існує локальний мінімум у [a, b];
        - поведінка φ(α) "розумна": якщо рухатися в напрямку зменшення φ,
          то після певного моменту значення починає зростати (локальна
          унімодальність уздовж напрямку).

    Опис алгоритму (спрощено):
        1) Стартуємо з α0 (за замовчуванням 0, проектований на [a, b]).
        2) Робимо пробний крок вправо (α0 + h0). Якщо φ зменшилась, рухаємося
           далі в цьому напрямку, поступово збільшуючи крок (expansion).
           Якщо ні – пробуємо крок вліво (α0 - h0).
        3) Як тільки отримуємо "поворот" (φ перестає зменшуватись або
           досягаємо межі [a, b]), фіксуємо дужку [α_L, α_R], у якій
           лежить локальний мінімум.
        4) Додатково уточнюємо мінімум у цій дужці методом золотого перерізу.

    Основний критерій зупинки для фази expansion – max_iter; для уточнення
    використовується власний ліміт і tol по довжині інтервалу.

    Параметри з options:
        alpha0        : float, початкова точка (за замовчуванням 0.0, потім
                        проектується на [a, b]);
        initial_step  : float, початковий крок h0 (за замовчуванням 0.5*(b - a));
        expand_factor : float > 1, коефіцієнт збільшення кроку (дефолт 2.0).
    """
    left = float(a)
    right = float(b)
    if not (left < right):
        raise ValueError("line_search_step_adaptation: потрібно a < b.")

    options = options or {}

    # Початкова точка
    alpha0 = float(options.get("alpha0", 0.0))
    if alpha0 < left:
        alpha0 = left
    if alpha0 > right:
        alpha0 = right

    phi0 = phi(alpha0)
    func_evals = 1
    iterations = 0

    initial_step = float(options.get("initial_step", 0.5 * (right - left)))
    if initial_step <= 0.0:
        initial_step = 0.5 * (right - left)

    expand_factor = float(options.get("expand_factor", 2.0))
    if expand_factor <= 1.0:
        expand_factor = 2.0

    # Спробуємо спочатку рух вправо
    direction = +1.0
    alpha1 = alpha0 + initial_step
    if alpha1 > right:
        alpha1 = right

    phi1 = phi(alpha1)
    func_evals += 1

    bracket_left = left
    bracket_right = right
    used_fallback = False

    if alpha1 != alpha0 and phi1 < phi0:
        # Маємо напрямок зменшення вправо
        direction = +1.0
        alpha_prev, phi_prev = alpha0, phi0
        alpha_curr, phi_curr = alpha1, phi1
        bracket_left, bracket_right = sorted((alpha_prev, alpha_curr))
    else:
        # Пробуємо вліво
        alpha1 = alpha0 - initial_step
        if alpha1 < left:
            alpha1 = left

        if alpha1 != alpha0:
            phi1 = phi(alpha1)
            func_evals += 1
            if phi1 < phi0:
                direction = -1.0
                alpha_prev, phi_prev = alpha0, phi0
                alpha_curr, phi_curr = alpha1, phi1
                bracket_left, bracket_right = sorted((alpha_prev, alpha_curr))
            else:
                used_fallback = True
        else:
            used_fallback = True

    # Якщо жоден напрямок не дав зменшення – просто робимо золотий переріз на [a, b]
    if used_fallback:
        golden_res = _line_search_golden_section(phi, left, right, tol, max_iter, options)
        meta = dict(golden_res.meta)
        meta.update(
            {
                "method": LINE_SEARCH_STEP_ADAPTATION,
                "inner_method": LINE_SEARCH_GOLDEN_SECTION,
                "fallback_to_golden": True,
                "bracket": (left, right),
                "alpha0": alpha0,
            }
        )
        return LineSearchResult(
            alpha=golden_res.alpha,
            phi_value=golden_res.phi_value,
            iterations=iterations + golden_res.iterations,
            func_evals=func_evals + golden_res.func_evals,
            meta=meta,
        )

    # Фаза розширення кроку (expansion)
    h = initial_step
    while iterations < max_iter:
        iterations += 1

        alpha_next = alpha_curr + direction * h

        if alpha_next < left or alpha_next > right:
            # Дісталися межі інтервалу – фіксуємо дужку
            if direction > 0:
                bracket_left, bracket_right = min(alpha_prev, alpha_curr), right
            else:
                bracket_left, bracket_right = left, max(alpha_prev, alpha_curr)
            break

        phi_next = phi(alpha_next)
        func_evals += 1

        if phi_next < phi_curr:
            # Успішний крок – рухаємося далі й збільшуємо крок
            alpha_prev, phi_prev = alpha_curr, phi_curr
            alpha_curr, phi_curr = alpha_next, phi_next
            bracket_left, bracket_right = sorted((alpha_prev, alpha_curr))
            h *= expand_factor
        else:
            # φ почала зростати – мінімум десь між попередньою точкою та новою
            bracket_left, bracket_right = sorted((alpha_prev, alpha_next))
            break

    # Локальне уточнення на отриманому інтервалі через золотий переріз
    golden_res = _line_search_golden_section(
        phi, bracket_left, bracket_right, tol, max_iter, options
    )

    meta = dict(golden_res.meta)
    meta.update(
        {
            "method": LINE_SEARCH_STEP_ADAPTATION,
            "inner_method": LINE_SEARCH_GOLDEN_SECTION,
            "fallback_to_golden": False,
            "bracket": (bracket_left, bracket_right),
            "alpha0": alpha0,
        }
    )

    return LineSearchResult(
        alpha=golden_res.alpha,
        phi_value=golden_res.phi_value,
        iterations=iterations + golden_res.iterations,
        func_evals=func_evals + golden_res.func_evals,
        meta=meta,
    )


def _line_search_cubic_4point(
    phi: Scalar1DFunction,
    a: float,
    b: float,
    tol: float,
    max_iter: int,
    options: Dict[str, Any],
) -> LineSearchResult:
    """
    Метод кубічної інтерполяції із чотирма точками.

    Припущення:
        - φ(α) неперервна і достатньо гладка (≈ C^2) на [a, b];
        - на [a, b] є єдиний локальний мінімум (унімодальність уздовж напрямку);
        - значення φ(α) скінченні й не є NaN/inf.

    Ідея:
        - підтримуємо 4 точки α0 < α1 < α2 < α3 і значення φ(α_i);
        - будуємо кубічний поліном p(α), що інтерполює ці 4 точки;
        - знаходимо мінімум p(α) як дійсний корінь p'(α) = 0 всередині (α0, α3),
          або беремо середину інтервалу, якщо коректного кореня немає;
        - додаємо нову точку α_new до набору, формуємо нові 4 точки так,
          щоб зберегти область, де φ мала (залишаємо найкращу точку і її
          найближчих сусідів);
        - зупинка за довжиною інтервалу [α0, α3] або max_iter.

    Зауваження:
        - метод чутливіший до числової нестійкості, ніж золотий переріз;
        - у реалізації використовується numpy.polyfit + numpy.roots;
        - у разі чисельних проблем можливий fallback на золотий переріз
          (див. meta["fallback_to_golden"]).
    """
    left = float(a)
    right = float(b)
    if not (left < right):
        raise ValueError("line_search_cubic_4point: потрібно a < b.")

    # Початкові 4 точки: кінці та дві внутрішні, рівновіддалені
    x0 = left
    x3 = right
    h = (right - left) / 3.0
    x1 = left + h
    x2 = left + 2.0 * h

    f0 = phi(x0)
    f1 = phi(x1)
    f2 = phi(x2)
    f3 = phi(x3)
    func_evals = 4
    iterations = 0

    xs_history = [(x0, x1, x2, x3)]
    eps = 1e-14

    while (x3 - x0) > tol and iterations < max_iter:
        iterations += 1

        xs = np.array([x0, x1, x2, x3], dtype=float)
        ys = np.array([f0, f1, f2, f3], dtype=float)

        # Побудова кубічного інтерполянта p(α) через 4 точки
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=np.RankWarning)
                coeffs = np.polyfit(xs, ys, 3)
        except Exception:
            # Якщо щось пішло не так – відкотитися до золотого перерізу
            golden_res = _line_search_golden_section(
                phi, x0, x3, tol, max_iter - iterations, options
            )
            meta = dict(golden_res.meta)
            meta.update(
                {
                    "method": LINE_SEARCH_CUBIC_4POINT,
                    "inner_method": LINE_SEARCH_GOLDEN_SECTION,
                    "fallback_to_golden": True,
                    "initial_interval": (a, b),
                }
            )
            return LineSearchResult(
                alpha=golden_res.alpha,
                phi_value=golden_res.phi_value,
                iterations=iterations + golden_res.iterations,
                func_evals=func_evals + golden_res.func_evals,
                meta=meta,
            )

        a3, a2, a1, a0 = coeffs

        # Знаходимо стаціонарні точки p'(α) = 0
        roots = None
        if abs(a3) < eps:
            if abs(a2) < eps:
                # Поліном майже лінійний або константний – беремо середину
                candidate = 0.5 * (x0 + x3)
            else:
                roots = np.roots([2.0 * a2, a1])
        else:
            roots = np.roots([3.0 * a3, 2.0 * a2, a1])

        if roots is not None:
            real_roots = [r.real for r in roots if abs(r.imag) < 1e-12]
            real_roots = [r for r in real_roots if x0 < r < x3]
            if real_roots:
                xs4 = np.array([x0, x1, x2, x3], dtype=float)
                fs4 = np.array([f0, f1, f2, f3], dtype=float)
                best_idx = int(np.argmin(fs4))
                x_best = float(xs4[best_idx])
                # Обираємо корінь, найближчий до поточної найкращої точки
                candidate = min(real_roots, key=lambda r: abs(r - x_best))
            else:
                candidate = 0.5 * (x0 + x3)

        # Додатковий захист – не виходити за межі інтервалу
        if candidate <= x0 or candidate >= x3:
            candidate = 0.5 * (x0 + x3)

        f_new = phi(candidate)
        func_evals += 1

        # Оновлюємо набір точок (4 з 5): завжди залишаємо найкращу точку
        # та найближчі до неї по x, плюс candidate.
        xs5 = [x0, x1, x2, x3, candidate]
        fs5 = [f0, f1, f2, f3, f_new]
        pairs = sorted(zip(xs5, fs5), key=lambda t: t[0])

        # Прибираємо дублікати по x (залишаємо кращі за значенням φ)
        dedup: list[tuple[float, float]] = []
        for x_val, f_val in pairs:
            if not dedup or abs(x_val - dedup[-1][0]) > 1e-14:
                dedup.append((x_val, f_val))
            else:
                if f_val < dedup[-1][1]:
                    dedup[-1] = (x_val, f_val)
        pairs = dedup

        if len(pairs) < 4:
            # Недостатньо точок для наступної кубічної побудови – завершуємо
            break

        xs_vals = [p[0] for p in pairs]
        fs_vals = [p[1] for p in pairs]

        idx_best = min(range(len(pairs)), key=lambda i: fs_vals[i])
        idx_cand = min(range(len(pairs)), key=lambda i: abs(xs_vals[i] - candidate))

        chosen = {idx_best, idx_cand}
        # Додаємо найближчі до найкращої точки сусіди, поки не наберемо 4 точки
        while len(chosen) < 4:
            remaining = [i for i in range(len(pairs)) if i not in chosen]
            x_best = xs_vals[idx_best]
            j = min(remaining, key=lambda i: abs(xs_vals[i] - x_best))
            chosen.add(j)

        sel = sorted(chosen)
        window = [pairs[i] for i in sel]

        (x0, f0), (x1, f1), (x2, f2), (x3, f3) = window
        xs_history.append((x0, x1, x2, x3))

    xs_final = np.array([x0, x1, x2, x3], dtype=float)
    fs_final = np.array([f0, f1, f2, f3], dtype=float)
    best_idx = int(np.argmin(fs_final))
    alpha_star = float(xs_final[best_idx])
    phi_star = float(fs_final[best_idx])

    meta = {
        "method": LINE_SEARCH_CUBIC_4POINT,
        "interval": (x0, x3),
        "stopped_by": "tol" if (x3 - x0) <= tol else "max_iter",
        "xs_history": xs_history,
    }

    return LineSearchResult(
        alpha=alpha_star,
        phi_value=phi_star,
        iterations=iterations,
        func_evals=func_evals,
        meta=meta,
    )


__all__ = [
    "Scalar1DFunction",
    "LineSearchResult",
    "LineSearchMethod",
    "LINE_SEARCH_DEFAULT",
    "LINE_SEARCH_DICHOTOMY",
    "LINE_SEARCH_INTERVAL_HALVING",
    "LINE_SEARCH_GOLDEN_SECTION",
    "LINE_SEARCH_STEP_ADAPTATION",
    "LINE_SEARCH_CUBIC_4POINT",
    "LINE_SEARCH_ARMIJO",
    "line_search_1d",
]
