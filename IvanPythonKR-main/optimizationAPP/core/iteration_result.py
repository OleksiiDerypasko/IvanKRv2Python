"""
iteration_result.py

Структура даних для представлення результатів окремих ітерацій
оптимізаційного процесу. Використовується як у движку, так і в GUI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

ArrayLike = np.ndarray


@dataclass
class IterationResult:
    """
    Опис однієї ітерації оптимізаційного процесу.

    Атрибути:
        index      - номер ітерації (0, 1, 2, ...)
        x          - значення вектора змінних x_k
        f          - значення функції f(x_k)
        step_norm  - норма кроку ||x_k - x_{k-1}|| (для k=0 = 0.0)
        meta       - довільна додаткова інформація (α, β, тип кроку, ...)
    """
    index: int
    x: np.ndarray
    f: float
    step_norm: float
    meta: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "ArrayLike",
    "IterationResult",
]
