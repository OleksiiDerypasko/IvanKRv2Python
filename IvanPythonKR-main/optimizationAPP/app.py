"""
app.py

Контролер для GUI-застосунку мінімізації багатовимірних функцій.

Зв'язує:
    - ui.MainWindow (PyQt6)
    - core.OptimizationEngine
    - конкретні методи оптимізації (Cauchy, Fletcher–Reeves, Polak–Ribiere,
      Newton, Nelder–Мід, Hook–Jeeves)
    - core.functions.FUNCTIONS

Функціонал:
    - реагує на сигнал MainWindow.optimizationRequested(OptimizationConfig);
    - створює відповідний Optimizer;
    - запускає OptimizationEngine;
    - у callback оновлює таблицю ітерацій;
    - після завершення — оновлює графік f(k) та contour+траєкторію;
    - у режимі "Запустити всі методи" показує зведену таблицю результатів,
      включно з комбінаціями (градієнтний метод × метод лінійного пошуку).
"""

from __future__ import annotations

import sys
from typing import List, Optional, Tuple

from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.control_panel import OptimizationConfig
from ui.styles import apply_app_style
from ui.dialogs import show_error, show_summary

from core.engine import OptimizationEngine, OptimizationRunResult
from core.iteration_result import IterationResult
from core.functions import FUNCTIONS
from core.cauchy import CauchyMethod
from core.fletcher_reeves import FletcherReevesMethod
from core.polak_ribiere import PolakRibiereMethod
from core.newton import NewtonMethod
from core.nelder_mead import NelderMeadMethod
from core.hook_jeeves import HookJeevesMethod
from core.results_summary import ResultsSummary


# ---------------------------------------------------------------------------
# Back-compat: локальний враппер під старий інтерфейс show_error_dialog
# ---------------------------------------------------------------------------

def show_error_dialog(parent, title: str, message: str) -> None:
    """
    Сумісний з попереднім API show_error_dialog(parent, title, message),
    але всередині просто викликає ui.dialogs.show_error.
    """
    show_error(parent, message, title=title)


# ---------------------------------------------------------------------------
# Константи для sweep по методах лінійного пошуку
# ---------------------------------------------------------------------------

# Ключі градієнтних методів, для яких має сенс ганяти різні line search
GRADIENT_METHOD_KEYS = [
    "cauchy",
    "fletcher_reeves",
    "polak_ribiere",
    "newton",
]

# Варіанти лінійного пошуку для sweep у "Запустити всі методи"
# (GUI-ключ, людино-зрозуміла назва)
LINE_SEARCH_VARIANTS_FOR_SWEEP: List[Tuple[str, str]] = [
    ("dichotomy", "дихотомія"),
    ("interval_halving", "розподіл інтервалу навпіл"),
    ("golden_section", "золотий переріз"),
    ("cubic4", "кубічна інтерполяція (4 точки)"),
]


# ---------------------------------------------------------------------------
# Хелпер для line search
# ---------------------------------------------------------------------------

def _normalize_line_search_key(line_search_key: Optional[str]) -> Optional[str]:
    """
    Нормалізувати ключ методу лінійного пошуку з GUI до того,
    що очікують core-методи.

    GUI варіанти:
        "default", "dichotomy", "interval_halving",
        "golden_section", "cubic4"

    В core.line_search:
        "dichotomy", "interval_halving", "golden_section",
        "step_adaptation", "cubic_4point", "armijo_backtracking"
    """
    if not line_search_key:
        return None

    key = line_search_key.lower().strip()

    if key in ("default",):
        # "default" означає: не задаємо явно, метод сам вирішить
        return None
    if key == "dichotomy":
        return "dichotomy"
    if key == "interval_halving":
        return "interval_halving"
    if key == "golden_section":
        return "golden_section"
    if key in ("cubic4", "cubic_4point"):
        return "cubic_4point"

    # Невідомий ключ — поводимось як "default"
    return None


# ---------------------------------------------------------------------------
# Допоміжна фабрика Optimizer-ів
# ---------------------------------------------------------------------------

def create_optimizer(
    method_key: str,
    func_key: str,
    eps: float,
    line_search_key: Optional[str] = None,
):
    """
    Створити відповідний Optimizer по ключу методу та функції.

    method_key:
        "cauchy", "fletcher_reeves", "polak_ribiere",
        "newton", "nelder_mead", "hook_jeeves"

    func_key:
        "f1".."f8" — ключ у core.functions.FUNCTIONS

    eps:
        Загальна точність; далі мапимо її на grad_tol / min_step / тощо.

    line_search_key:
        Ключ методу лінійного пошуку з GUI:
            "default", "dichotomy", "interval_halving",
            "golden_section", "cubic4"
        Для direct-search методів ігнорується.
    """
    tf = FUNCTIONS[func_key]
    ls_method = _normalize_line_search_key(line_search_key)

    # -------- Градієнтні методи --------
    if method_key == "cauchy":
        options = {
            "grad_tol": eps,
        }
        if ls_method is not None:
            options["line_search"] = ls_method
        return CauchyMethod(
            func=tf.func,
            grad=tf.grad,
            options=options,
        )

    if method_key == "fletcher_reeves":
        options = {
            "grad_tol": eps,
        }
        if ls_method is not None:
            options["line_search"] = ls_method
        return FletcherReevesMethod(
            func=tf.func,
            grad=tf.grad,
            options=options,
        )

    if method_key == "polak_ribiere":
        options = {
            "grad_tol": eps,
        }
        if ls_method is not None:
            options["line_search"] = ls_method
        return PolakRibiereMethod(
            func=tf.func,
            grad=tf.grad,
            options=options,
        )

    # -------- Ньютон --------
    if method_key == "newton":
        options = {
            "grad_tol": eps,
            # use_line_search за замовчуванням True у NewtonMethod,
            # тому явно не чіпаємо, щоб зберегти поточну поведінку.
        }
        if ls_method is not None:
            # тільки якщо користувач явно вибрав інший метод, передаємо його
            options["line_search"] = ls_method

        return NewtonMethod(
            func=tf.func,
            grad=tf.grad,
            hess=tf.hess,
            options=options,
        )

    # -------- Direct-search методи (без line search) --------
    if method_key == "nelder_mead":
        # direct search без градієнта
        return NelderMeadMethod(
            func=tf.func,
            options={"initial_simplex_scale": 0.1},
        )

    if method_key == "hook_jeeves":
        # direct search без градієнта
        return HookJeevesMethod(
            func=tf.func,
            options={"initial_step": 1.0, "min_step": eps},
        )

    raise ValueError(f"Невідомий метод оптимізації: {method_key}")


# ---------------------------------------------------------------------------
# Контролер
# ---------------------------------------------------------------------------

class OptimizationController:
    """
    Зв'язує MainWindow та OptimizationEngine.

    Схема:
        GUI (MainWindow) --[OptimizationConfig]--> Controller
        Controller -- створює Optimizer, запускає Engine
        Engine -- через callback -> Controller -> MainWindow.add_iteration_row(...)
        Після завершення: MainWindow.update_fk_plot(...) + update_contour_trajectory(...)
    """

    def __init__(self, window: MainWindow, engine: OptimizationEngine) -> None:
        self.window = window
        self.engine = engine

        # Підписуємося на сигнал від GUI
        self.window.optimizationRequested.connect(self.on_optimization_requested)

    # ------------------------------------------------------------------
    # Валідація вхідних даних
    # ------------------------------------------------------------------

    def _validate_config(self, cfg: OptimizationConfig) -> bool:
        """
        Перевіряє коректність введених даних перед запуском движка.

        Якщо щось не так — показує діалог помилки та повертає False.
        """
        x0 = cfg.x0
        if x0 is None or len(x0) != 2:
            show_error_dialog(
                self.window,
                "Некоректна початкова точка",
                "Початкова точка x₀ повинна мати два компоненти (x1, x2).",
            )
            self.window.statusBar().showMessage("Помилка: некоректна початкова точка")
            return False

        x1, x2 = float(x0[0]), float(x0[1])

        # Спеціальні обмеження для f1:
        #   f1 містить ділення на x1^2 та (x1 * x2)^4,
        #   тому заборонено x1 = 0 або x1 * x2 = 0.
        if cfg.function_key == "f1":
            if abs(x1) < 1e-12 or abs(x1 * x2) < 1e-12:
                show_error_dialog(
                    self.window,
                    "Некоректна початкова точка для f1",
                    (
                        "Функція f1 містить ділення на x1² та (x1·x2)⁴, "
                        "тому заборонено x1 = 0 або x1·x2 = 0.\n\n"
                        "Оберіть, будь ласка, іншу стартову точку, "
                        "наприклад x₀ = (1, 1)."
                    ),
                )
                self.window.statusBar().showMessage(
                    "Помилка: некоректна початкова точка для f1"
                )
                return False

        # eps та max_iter уже обмежені діапазоном спінбоксів, але на всяк:
        if cfg.eps <= 0.0:
            show_error_dialog(
                self.window,
                "Некоректне значення eps",
                "Точність eps повинна бути додатною.",
            )
            self.window.statusBar().showMessage("Помилка: eps має бути > 0")
            return False

        if cfg.max_iter <= 0:
            show_error_dialog(
                self.window,
                "Некоректне значення max_iter",
                "Максимальна кількість ітерацій повинна бути додатною.",
            )
            self.window.statusBar().showMessage(
                "Помилка: max_iter має бути додатним"
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Обробник сигналу від GUI
    # ------------------------------------------------------------------

    def on_optimization_requested(self, cfg: OptimizationConfig) -> None:
        """
        Головний вхід: натиснута кнопка "Запустити" в GUI.
        """
        # 1) валідовані вхідні дані
        if not self._validate_config(cfg):
            return

        func_key = cfg.function_key

        if func_key not in FUNCTIONS:
            show_error_dialog(
                self.window,
                "Функція не знайдена",
                f"Функція з ключем '{func_key}' не знайдена.",
            )
            self.window.statusBar().showMessage(
                f"Помилка: функція '{func_key}' не знайдена"
            )
            return

        if cfg.run_all_methods:
            # Запустити всі методи для обраної функції та побудувати зведення
            self._run_all_methods_for_function(cfg)
        else:
            # Запустити тільки один обраний метод
            self._run_single_method(cfg)

    # ------------------------------------------------------------------
    # Запуск одного методу
    # ------------------------------------------------------------------

    def _run_single_method(self, cfg: OptimizationConfig) -> None:
        func_key = cfg.function_key
        method_key = cfg.method_key

        tf = FUNCTIONS[func_key]

        try:
            optimizer = create_optimizer(
                method_key=method_key,
                func_key=func_key,
                eps=cfg.eps,
                line_search_key=cfg.line_search_key,
            )
        except ValueError as exc:
            show_error_dialog(self.window, "Помилка конфігурації методу", str(exc))
            self.window.statusBar().showMessage(str(exc))
            # скидаємо статистику під таблицею
            try:
                self.window.update_run_stats(None, None, None)
            except AttributeError:
                pass
            return

        # Перед запуском таблицю вже чистив MainWindow, але на всяк випадок:
        self.window.clear_iterations_table()

        # Callback від движка — на кожній ітерації додаємо рядок у таблицю
        def iteration_callback(it: IterationResult) -> None:
            # Прагнемо виводити "крок" — якщо метод віддає alpha в meta, беремо його,
            # інакше — норму кроку.
            step_value = it.meta.get("alpha", it.step_norm)
            self.window.add_iteration_row(
                k=it.index,
                x=it.x,
                f_val=it.f,
                step_value=step_value,
            )

        # Запуск движка з захистом від винятків
        try:
            result: OptimizationRunResult = self.engine.run(
                optimizer=optimizer,
                x0=cfg.x0,
                max_iter=cfg.max_iter,
                tol_step=cfg.eps,
                tol_f=cfg.eps,
                callback=iteration_callback,
            )
        except Exception as exc:  # noqa: BLE001 — тут навмисно ловимо все
            show_error_dialog(
                self.window,
                "Помилка під час оптимізації",
                f"Під час виконання методу '{method_key}' виникла помилка:\n\n{exc}",
            )
            self.window.statusBar().showMessage(f"Помилка під час оптимізації: {exc}")
            # скидаємо статистику
            try:
                self.window.update_run_stats(None, None, None)
            except AttributeError:
                pass
            return

        # Оновлюємо графіки
        self.window.update_fk_plot(result.iterations)
        self.window.update_contour_trajectory(tf.func, result.iterations)

        # Оновлюємо статистику під таблицею:
        #   - кількість викликів f(x)
        #   - кінцевий крок пошуку (alpha або норма кроку)
        #   - кількість зовнішніх ітерацій
        last_step = None
        if result.iterations:
            last_it = result.iterations[-1]
            last_step = last_it.meta.get("alpha", last_it.step_norm)

        n_outer_iters = result.n_iter

        try:
            self.window.update_run_stats(result.func_evals, last_step, n_outer_iters)
        except AttributeError:
            # якщо старе вікно без цього API — просто ігноруємо
            pass

        # Повідомлення в статус-бар
        msg = (
            f"Метод: {result.method_name}, зупинка: {result.stopped_by}, "
            f"ітерацій: {result.n_iter}, f* = {result.f_star:.6e}, "
            f"x* = {result.x_star.tolist()}"
        )
        self.window.statusBar().showMessage(msg)

    # ------------------------------------------------------------------
    # Запуск усіх методів для однієї функції (зведена таблиця)
    # ------------------------------------------------------------------

    def _run_all_methods_for_function(self, cfg: OptimizationConfig) -> None:
        """
        Запустити всі методи для вибраної функції.

        У GUI:
            - у таблиці після завершення буде показаний останній базовий метод;
            - окремим діалогом відкриється зведена таблиця для всіх методів,
              включно з комбінаціями (градієнтний метод × метод лінійного пошуку).
        """
        func_key = cfg.function_key
        tf = FUNCTIONS[func_key]

        # Базові методи, які й так були
        base_method_keys: List[str] = [
            "cauchy",
            "fletcher_reeves",
            "polak_ribiere",
            "newton",
            "nelder_mead",
            "hook_jeeves",
        ]

        summary = ResultsSummary()

        # ------------------ 1) Базовий прогін усіх методів ------------------
        for mkey in base_method_keys:
            try:
                optimizer = create_optimizer(
                    method_key=mkey,
                    func_key=func_key,
                    eps=cfg.eps,
                    line_search_key=cfg.line_search_key,
                )
            except ValueError:
                continue

            try:
                result: OptimizationRunResult = self.engine.run(
                    optimizer=optimizer,
                    x0=cfg.x0,
                    max_iter=cfg.max_iter,
                    tol_step=cfg.eps,
                    tol_f=cfg.eps,
                    callback=None,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Метод {mkey} для {func_key} завершився помилкою: {exc}")
                continue

            # Тут методи йдуть як є: "Метод Коші", "Метод Ньютона" тощо
            summary.add_run(result)

        # ------------------ 2) Додаткові прогін по line search ------------------
        # Для кожного градієнтного методу ганяємо всі 1D-пошуки з LINE_SEARCH_VARIANTS_FOR_SWEEP
        for mkey in GRADIENT_METHOD_KEYS:
            for ls_gui_key, ls_label in LINE_SEARCH_VARIANTS_FOR_SWEEP:
                try:
                    optimizer = create_optimizer(
                        method_key=mkey,
                        func_key=func_key,
                        eps=cfg.eps,
                        line_search_key=ls_gui_key,
                    )
                except ValueError:
                    continue

                try:
                    result_ls: OptimizationRunResult = self.engine.run(
                        optimizer=optimizer,
                        x0=cfg.x0,
                        max_iter=cfg.max_iter,
                        tol_step=cfg.eps,
                        tol_f=cfg.eps,
                        callback=None,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[WARN] Метод {mkey} (line_search={ls_gui_key}) "
                        f"для {func_key} завершився помилкою: {exc}"
                    )
                    continue

                # Створюємо "декорований" результат з розширеною назвою методу,
                # щоб у зведеній таблиці було видно, який line search використано.
                decorated_name = f"{result_ls.method_name} ({ls_label})"

                decorated_run = OptimizationRunResult(
                    method_name=decorated_name,
                    iterations=result_ls.iterations,
                    x_star=result_ls.x_star,
                    f_star=result_ls.f_star,
                    n_iter=result_ls.n_iter,
                    func_evals=result_ls.func_evals,
                    grad_evals=result_ls.grad_evals,
                    hess_evals=result_ls.hess_evals,
                    stopped_by=result_ls.stopped_by,
                )

                summary.add_run(decorated_run)

        # ------------------ 3) Перевірка, що є хоч якісь результати ------------------
        rows = summary.as_rows()
        if not rows:
            show_error_dialog(
                self.window,
                "Немає даних для зведеної таблиці",
                "Жоден із методів не завершився коректно. "
                "Перевірте початкову точку, eps та інші параметри.",
            )
            self.window.statusBar().showMessage(
                "Зведену таблицю не побудовано — усі методи впали з помилкою"
            )
            try:
                self.window.update_run_stats(None, None, None)
            except AttributeError:
                pass
            return

        # ------------------ 4) В основному вікні покажемо останній базовий метод ------------------
        last_method_key = base_method_keys[-1]  # Hook–Jeeves
        try:
            optimizer_last = create_optimizer(
                method_key=last_method_key,
                func_key=func_key,
                eps=cfg.eps,
                line_search_key=cfg.line_search_key,
            )
        except ValueError:
            optimizer_last = None

        if optimizer_last is not None:
            self.window.clear_iterations_table()

            def iteration_callback(it: IterationResult) -> None:
                step_value = it.meta.get("alpha", it.step_norm)
                self.window.add_iteration_row(
                    k=it.index,
                    x=it.x,
                    f_val=it.f,
                    step_value=step_value,
                )

            try:
                result_last = self.engine.run(
                    optimizer=optimizer_last,
                    x0=cfg.x0,
                    max_iter=cfg.max_iter,
                    tol_step=cfg.eps,
                    tol_f=cfg.eps,
                    callback=iteration_callback,
                )
            except Exception as exc:  # noqa: BLE001
                show_error_dialog(
                    self.window,
                    "Помилка під час оптимізації (зведення)",
                    f"Під час виконання методу '{last_method_key}' виникла помилка:\n\n{exc}",
                )
                self.window.statusBar().showMessage(
                    f"Помилка під час оптимізації (зведення): {exc}"
                )
                try:
                    self.window.update_run_stats(None, None, None)
                except AttributeError:
                    pass
            else:
                self.window.update_fk_plot(result_last.iterations)
                self.window.update_contour_trajectory(tf.func, result_last.iterations)

                # Оновлюємо статистику під таблицею для "останнього" методу
                last_step = None
                if result_last.iterations:
                    last_it = result_last.iterations[-1]
                    last_step = last_it.meta.get("alpha", last_it.step_norm)

                n_outer_iters = result_last.n_iter

                try:
                    self.window.update_run_stats(
                        result_last.func_evals, last_step, n_outer_iters
                    )
                except AttributeError:
                    pass

        # ------------------ 5) Показати діалог зі зведеною таблицею ------------------
        show_summary(self.window, summary)

        # ------------------ 6) Коротке повідомлення в статус-бар про найкращий метод ------------------
        best = summary.best_by_f()
        if best is not None:
            msg = (
                f"Найкращий метод за f*: {best.method_name}, "
                f"f* = {best.f_star:.6e}, "
                f"x* = {best.x_star.tolist()}"
            )
        else:
            msg = "Не вдалося визначити найкращий метод (немає коректних результатів)."

        self.window.statusBar().showMessage(msg)


# ---------------------------------------------------------------------------
# Точка входу
# ---------------------------------------------------------------------------

def main() -> None:
    app = QApplication(sys.argv)

    # Глобальний стиль застосунку
    apply_app_style(app)

    window = MainWindow()
    engine = OptimizationEngine()

    # Контролер прив’язується до сигналу вікна і до движка
    _controller = OptimizationController(window, engine)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
