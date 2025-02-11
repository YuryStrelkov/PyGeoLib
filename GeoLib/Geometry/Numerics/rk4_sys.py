from typing import Callable, Tuple, Iterable
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np
import math


def predator_prey() -> Tuple[Callable[[float, np.ndarray], float], ...]:
    def equation_1(_t: float, _xs: np.ndarray, _alpha: float, _beta: float, _delta: float, _gamma: float) -> float:
        return _alpha * _xs[0] - _beta  * _xs[0] * _xs[1]

    def equation_2(_t: float, _xs: np.ndarray, _alpha: float, _beta: float, _delta: float, _gamma: float) -> float:
        return _delta * _xs[0] * _xs[1] - _gamma *  _xs[1]

    return equation_1, equation_2,


def pendulum() -> Tuple[Callable[[float, np.ndarray], float], ...]:
    # https://simonebertonilab.com/the-pendulum-model-differential-equations-into-simulations/
    def equation_1(_t: float, _xs: np.ndarray, torque: float, _length: float, _mass: float, _tension: float) -> float:
        return _xs[1]

    def equation_2(_t: float, _xs: np.ndarray, torque: float, _length: float, _mass: float, _tension: float) -> float:
        m_l2 = 1 / (_mass * _length * _length)
        return torque * m_l2 - _tension * m_l2 * _xs[1] - 9.81 / _length * math.sin(_xs[0])

    return equation_1, equation_2,


class RungeKuttaModes(Enum):
    ORIGINAL_RK4 = 0,
    RK_CASE_1 = 1,
    RK_CASE_2 = 2,
    RK_CASE_3 = 3,
    RK_CASE_4 = 4,
    RK_CASE_5 = 5,


class RungeKutta4:
    _RUNGE_KUTTA_CASES = {
        RungeKuttaModes.ORIGINAL_RK4: (lambda x, k_1: x + 0.5 * k_1,
                                       lambda x, k_1, k_2: x + 0.5 * k_2,
                                       lambda x, k1, k_2, k_3: x + k_3,
                                       lambda k_1, k_2, k_3, k_4: (k_1 + 2.0 * k_2 + 2.0 * k_3 + k_4) / 6.0),
        RungeKuttaModes.RK_CASE_1: (lambda x, k_1: x + 0.5 * k_1,
                                    lambda x, k_1, k_2: x + 0.5 * k_2,
                                    lambda x, k1, k_2, k_3: x + k_3,
                                    lambda k_1, k_2, k_3, k_4: (k_1 + 2.0 * k_2 + 2.0 * k_3 + k_4) / 6.0),
        RungeKuttaModes.RK_CASE_2: (lambda x, k_1: x + 0.5 * k_1,
                                    lambda x, k_1, k_2: x + 0.25 * (k_1 + k_2),
                                    lambda x, k1, k_2, k_3: x - k_2 - 2 * k_3,
                                    lambda k_1, k_2, k_3, k_4: (k_1 + 3.0 * k_2 + k_3 + k_4) / 6.0),
        RungeKuttaModes.RK_CASE_3: (lambda x, k_1: x + 0.5 * k_1,
                                    lambda x, k_1, k_2: x - k_1 * 0.5 + k_2,
                                    lambda x, k1, k_2, k_3: x + (k_2 + k_3) * 0.5,
                                    lambda k_1, k_2, k_3, k_4: (k_1 + 3.0 * k_2 + k_3 + k_4) / 6.0),
        RungeKuttaModes.RK_CASE_4: (lambda x, k_1: x + 0.5 * k_1,
                                    lambda x, k_1, k_2: x + (k_1 + k_2) * 0.25,
                                    lambda x, k1, k_2, k_3: x - k_2 - 2 * k_3,
                                    lambda k_1, k_2, k_3, k_4: (k_1 + 3.0 * k_2 + k_3 + k_4) / 6.0),
        RungeKuttaModes.RK_CASE_5: (lambda x, k_1: x + 0.5 * k_1,
                                    lambda x, k_1, k_2: x + k_1 * 0.3 + k_2 * 0.2,
                                    lambda x, k1, k_2, k_3: x - k_2 * 1.5 + 2.5 * k_3,
                                    lambda k_1, k_2, k_3, k_4: (k_1 - k_2 + 5.0 * k_3 + k_4) / 6.0),
    }
    __slots__ = ("_t_start", "_t_end", "_x_start", "_ode_system", "_t_step", "_rk_mode")

    def __init__(self, *,
                 t_start: float = 0,
                 t_end: float = 10.0,
                 t_step: float = 0.001,
                 x_start: np.ndarray = None,
                 ode_system: Iterable[Callable[[float, np.ndarray, Tuple[float, ...]], float]] = None):
        self._t_start, self._t_end = (t_end, t_start) if t_end < t_start else (t_start, t_end)
        self._ode_system = list(ode_system) if ode_system else None
        self._x_start = x_start if x_start else np.zeros((len(self._ode_system),), dtype=float)\
            if self._ode_system else None
        self._t_step = t_step
        self._rk_mode = RungeKuttaModes.ORIGINAL_RK4

    def _eval_ode_sys(self, t_val: float, x_vector: np.ndarray, args: Tuple[float, ...]) -> np.ndarray:
        result = np.array(tuple(func(t_val, x_vector, *args) for func in self._ode_system))
        return np.squeeze(result) if any(d == 1 for d in result.shape) else result

    @property
    def solution_mode(self) -> RungeKuttaModes:
        """
        Определяет способ решения системы ОДУ (Оригинальный Рунге-Кутта 4-ого порядка или дна из модификаций из статьи).
        :return:
        """
        return self._rk_mode

    @solution_mode.setter
    def solution_mode(self, value: RungeKuttaModes) -> None:
        if isinstance(value, RungeKuttaModes):
            self._rk_mode = value

    @property
    def t_step(self) -> float:
        """
        Фиксированный шаг решения по времени.
        :return:
        """
        return self._t_step

    @property
    def x_start(self) -> np.ndarray:
        """
        Начальные условия для системы ОДУ.
        :return:
        """
        return self._x_start

    @property
    def t_start(self) -> float:
        """
        Начальное значение времени.
        :return:
        """
        return self._t_start

    @property
    def t_end(self) -> float:
        """
        Конечное значение времени.
        :return:
        """
        return self._t_end

    @property
    def time_steps_n(self) -> int:
        """
        Количество измерений функции.
        :return:
        """
        return int((self.t_end - self.t_start) / self.t_step)

    @x_start.setter
    def x_start(self, value: np.ndarray) -> None:
        if isinstance(value, np.ndarray):
            self._x_start = value

    @t_start.setter
    def t_start(self, value: float) -> None:
        if isinstance(value, float):
            self._t_start = min(self.t_end - self.t_step * 10, value)

    @t_end.setter
    def t_end(self, value: float) -> None:
        if isinstance(value, float):
            self._t_end = max(self.t_start + self.t_step * 10, value)

    @t_step.setter
    def t_step(self, value: float) -> None:
        if isinstance(value, float):
            self._t_step = min(abs(value), (self.t_end - self.t_start) * 0.1)

    def clear_system(self):
        """
        Полностью удаляет все уравнения из системы ОДУ.
        :return:
        """
        if self._ode_system:
            self._ode_system.clear()

    def append_equation(self, equation) -> None:
        """
        Добавляет новое уравнение в систему ОДУ. НЕ ПРОВЕРЯЕТ, ЧТО ПЕРЕДАВАЕМОЕ УРАВНЕНИЕ ЯВЛЯЕТСЯ ЛЯМБДОЙ ИЛИ ФУНКЦИЕЙ!
        :param equation:
        :return:
        """
        if self._ode_system is None:
            self._ode_system = []
        self._ode_system.append(equation)
        self._x_start = np.array((0.0, )) if self._x_start is None else np.hstack((self._x_start, 0.0))

    def append_equations(self, equations: Iterable[Callable[[float, np.ndarray, Tuple[float, ...]], float]],
                         clear_sys: bool = True) -> None:
        """
        Добавляет новые уравнения в систему ОДУ. НЕ ПРОВЕРЯЕТ, ЧТО ПЕРЕДАВАЕМЫУ УРАВНЕНИЯ ЯВЛЯЕТСЯ ЛЯМБДОЙ ИЛИ ФУНКЦИЕЙ!
        :param equations: список уравнений для системы ОДУ. (Должны быть ЛЯМБДАМИ ИЛИ ФУНКЦИЯМИ)
        :param clear_sys: параметр, который указывает необходимость удаления старых уравнений системы ОДУ.
        :return:
        """
        if clear_sys and self._ode_system:
            self.clear_system()
        for equation in equations:
            self.append_equation(equation)

    def __call__(self, args: Tuple[float, ...] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        rk_instance = RungeKutta4 (...)
        ...
        result = rk_instance (args)
        :param args:
        :return:
        """
        if len(self._ode_system) == 0:
            return np.array((self.t_start,)), self.x_start

        t_values = np.zeros((self.time_steps_n, ), dtype=float)
        x_values = np.zeros((self.time_steps_n, *self.x_start.shape))
        t_values[0], x_values[0] = self.t_start, self.x_start
        args = args if args else ()
        _k2, _k3, _k4, _k_total = RungeKutta4._RUNGE_KUTTA_CASES[self.solution_mode]
        for index in range(1, self.time_steps_n):
            t_curr = t_values[index - 1]
            x_curr = x_values[index - 1, :]
            k_1 = self._eval_ode_sys(t_curr, x_curr, args) * self.t_step
            k_2 = self._eval_ode_sys(t_curr + self.t_step * 0.5, _k2(x_curr, k_1), args) * self.t_step
            k_3 = self._eval_ode_sys(t_curr + self.t_step * 0.5, _k3(x_curr, k_1, k_2), args) * self.t_step
            k_4 = self._eval_ode_sys(t_curr + self.t_step, _k4(x_curr, k_1, k_2, k_3), args) * self.t_step
            t_values[index] = t_curr + self.t_step
            x_values[index, :] = x_curr + _k_total(k_1, k_2, k_3, k_4)
        return t_values, x_values


def matrix_a(size: int = 7) -> np.ndarray:
    mat = np.zeros((size, size,), dtype=float)
    for index in range(size):
        mat[index, index] = -2
        if index - 1 > -1:
            mat[index, index - 1] = 1
        if index + 1 < size:
            mat[index, index + 1] = 1
    return mat


def matrix_c(t: float, size: int = 7) -> np.ndarray:
    return np.eye(size, dtype=float)


def matrix_r(t: float, size: int = 7) -> np.ndarray:
    return matrix_c(t, size)


def matrix_q(t: float, size: int = 7) -> np.ndarray:
    return matrix_c(t, size)


def kalman_bucy_ode_1(t: float, p: np.ndarray, a: np.ndarray) -> np.ndarray:
    w, _ = a.shape
    c = matrix_c(t, w)
    q = matrix_q(t, w)
    r = np.linalg.inv(matrix_r(t, w))
    return a @ p + p @ a.T - p.T @ c.T @ r @ c @ p + q


def predator_prey_example():
    ode_rk = RungeKutta4()
    ode_rk.t_end = 10.0
    ode_rk.append_equations(predator_prey())
    ode_rk.x_start = np.array((5.0, 10.0))
    alpha_beta_delta_gamma = (1.0, 0.1, 0.075, 1.5)
    time_series, result = ode_rk(alpha_beta_delta_gamma)
    plt.plot(time_series, result[:, 0], 'g')
    plt.plot(time_series, result[:, 1], 'r')
    plt.legend(('prey', 'predator'))
    plt.xlabel('time units')
    plt.ylabel('population units')
    plt.show()


def pendulum_example():
    ode_rk = RungeKutta4()
    ode_rk.t_end = 10.0
    mass = 0.5  # [kg]
    length = 1.0  # [m]
    tension = 0.5  # [N * m * s]
    torque = 3.0
    ode_rk.append_equations(pendulum())
    time_series, result = ode_rk((torque, length, mass, tension))
    plt.plot(time_series, result[:, 0], 'g')
    plt.plot(time_series, result[:, 1], 'r')
    plt.legend((r'$\theta\left(t\right),\left[rad\right]$',
                r'$\frac{\partial\theta\left(t\right)}{\partial t},\left[\frac{rad}{sec}\right]$'))
    plt.xlabel(r'$time, \left[sec\right]$')
    plt.show()


def kalman_filter():
    rk = RungeKutta4()
    rk.t_step = 0.025
    rk.t_end = 1.0
    a = matrix_a()
    rk.append_equations((lambda t, x: kalman_bucy_ode_1(t, x, a),))
    rk.x_start = np.ones((7, 7), dtype=float)
    t, result = rk()
    print(f"result: {result.shape}")
    layer, rows, cols = result.shape
    for row in range(rows):
        for col in range(cols):
            plt.plot(t, result[:, row, col])
    plt.xlabel(r'$time, \left[sec\right]$')
    plt.show()


if __name__ == "__main__":
    kalman_filter()
    exit()
    predator_prey_example()
    pendulum_example()


    print(matrix_a())
    from math import exp
    ode_rk = RungeKutta4()
    ode_rk.t_step = 0.0125
    ode_rk.t_end = 2.0
    # ARTICLE EXAMPLE - 1
    # ode_rk.append_equations((lambda t, x: 1.0 + 2 * x - x * x, ))
    # ode_rk.x_start = np.array((0.0,))
    # ARTICLE EXAMPLE - 2
    ode_rk.solution_mode = RungeKuttaModes.RK_CASE_5
    ode_rk.append_equations((lambda t, x: exp(t) - exp(3 * t) + 2 * exp(2 * t) * x - exp(t) * x * x,))
    ode_rk.x_start = np.array((1.0,))
    time_series, result = ode_rk()
    plt.plot(time_series, result[:, 0], 'g')
    plt.plot(time_series, result[:, 0], '.b')
    plt.xlabel(r'$time, \left[sec\right]$')
    plt.show()

    # predator_prey_example()
    # pendulum_example()
