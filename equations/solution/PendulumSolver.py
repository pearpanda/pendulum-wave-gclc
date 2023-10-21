import sympy as sym
import numpy as np
from sympy import Symbol, Function, diff
from sympy.abc import t, theta, omega, alpha
from scipy.integrate import quad, solve_ivp
from numpy.polynomial.polynomial import polyfit
from math import pi


class PendulumSolver:
    def __init__(self, degree: int = 6) -> None:
        self.degree = degree

        self.t = t
        self.g = Symbol('g', positive=True)
        self.l = Symbol('l', positive=True)
        self.theta = theta
        self.omega = omega
        self.alpha = alpha

        self.angle = Function('angle')
        self.velocity = diff(self.angle(self.t), self.t, 1)
        self.acceleration = diff(self.angle(self.t), self.t, 2)

        self.equation = self._solve_generalized()

    def solve(self, l: float, t0: float, g: float = 9.81):
        accel = self._ode(l, g)
        f = lambda t, values: [values[1], accel(values[0], values[1])]

        T = self.period(t0, l, g)
        t = np.linspace(0, T / 2, 200)
        self.ode = solve_ivp(f, t_span=[t[0], t[-1]], y0=[t0, 0], t_eval=t)

        return polyfit(self.ode.t, self.ode.y[0], deg=self.degree)

    def period(self, t0: float, l: float, g: float = 9.81) -> float:
        I, _ = quad(lambda x: 1 / np.sqrt(np.cos(x) - np.cos(t0)), 0, t0)
        return 4.0 * np.sqrt(l / (2 * g)) * I

    def _ode(self, l: float, g: float = 9.81):
        expression = sym.simplify(self.equation.subs({ self.l: l, self.g: g }))
        return sym.lambdify([self.theta, self.omega], expression, modules=['scipy', 'numpy'])

    def _solve_generalized(self):
        equation = self.acceleration + (self.g / self.l) * sym.sin(self.angle(self.t))
        equation = equation.subs({
            self.angle(self.t): self.theta,
            self.velocity: self.omega,
            self.acceleration: self.alpha
        })
        return sym.solve(equation, self.alpha)[0]
