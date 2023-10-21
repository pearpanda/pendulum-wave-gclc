import numpy as np
from PendulumSolver import PendulumSolver
from collections import namedtuple
from math import pi

if __name__ == '__main__':
    k = 6
    solver = PendulumSolver(degree=k)

    n = 7
    lengths = [0.05977, 0.06192, 0.06419, 0.06659, 0.06913, 0.07181, 0.07466]

    t0 = pi / 3
    periods = [solver.period(t0, l) / 2 for l in lengths]
    polynomials = [solver.solve(l, t0) for l in lengths]

    with open('equations/data/generate.txt', 'w') as file:
        file.write(f'number N {n}\n')

        for i in range(n):
            file.write(f'number duzine[{i}] {lengths[i]}\n')

        for i in range(n):
            file.write(f'number periodi[{i}] {periods[i]}\n')

        for i in range(n):
            for j in range(k + 1):
                file.write(f'number polinomi[{i}][{j}] {polynomials[i][j]}\n')
