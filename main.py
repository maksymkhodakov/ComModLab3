import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def analytical_solution(t):
    return 1 / (0.25 * t**2 + 0.875)


def f(t, y):
    return -0.5 * t * y ** 2


def heun_step(t, y, h):
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.75 * h, y + 0.75 * h * k2)
    return y + (2 * k1 + 3 * k2 + 4 * k3) * h / 9


def heun_method_adaptive(t0, y0, h, t_end, tol):
    t_values = [t0]
    y_values = [y0]

    while t_values[-1] < t_end:
        t_n = t_values[-1]
        y_n = y_values[-1]

        # Один крок з кроком h
        y_temp1 = heun_step(t_n, y_n, h)
        # Два кроки з кроком h/2
        y_temp2 = heun_step(t_n, y_n, h / 2)
        y_temp2 = heun_step(t_n + h / 2, y_temp2, h / 2)

        error = abs(y_temp1 - y_temp2)

        # Якщо помилка менше заданої точності, приймаємо крок
        if error < tol:
            t_values.append(t_n + h)
            y_values.append(y_temp1)
        # Інакше зменшуємо крок
        else:
            h = h / 2
            continue

        # Наступний крок
        h = h * (tol / error) ** 0.5

    return t_values, y_values


t0 = 0.5
y0 = 1
t_end = 3.5
h = 0.1
tol = 1e-4

t_values, y_values = heun_method_adaptive(t0, y0, h, t_end, tol)

analytical_y_values = [analytical_solution(t) for t in t_values]


plt.plot(t_values, y_values, 'o-', label="Approximate solution (Heun)")
plt.plot(t_values, analytical_y_values, 'r--', label="Analytical solution")
plt.xlabel('t')
plt.ylabel('y')
plt.title('Adaptive Heun method for solving ODE')
plt.legend()
plt.grid(True)
plt.show()
