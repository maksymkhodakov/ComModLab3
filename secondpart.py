import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def analytical_solution(t):
    return 1 / (0.25 * t ** 2 + 0.875)


def f(t, y):
    return -0.5 * t * y ** 2


# Метод Ейлера
def euler_step(t, y, h):
    return y + h * f(t, y)


def milne_simpson(t0, y0, h, t_end, tol):
    t_values = [t0]
    y_values = [y0]
    max_attempts = 10
    attempts = 0

    # Отримуємо 3 початкові точки за допомогою методу Ейлера
    for i in range(3):
        y_new = euler_step(t_values[-1], y_values[-1], h)
        t_values.append(t_values[-1] + h)
        y_values.append(y_new)

    while t_values[-1] < t_end:
        t_n = t_values[-1]
        y_n = y_values[-1]

        y_pred = y_values[-4] + h / 3 * (
                2 * f(t_values[-3], y_values[-3]) - f(t_values[-2], y_values[-2]) + 2 * f(t_n, y_n))

        # Коригуюча формула:
        y_corr = y_values[-2] + h / 3 * (
                f(t_values[-2], y_values[-2]) + 4 * f(t_values[-1], y_pred) + f(t_n + h, y_pred))

        error = abs(y_corr - y_pred)

        if error < tol:
            t_values.append(t_values[-1] + h)
            y_values.append(y_corr)
            attempts = 0  # reset attempts counter
        else:
            h = h / 2
            attempts += 1

            if attempts >= max_attempts:
                raise ValueError("Failed to converge after {} attempts.".format(max_attempts))
            if h < 1e-6:  # too small step
                raise ValueError("Step size became too small.")
            continue

    return t_values, y_values


t0 = 0.5
y0 = 1
t_end = 3.5
h = 1
tol = 1e-4

t_values, y_values = milne_simpson(t0, y0, h, t_end, tol)
analytical_y_values = [analytical_solution(t) for t in t_values]

plt.plot(t_values, y_values, 'o-', label="Approximate solution")
plt.plot(t_values, analytical_y_values, 'r--', label="Analytical solution")
plt.xlabel('t')
plt.ylabel('y')
plt.title('Milne-Simpson method for solving ODE')
plt.legend()
plt.grid(True)
plt.show()
