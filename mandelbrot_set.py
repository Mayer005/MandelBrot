import numpy as np


def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    c = x + y[:, None] * 1j
    z = np.zeros_like(c)
    div_time = np.zeros(c.shape, dtype=int)

    for i in range(max_iter):
        mask = (np.abs(z) <= 2)
        z[mask] = z[mask] ** 2 + c[mask]
        div_time[mask] = i

    return div_time
