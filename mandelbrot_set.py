import numpy as np
import matplotlib.pyplot as plt
from numba import jit


@jit(nopython=True)
def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    res = np.zeros((height, width), dtype=np.int32)


    for i in range(height):
        for j in range(width):
            c = complex(x[j], y[i])
            z = complex(0, 0)
            iteration = 0

            while abs(z) <= 2 and iteration < max_iter:
                z = z * z + c
                iteration += 1

            res[i, j] = iteration
    return res

   

