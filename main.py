import numpy as np
import matplotlib.pyplot as plt
import mandelbrot_set as mb


# Paraméterek
xmin, xmax = -2.0, 1.0
ymin, ymax = -1.5, 1.5
width, height = 1000, 1000
max_iter = 256

# Mandelbrot halmaz kiszámítása
mandelbrot_image = mb.mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)

# Ábrázolás
plt.figure(figsize=(10, 10))
plt.imshow(mandelbrot_image.T, extent=[xmin, xmax, ymin, ymax], cmap='hot')
plt.colorbar()
plt.title('Mandelbrot halmaz')
plt.show()