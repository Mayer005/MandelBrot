import numpy as np
import matplotlib.pyplot as plt
import mandelbrot_set as mb
from matplotlib.animation import FuncAnimation


fig, ax = plt.subplots(figsize=(10, 10))
img = ax.imshow(np.zeros((800, 800)), cmap='hot', origin='lower', vmin=0, vmax=100)
plt.colorbar(img)
title = ax.set_title("Mandelbrot set - initialization")


zoom_center = (-0.745428, 0.113009)
initial_size = 3.0
zoom_factor = 0.3
max_iter_start = 100
iter_increase = 50

def init():

    mandelbrot = mb.mandelbrot_set(-2.0, 1.0, -1.5, 1.5, 800, 800, max_iter_start)
    img.set_array(mandelbrot)
    img.set_extent([-2.0, 1.0, -1.5, 1.5])
    title.set_text(f"Beginning state (iterations: {max_iter_start})")
    return [img]

def update(frame):
    current_size = initial_size * (zoom_factor ** frame)
    xmin = zoom_center[0] - current_size/2
    xmax = zoom_center[0] + current_size/2
    ymin = zoom_center[1] - current_size/2
    ymax = zoom_center[1] + current_size/2
    current_iter = max_iter_start + frame * iter_increase
    
    mandelbrot = mb.mandelbrot_set(xmin, xmax, ymin, ymax, 800, 800, current_iter)
    

    img.set_array(mandelbrot)
    img.set_extent([xmin, xmax, ymin, ymax])
    img.set_clim(vmin=0, vmax=current_iter*0.8)  
    title.set_text(f"Zoom: {(1/current_size):.1f}x, Iterations: {current_iter}")
    
    print(f"Frame {frame} rendered - Zoom: {current_size:.5f}")
    return [img]


ani = FuncAnimation(fig, update, frames=300, init_func=init,
                   interval=1000, blit=True, repeat=False)

plt.tight_layout()
plt.show()