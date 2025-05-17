import numpy as np
import matplotlib.pyplot as plt
import mandelbrot_set as mb


def zoom_to_point(x_center, y_center, width, zoom_factor):

    new_width = width / zoom_factor
    xmin = x_center - new_width / 2
    xmax = x_center + new_width / 2
    

    height_ratio = (original_ymax - original_ymin) / (original_xmax - original_xmin)
    new_height = new_width * height_ratio
    ymin = y_center - new_height / 2
    ymax = y_center + new_height / 2
    
    return xmin, xmax, ymin, ymax


original_xmin, original_xmax = -2.0, 1.0
original_ymin, original_ymax = -1.5, 1.5
width, height = 800, 800
max_iter = 256


zoom_center_x = -0.745428  
zoom_center_y = 0.113009
zoom_steps = 5 
zoom_factor = 4  


current_xmin, current_xmax = original_xmin, original_xmax
current_ymin, current_ymax = original_ymin, original_ymax
current_max_iter = max_iter


for step in range(zoom_steps + 1):
    
    mandelbrot_image = mb.mandelbrot_set(
        current_xmin, current_xmax,
        current_ymin, current_ymax,
        width, height,
        current_max_iter
    )
    

    plt.figure(figsize=(10, 10))
    plt.imshow(
        mandelbrot_image.T,
        extent=[current_xmin, current_xmax, current_ymin, current_ymax],
        cmap='hot',
        origin='lower'
    )
    plt.colorbar()
    plt.title(f'Mandelbrot set (iter: {current_max_iter})\n'
              f'x: [{current_xmin:.6f}, {current_xmax:.6f}]\n'
              f'y: [{current_ymin:.6f}, {current_ymax:.6f}]')
    plt.xlabel('Re(c)')
    plt.ylabel('Im(c)')
    plt.show()

    if step < zoom_steps:
        current_xmin, current_xmax, current_ymin, current_ymax = zoom_to_point(
            zoom_center_x, zoom_center_y,
            current_xmax - current_xmin,
            zoom_factor
        )
        current_max_iter = int(current_max_iter * 1.5) 