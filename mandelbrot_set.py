import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
import math
from time import time



@cuda.jit(device=True)
def mandelbrot_optimized(c_real, c_imag, max_iter):

    q = (c_real - 0.25)**2 + c_imag**2
    if q * (q + (c_real - 0.25)) < 0.25 * c_imag**2:
        return max_iter
    

    if (c_real + 1.0)**2 + c_imag**2 < 0.0625:
        return max_iter
    

    z_real = np.float64(0.0)
    z_imag = np.float64(0.0)
    

    for i in range(max_iter):

        z_real_sq = z_real * z_real
        z_imag_sq = z_imag * z_imag
        

        if z_real_sq + z_imag_sq > 4.0:
            log_zn = math.log(z_real_sq + z_imag_sq) / 2
            nu = math.log(log_zn / math.log(2)) / math.log(2)
            return i + 1 - nu
        

        z_imag = 2.0 * z_real * z_imag + c_imag
        z_real = z_real_sq - z_imag_sq + c_real
        
        if i > 50 and i % 25 == 0 and z_real_sq + z_imag_sq < 1e-10:
                return max_iter
    
    return max_iter

@cuda.jit
def compute_mandelbrot_optimized(min_x, max_x, min_y, max_y, image, max_iter):
    width = image.shape[1]
    height = image.shape[0]
    
    i, j = cuda.grid(2)
    
    if i < width and j < height:
        x = min_x + i * (max_x - min_x) / (width - 1)
        y = min_y + j * (max_y - min_y) / (height - 1)
        
        image[j, i] = mandelbrot_optimized(x, y, max_iter)

@cuda.jit
def apply_color_mapping(raw_image, color_image, max_iter, color_mode):
    i, j = cuda.grid(2)
    width = raw_image.shape[1]
    height = raw_image.shape[0]
    
    if i < width and j < height:
        iter_value = raw_image[j, i]
        
        if iter_value >= max_iter:
            color_image[j, i, 0] = 0
            color_image[j, i, 1] = 0
            color_image[j, i, 2] = 0
        else:
            if color_mode == 0:  
                hue = 0.7 + 0.3 * iter_value / max_iter
                saturation = 0.8
                value = 1.0 if iter_value < max_iter else 0.0
                
                h = hue * 6.0
                sector = int(h) % 6
                f = h - int(h)
                
                p = value * (1.0 - saturation)
                q = value * (1.0 - saturation * f)
                t = value * (1.0 - saturation * (1.0 - f))
                
                if sector == 0:
                    color_image[j, i, 0] = int(value * 255)
                    color_image[j, i, 1] = int(t * 255)
                    color_image[j, i, 2] = int(p * 255)
                elif sector == 1:
                    color_image[j, i, 0] = int(q * 255)
                    color_image[j, i, 1] = int(value * 255)
                    color_image[j, i, 2] = int(p * 255)
                elif sector == 2:
                    color_image[j, i, 0] = int(p * 255)
                    color_image[j, i, 1] = int(value * 255)
                    color_image[j, i, 2] = int(t * 255)
                elif sector == 3:
                    color_image[j, i, 0] = int(p * 255)
                    color_image[j, i, 1] = int(q * 255)
                    color_image[j, i, 2] = int(value * 255)
                elif sector == 4:
                    color_image[j, i, 0] = int(t * 255)
                    color_image[j, i, 1] = int(p * 255)
                    color_image[j, i, 2] = int(value * 255)
                else:
                    color_image[j, i, 0] = int(value * 255)
                    color_image[j, i, 1] = int(p * 255)
                    color_image[j, i, 2] = int(q * 255)
            
            elif color_mode == 1:
                t = iter_value / max_iter
                r = int(255 * (9 * (1-t) * t * t * t * t))
                g = int(255 * (15 * (1-t) * (1-t) * t * t * t))
                b = int(255 * (8.5 * (1-t) * (1-t) * (1-t) * t * t))
                
                color_image[j, i, 0] = r
                color_image[j, i, 1] = g
                color_image[j, i, 2] = b
            
            else: 
                t = iter_value / max_iter
                smooth_t = math.log(1 + t * 10) / math.log(11) 
                r = int(255 * (smooth_t**0.7))
                g = int(255 * (smooth_t**0.5))
                b = int(255 * (smooth_t**0.3))
                
                color_image[j, i, 0] = r
                color_image[j, i, 1] = g
                color_image[j, i, 2] = b

class MandelbrotRenderer:
    def __init__(self, width=1024, height=1024, max_iter=1000):
        self.width = width
        self.height = height
        self.max_iter = max_iter
        
        self.raw_image = np.zeros((height, width), dtype=np.float32)
        
        self.color_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        self.gpu_config()
        

        self.x_center, self.y_center = -0.5, 0.0 
        self.size = 3.0  
        self.aspect_ratio = width / height
        
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.img = None 
        self.paused = False 
        self.zoom_factor = 1.5  
        self.color_mode = 0  
        
        self.last_render_time = 0
        self.render_count = 0
        self.total_render_time = 0
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        self.history = []
        self.history_index = -1
        self.max_history = 20 
        
        self.show_controls()
    
    def gpu_config(self):
        device = cuda.get_current_device()
        
        max_threads_per_block = device.MAX_THREADS_PER_BLOCK
        max_x = device.MAX_BLOCK_DIM_X
        max_y = device.MAX_BLOCK_DIM_Y
        
        block_size_x = min(32, max_x)
        block_size_y = min(32, max_y)
        
        while block_size_x * block_size_y > max_threads_per_block:
            block_size_x = max(8, block_size_x - 8)
            block_size_y = max(8, block_size_y - 8)
        
        self.threads_per_block = (block_size_x, block_size_y)
        self.blocks_per_grid = (
            (self.width + self.threads_per_block[0] - 1) // self.threads_per_block[0],
            (self.height + self.threads_per_block[1] - 1) // self.threads_per_block[1]
        )
        
        print(f"GPU config: {self.threads_per_block} threads/block, {self.blocks_per_grid} blocks/grid")
    
    def calculate_bounds(self):
        x_half = self.size / 2
        y_half = x_half / self.aspect_ratio
        return (
            self.x_center - x_half,
            self.x_center + x_half,
            self.y_center - y_half,
            self.y_center + y_half
        )
    
    def auto_adjust_iterations(self):
        zoom_level = 1.0 / self.size
        base_iterations = 100
        
        # Logarithmic scaling - more zoom = more iterations
        self.max_iter = min(
            50000, 
            max(
                100,  
                int(base_iterations * math.log2(zoom_level + 1) + 500)
            )
        )
    
    def render(self):
        self.auto_adjust_iterations()
        min_x, max_x, min_y, max_y = self.calculate_bounds()
        
        d_raw_image = cuda.to_device(self.raw_image)

        compute_mandelbrot_optimized[self.blocks_per_grid, self.threads_per_block](
            min_x, max_x, min_y, max_y, d_raw_image, self.max_iter
        )
        
        d_color_image = cuda.to_device(self.color_image)
        
        apply_color_mapping[self.blocks_per_grid, self.threads_per_block](
            d_raw_image, d_color_image, self.max_iter, self.color_mode
        )
        
        d_raw_image.copy_to_host(self.raw_image)
        d_color_image.copy_to_host(self.color_image)
        
        return self.color_image
    
    def update_display(self):
        start_time = time()
        image = self.render()
        render_time = time() - start_time
        
        self.last_render_time = render_time
        self.render_count += 1
        self.total_render_time += render_time
        avg_render_time = self.total_render_time / self.render_count
        
        if self.img is None:
            self.img = self.ax.imshow(image, origin='lower')
        else:
            self.img.set_data(image)
        
        self.ax.set_title(
            f'Mandelbrot | Zoom: {1/self.size:.1e}x | '
            f'Iter: {self.max_iter} | Render: {render_time:.3f}s | '
            f'Avg: {avg_render_time:.3f}s | '
            f'Position: ({self.x_center:.10f}, {self.y_center:.10f})'
        )
        
        self.fig.canvas.draw_idle()
    
    def add_to_history(self):

        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        
        self.history.append((self.x_center, self.y_center, self.size))
        
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        self.history_index = len(self.history) - 1
    
    def on_key(self, event):
        move_step = 0.1 * self.size
        
        if event.key == 'right':
            self.x_center += move_step
            self.add_to_history()
        elif event.key == 'left':
            self.x_center -= move_step
            self.add_to_history()
        elif event.key == 'up':
            self.y_center += move_step
            self.add_to_history()
        elif event.key == 'down':
            self.y_center -= move_step
            self.add_to_history()
        elif event.key == '+' or event.key == '=':
            self.size /= self.zoom_factor
            self.add_to_history()
        elif event.key == '-':
            self.size *= self.zoom_factor
            self.add_to_history()
        elif event.key == ' ':
            self.paused = not self.paused
        elif event.key == 'escape':
            plt.close()
        elif event.key == 'c':
            self.color_mode = (self.color_mode + 1) % 3
        elif event.key == 'r':
            self.x_center, self.y_center = -0.5, 0.0
            self.size = 3.0
            self.add_to_history()
        elif event.key == '[' or event.key == 'backspace':
            if self.history_index > 0:
                self.history_index -= 1
                self.x_center, self.y_center, self.size = self.history[self.history_index]
        elif event.key == ']':
            if self.history_index < len(self.history) - 1:
                self.history_index += 1
                self.x_center, self.y_center, self.size = self.history[self.history_index]
        elif event.key == 's':
            filename = f"mandelbrot_{self.x_center:.6f}_{self.y_center:.6f}_zoom{1/self.size:.1e}.png"
            plt.imsave(filename, self.color_image, origin='lower')
            print(f"Image saved: {filename}")
        elif event.key == 'h':
            self.show_controls()
        
        if not self.paused:
            self.update_display()
    
    def on_click(self, event):

        if event.inaxes == self.ax:

            width, height = self.width, self.height
            

            min_x, max_x, min_y, max_y = self.calculate_bounds()
            x = min_x + event.xdata * (max_x - min_x) / width
            y = min_y + event.ydata * (max_y - min_y) / height
            

            self.x_center, self.y_center = x, y
            

            if event.button == 3:  
                self.size *= self.zoom_factor
            else:  
                self.size /= self.zoom_factor
            
            self.add_to_history()
            
            if not self.paused:
                self.update_display()
    
    def on_scroll(self, event):
        if event.inaxes == self.ax:
            if event.button == 'up':
                self.size /= self.zoom_factor
            elif event.button == 'down':
                self.size *= self.zoom_factor
            
            self.add_to_history()
            
            if not self.paused:
                self.update_display()
    
    def show_controls(self):
        commands = """
            'Arrow keys': 'Movement',
            '+ / -': 'Zoom in/out',
            'Space': 'Pause',
            'Escape': 'Quit',
            'C': 'Cycle color modes',
            'R': 'Reset view',
            '[ or backspace': 'Navigate back in history',
            ']': 'Navigate forward in history',
            'S': 'Save image',
            'H': 'Show this help'\n
                    """
        
        print(f"Controls: {commands}")

    
    def show(self):

        self.add_to_history()
        
        self.update_display()
        
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        
        plt.tight_layout()
        plt.show()