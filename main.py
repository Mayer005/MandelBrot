import mandelbrot_set as mb
from matplotlib.animation import FuncAnimation




if __name__ == "__main__":

    renderer = mb.MandelbrotRenderer(width=1000, height=1000)
    renderer.show()