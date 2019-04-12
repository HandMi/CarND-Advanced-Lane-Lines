
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy
import cv2

def color_space(hue, s_gradient, v_gradient):
    h = hue*numpy.ones((500,500), dtype=numpy.uint8)
    hsv_color = cv2.merge((h, s_gradient, v_gradient))
    rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)
    return rgb_color

def color_gradients(lower_b,upper_b,number_of_frames):
    s_gradient = numpy.ones((500,1), dtype=numpy.uint8)*numpy.linspace(lower_b[1], upper_b[1], 500, dtype=numpy.uint8)
    v_gradient = numpy.rot90(numpy.ones((500,1), dtype=numpy.uint8)*numpy.linspace(lower_b[2], upper_b[2], 500, dtype=numpy.uint8))
    h_array = numpy.linspace(lower_b[0], upper_b[0],number_of_frames, dtype=numpy.uint8)
    return s_gradient,v_gradient,h_array

def animate_hsv_colors(fig, ax, light, dark, number_of_frames):
    s_gradient_1,v_gradient_1,h_array_1 = color_gradients(light,dark,number_of_frames)
    def animate_colors(i):
        ax.imshow(color_space(h_array_1[i],s_gradient_1,v_gradient_1))
        ax.axis('off')
    ani = matplotlib.animation.FuncAnimation(fig, animate_colors,frames=number_of_frames)
    return ani
  