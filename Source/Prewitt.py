import numpy as np
import cv2
from matplotlib import pyplot as plt
from convolve_np import convolve_np

img = cv2.imread('images/jet.jpg', cv2.IMREAD_GRAYSCALE)

height = img.shape[0]
width = img.shape[1]

Hx = 1/3*np.array([[1, 0, -1],
               [1, 0, -1],
               [1, 0, -1]])

Hy = 1/3*np.array([[-1, -1, -1],
               [0, 0, 0],
               [1, 1, 1]])

img_x = convolve_np(img, Hx)
img_y = convolve_np(img, Hy)

img_out = np.sqrt(np.power(img_x, 2) + np.power(img_y, 2))
img_out = (img_out / np.max(img_out)) * 255

cv2.imwrite('images/edge_prewitt.jpg', img_out)

plt.imshow(img_out, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()