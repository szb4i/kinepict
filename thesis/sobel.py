import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.signal import convolve2d
import sys
 
# setting path
sys.path.append('../')
from utils.kernels.sobel_kernels import get_sobel_x_kernel, get_sobel_y_kernel


img = plt.imread('../data/tree.jpg')[:,:,0]
kernel_x = get_sobel_x_kernel()
kernel_y = get_sobel_y_kernel()
gx = convolve2d(img, kernel_x, mode='same', boundary = 'symm', fillvalue=0)
gy = convolve2d(img, kernel_y, mode='same', boundary = 'symm', fillvalue=0)
magnitude = np.hypot(gx, gy)
plt.figure(figsize=(12,7))

plt.subplot(1,4,1)
plt.imshow(img, cmap='gray')
plt.axis("off")
plt.gca().set_title('original', fontdict={'fontsize':18})
plt.subplot(1,4,2)
plt.imshow(gx, cmap='gray')
plt.axis("off")
plt.gca().set_title('$g_x$', fontdict={'fontsize':18})
plt.subplot(1,4,3)
plt.imshow(gy, cmap='gray')
plt.axis("off")
plt.gca().set_title('$g_y$', fontdict={'fontsize':18})
plt.subplot(1,4,4)
plt.imshow(magnitude, cmap='gray')
plt.axis("off")
plt.gca().set_title('$M(x,y)$', fontdict={'fontsize':18})
plt.tight_layout()
plt.show()