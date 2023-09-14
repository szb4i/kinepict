import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.signal import convolve2d
import sys
 
# setting path
sys.path.append('../')
from utils.kernels.gaussian_kernel import get_gaussian_kernel


img = plt.imread('../data/tree.jpg')[:,:,0]
kernel = get_gaussian_kernel(sig=5)
img_filtered_1 = convolve2d(img, kernel, mode='same')
kernel = get_gaussian_kernel(sig=5, l=7)
img_filtered_2 = convolve2d(img, kernel, mode='same')
plt.figure(figsize=(12,7))
plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.axis("off")
plt.gca().set_title('original', fontdict={'fontsize':18})
plt.subplot(1,3,2)
plt.imshow(img_filtered_1, cmap='gray')
plt.axis("off")
plt.gca().set_title('3x3, $\sigma=5$', fontdict={'fontsize':18})
plt.subplot(1,3,3)
plt.imshow(img_filtered_2, cmap='gray')
plt.axis("off")
plt.gca().set_title('7x7, $\sigma=5$', fontdict={'fontsize':18})
plt.tight_layout()
plt.show()
