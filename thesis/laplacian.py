import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.signal import convolve2d
import sys
 
# setting path
sys.path.append('../')

from utils.kernels.laplacian_kernel import get_laplacian_kernel
from utils.img.scaler import scale

img = scale(plt.imread('../data/tree.jpg')[:,:,0])
kernel = get_laplacian_kernel()
img_laplacian = convolve2d(img, kernel, mode='same', boundary = 'symm', fillvalue=0)
plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.axis("off")
plt.gca().set_title('original', fontdict={'fontsize':18})
plt.subplot(1,3,2)
plt.imshow(img_laplacian, cmap='gray')
plt.axis("off")
plt.gca().set_title('laplacian', fontdict={'fontsize':18})
plt.subplot(1,3,3)
plt.imshow(img + 0.5*scale(img_laplacian), cmap='gray')
plt.axis("off")
plt.gca().set_title('sum', fontdict={'fontsize':18})
plt.tight_layout()
plt.show()