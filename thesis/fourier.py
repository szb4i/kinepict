import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy import ndimage, misc
import sys

# setting path
sys.path.append('../')
from utils.patterns.rectangle import get_rectangle
from utils.kernels.sobel_kernels import get_sobel_x_kernel, get_sobel_y_kernel
from utils.kernels.zeropad_kernel import get_zeropadded_kernel
from utils.kernels.gaussian_kernel import get_gaussian_kernel
from utils.file.dva_reader import read_dva
from utils.file.dva_reader import read_dva
from utils.img.scaler import scale

img = scale(plt.imread('../data/tree.jpg')[:,:,0])
img_ft = np.fft.fft2(img)
img_ft_abs = np.log(np.abs(img_ft))
img_ft_abs_shifted = np.fft.fftshift(img_ft_abs)

### fig:tree-fourier
# plt.subplot(1,3,1)
# plt.imshow(img, cmap='gray')
# plt.axis("off")
# plt.gca().set_title('spatial domain', fontdict={'fontsize':18})
# plt.subplot(1,3,2)
# plt.imshow(img_ft_abs, cmap='gray')
# plt.axis("off")
# plt.gca().set_title('frequency domain', fontdict={'fontsize':18})
# plt.subplot(1,3,3)
# plt.imshow(img_ft_abs_shifted, cmap='gray')
# plt.axis("off")
# plt.gca().set_title('frequency domain shifted', fontdict={'fontsize':18})
# plt.tight_layout()
# plt.show()

### fig:tree-ilpf
# u = np.fft.fftfreq(img.shape[1])
# v = np.fft.fftfreq(img.shape[0])
# vv, uu = np.meshgrid(v, u, indexing='ij')
# filter_f = (np.hypot(vv, uu) < 0.05)
# img_ft = np.fft.fft2(img)
# img_filtered = np.fft.ifft2(img_ft*filter_f).real
# plt.subplot(1,3,1)
# plt.imshow(img, cmap='gray')
# plt.axis("off")
# plt.gca().set_title('img', fontdict={'fontsize':18})
# plt.subplot(1,3,2)
# plt.imshow(np.fft.fftshift(filter_f), cmap='gray')
# plt.axis("off")
# plt.gca().set_title('filter', fontdict={'fontsize':18})
# plt.subplot(1,3,3)
# plt.imshow(img_filtered, cmap='gray')
# plt.axis("off")
# plt.gca().set_title('filtered img', fontdict={'fontsize':18})
# plt.tight_layout()
# plt.show()

### fig:tree-gauss
# v = np.fft.fftfreq(img.shape[0])
# u = np.fft.fftfreq(img.shape[1])
# vv, uu = np.meshgrid(v, u, indexing='ij')
# sigma = 1/10
# filter_f = np.exp(-np.hypot(uu, vv)**2/(2*sigma**2))
# img_f = np.fft.fft2(img)
# img_filtered = np.fft.ifft2(img_f*filter_f).real
# plt.subplot(1,3,1)
# plt.imshow(img, cmap='gray')
# plt.axis("off")
# plt.gca().set_title('img', fontdict={'fontsize':18})
# plt.subplot(1,3,2)
# plt.imshow(np.fft.fftshift(filter_f), cmap='gray')
# plt.axis("off")
# plt.gca().set_title('filter', fontdict={'fontsize':18})
# plt.subplot(1,3,3)
# plt.imshow(img_filtered, cmap='gray')
# plt.axis("off")
# plt.gca().set_title('filtered img', fontdict={'fontsize':18})
# plt.tight_layout()
# plt.show()