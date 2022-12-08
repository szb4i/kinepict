import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy import ndimage, misc

from utils.patterns.rectangle import get_rectangle
from utils.kernels.sobel_kernels import get_sobel_x_kernel, get_sobel_y_kernel
from utils.kernels.zeropad_kernel import get_zeropadded_kernel
from utils.kernels.gaussian_kernel import get_gaussian_kernel
from utils.file.dva_reader import read_dva
from utils.file.dva_reader import read_dva
from utils.img.scaler import scale
from utils.img.clahe import clahe
import cv2 as cv

### read dva
img = read_dva('./data/X-ray 70%/hasE.IMA')
img = ((img/img.max())*255).astype(np.uint8)
# plt.imshow(img, cmap='gray')
# plt.show()

### high frequency emphasis
v = np.fft.fftfreq(img.shape[0])
u = np.fft.fftfreq(img.shape[1])
vv, uu = np.meshgrid(v, u, indexing='ij')
D0 = 10
k1 = 0.5  
k2 = 0.75
filter_f = k1 + k2*(1 - np.exp(-np.hypot(uu, vv)**2/(2*D0**2)))
img_f = np.fft.fft2(img)
img_filtered_s = np.fft.ifft2(img_f*filter_f).real


# https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
# img_filtered_s_8bit = ((img_filtered_s/img_filtered_s.max())*255).astype(np.uint8)
# clahe1 = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cl1 = clahe1.apply(img_filtered_s_8bit)

clahe_img = clahe(img_filtered_s,np.finfo('d').max,0,0)

np.savetxt('./outputs/method3' + '.txt', img_filtered_s, delimiter='\t')
