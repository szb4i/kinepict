import numpy as np
import matplotlib.pyplot as plt
from utils.file.dva_reader import read_dva
from skimage.filters import difference_of_gaussians
from utils.kernels.gaussian_kernel import get_gaussian_kernel
from scipy.signal import convolve2d

def denoise(img, kernel_size=3):
    kernel = get_gaussian_kernel(kernel_size)
    conv_img = convolve2d(img, kernel, mode='same')
    return conv_img

img = read_dva('./data/23_kep_test/Prostate/US_9660096_24_1.IMA')

# img_blured_3 = denoise(img, 3)
# img_blured_5 = denoise(img, 5)
# plt.figure(figsize=(12,7))
# plt.subplot(1,2,1)
# plt.imshow(img_blured_3, cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(img_blured_5, cmap='gray')
# plt.show()

while True:
    img = denoise(img)
    plt.figure(figsize=(12,7))
    plt.imshow(img, cmap='gray')
    plt.show()


### save images as text
# np.savetxt('./outputs/gaussian_blur/img_blured_3.txt', img_blured_3, delimiter='\t')