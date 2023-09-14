import numpy as np
import matplotlib.pyplot as plt
from utils.file.dva_reader import read_dva
from skimage.filters import difference_of_gaussians
from utils.kernels.gaussian_kernel import get_gaussian_kernel
from scipy.signal import convolve2d


### difference of gausssains with numpy
def enhance(img):
    img_f = np.fft.fft2(img)
    sigma_1 = 0.19
    sigma_2 = 0.2
    v = np.fft.fftfreq(img.shape[0])
    u = np.fft.fftfreq(img.shape[1])
    vv, uu = np.meshgrid(v, u, indexing='ij')
    gauss_1 = np.exp(-(np.hypot(uu, vv))**2/(2*sigma_1**2))
    gauss_2 = np.exp(-(np.hypot(uu, vv))**2/(2*sigma_2**2))
    gauss_diff = gauss_2 - gauss_1
    img_filtered = np.fft.ifft2(img_f * gauss_diff).real
    k_factor = 100
    filter_emphasised_f = 1 + k_factor*(gauss_diff)
    img_filtered_emphasised = np.fft.ifft2(img_f*filter_emphasised_f).real
    return img_filtered_emphasised

def denoise(img):
    kernel = get_gaussian_kernel()
    conv_img = convolve2d(img, kernel, mode='same')
    return conv_img


# img = read_dva('./data/23_kep_test/Carotis 100%/CAR17IM0')
img = read_dva('./data/23_kep_test/Prostate/US_9660096_24_1.IMA')

img_blured = denoise(img)
img_filtered = enhance(img)

img_enhanced_and_blured = denoise(enhance(img))
img_blured_and_enhanced = enhance(denoise(img))
plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
plt.imshow(img_enhanced_and_blured, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(img_blured_and_enhanced, cmap='gray')
plt.show()