import numpy as np
import matplotlib.pyplot as plt
import time
from utils.file.dva_reader import read_dva
from utils.img.scaler import scale
from utils.filter.anisodiff import anisodiff
from skimage import exposure
from utils.kernels.gaussian_kernel import get_gaussian_kernel
from scipy.signal import convolve2d

### read dva
# img = read_dva('./data/23_kep_test/Carotis 100%/CAR17IM0')
img = read_dva('./data/23_kep_test/Prostate/US_9660096_24_1.IMA')
img = scale(img)
img_f = np.fft.fft2(img)

### c)
time_start = time.time()
# sigma_1 = 0.02
sigma_1 = 0.19
sigma_2 = 0.2
v = np.fft.fftfreq(img.shape[0])
u = np.fft.fftfreq(img.shape[1])
vv, uu = np.meshgrid(v, u, indexing='ij')
gauss_1 = np.exp(-(np.hypot(uu, vv))**2/(2*sigma_1**2))
gauss_2 = np.exp(-(np.hypot(uu, vv))**2/(2*sigma_2**2))
gauss_diff = gauss_2 - gauss_1
img_filtered_c = np.fft.ifft2(img_f * gauss_diff).real

img_sum = img + 100*img_filtered_c


# anisodiff nem segit sokat
img_smooth = anisodiff(img_sum)
img_smooth = scale(img_smooth)
img_clahe = exposure.equalize_adapthist(img_smooth,kernel_size=[90,90],clip_limit=0.00015,nbins=26200)

kernel = get_gaussian_kernel()
img_blured = convolve2d(img_clahe, kernel, mode='same')

plt.figure(figsize=(12,7))
plt.imshow(img_smooth, cmap='gray')
plt.show()