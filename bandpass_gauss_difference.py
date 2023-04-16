import numpy as np
import matplotlib.pyplot as plt
from utils.file.dva_reader import read_dva
from skimage.filters import difference_of_gaussians

### read dva
# img = read_dva('./data/23_kep_test/Carotis 100%/CAR17IM0')
img = read_dva('./data/23_kep_test/Prostate/US_9660096_24_1.IMA')
img_f = np.fft.fft2(img)

### difference of gausssains with numpy
sigma_1 = 0.19
sigma_2 = 0.2
v = np.fft.fftfreq(img.shape[0])
u = np.fft.fftfreq(img.shape[1])
vv, uu = np.meshgrid(v, u, indexing='ij')
gauss_1 = np.exp(-(np.hypot(uu, vv))**2/(2*sigma_1**2))
gauss_2 = np.exp(-(np.hypot(uu, vv))**2/(2*sigma_2**2))
gauss_diff = gauss_2 - gauss_1
img_filtered = np.fft.ifft2(img_f * gauss_diff).real

### difference of gausssains with skimage
# img_filtered_3 = difference_of_gaussians(img, 0.19, 0.2)

### plotting filter
plt.figure(figsize=(12,7))
plt.imshow(np.fft.ifftshift(gauss_diff), cmap='gray')
plt.gca().set_title('bandpass')
plt.show()

### plot filtered image
plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.gca().set_title('img')
plt.subplot(1,2,2)
plt.imshow(img_filtered, cmap='gray')
plt.gca().set_title('img_filtered')
plt.show()

### high-pass emphasis
k_factor = 100
filter_emphasised_f = 1 + k_factor*(gauss_diff)
img_filtered_emphasised = np.fft.ifft2(img_f*filter_emphasised_f).real
plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
plt.imshow(img_filtered, cmap='gray')
plt.gca().set_title('original')
plt.subplot(1,2,2)
plt.imshow(img_filtered_emphasised, cmap='gray')
plt.gca().set_title('img_filtered_emphasised')
plt.show()

### summary plot
plt.figure(figsize=(12,7))
plt.suptitle("bandpass_gauss_difference", size=14)
plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.gca().set_title('img')
plt.subplot(1,3,2)
plt.imshow(img_filtered, cmap='gray')
plt.gca().set_title('img_filtered')
plt.subplot(1,3,3)
plt.imshow(img_filtered_emphasised, cmap='gray')
plt.gca().set_title('img_filtered_emphasised')
plt.show()

### save images as text
# np.savetxt('./outputs/bandpass_gauss_difference/img.txt', img, delimiter='\t')
# np.savetxt('./outputs/bandpass_gauss_difference/img_filtered_4.txt', img_filtered_4, delimiter='\t')
