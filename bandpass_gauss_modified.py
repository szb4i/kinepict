import numpy as np
import matplotlib.pyplot as plt
from utils.file.dva_reader import read_dva

### read dva
# img = read_dva('./data/23_kep_test/Carotis 100%/CAR17IM0')
img = read_dva('./data/23_kep_test/Prostate/US_9660096_24_1.IMA')
img_f = np.fft.fft2(img)

### bandpass gauss
v = np.fft.fftfreq(img.shape[0])
u = np.fft.fftfreq(img.shape[1])
vv, uu = np.meshgrid(v, u, indexing='ij')
C0 = 0.25
W = 0.1
# img_filtered_ringing
# W = 0.075
filter_f = np.exp(-(np.hypot(uu, vv) - C0)**2/(W**2))
img_filtered = np.fft.ifft2(img_f * filter_f).real

### plotting filter
plt.figure(figsize=(12,7))
plt.imshow(np.fft.ifftshift(filter_f), cmap='gray')
plt.gca().set_title('bandpass')
plt.show()

### plotting filtered image
plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.gca().set_title('img')
plt.subplot(1,2,2)
plt.imshow(img_filtered, cmap='gray')
plt.gca().set_title('img_filtered')
plt.show()

### high-pass emphasis
k_factor = 5
filter_emphasised_f = 1 + k_factor*(filter_f)
img_filtered_emphasised = np.fft.ifft2(img_f*filter_emphasised_f).real
plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.gca().set_title('original')
plt.subplot(1,2,2)
plt.imshow(img_filtered_emphasised, cmap='gray')
plt.gca().set_title('img_filtered_emphasised')
plt.show()

### summary plot
plt.figure(figsize=(12,7))
plt.suptitle("bandpass_gauss_modified", size=14)
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
# np.savetxt('./outputs/bandpass_gauss_modified/img_filtered.txt', img_filtered, delimiter='\t')
