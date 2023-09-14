# 1) Fourier
# a) Fourier aluláteresztő gauss szűrés 1/5 1/pixel térfrekvenciánál.
# b) Fourier felüll áteresztő gauss szűrés 1/50 1/pixel térfrekvenciánál.
# c) Az a) és b) egyszerre: szávszűrés.
# d) Milyen képet mutat, ha csak a legnagyobb (vagy második legnagyobb) frekvencia nagyságát ábrázoljuk képként?
# ötlet: szűkebb sávszűrés? mi lenne ha a felül áteresztőt nagyobbra vennénk?
import numpy as np
import matplotlib.pyplot as plt
import time
from utils.file.dva_reader import read_dva
from utils.img.scaler import scale

### read dva
img = read_dva('./data/23_kep_test/Carotis 100%/CAR17IM0')
# img = read_dva('./data/23_kep_test/Prostate/US_9660096_24_1.IMA')
img = scale(img)

img_f = np.fft.fft2(img)

### a)
sigma_1 = 0.2
v = np.fft.fftfreq(img.shape[0])
u = np.fft.fftfreq(img.shape[1])
vv, uu = np.meshgrid(v, u, indexing='ij')
gauss_1 = np.exp(-(np.hypot(uu, vv))**2/(2*sigma_1**2))
img_filtered_a = np.fft.ifft2(img_f * gauss_1).real

### b)
sigma_1 = 0.02
v = np.fft.fftfreq(img.shape[0])
u = np.fft.fftfreq(img.shape[1])
vv, uu = np.meshgrid(v, u, indexing='ij')
gauss_1 = 1 - np.exp(-(np.hypot(uu, vv))**2/(2*sigma_1**2))
img_filtered_b = np.fft.ifft2(img_f * gauss_1).real

### c)
time_start = time.time()
sigma_1 = 0.02
# sigma_1 = 0.19
sigma_2 = 0.2
v = np.fft.fftfreq(img.shape[0])
u = np.fft.fftfreq(img.shape[1])
vv, uu = np.meshgrid(v, u, indexing='ij')
gauss_1 = np.exp(-(np.hypot(uu, vv))**2/(2*sigma_1**2))
gauss_2 = np.exp(-(np.hypot(uu, vv))**2/(2*sigma_2**2))
gauss_diff = gauss_2 - gauss_1
img_filtered_c = np.fft.ifft2(img_f * gauss_diff).real
time_end = time.time()
print(time_end-time_start)

plt.imshow(img + 100*img_filtered_c, cmap='gray')

# plt.figure(figsize=(12,7))
# plt.imshow(img + 100*img_filtered_c, cmap='gray')
# plt.show()



### d)
sigma_1 = 0.9
v = np.fft.fftfreq(img.shape[0])
u = np.fft.fftfreq(img.shape[1])
vv, uu = np.meshgrid(v, u, indexing='ij')
gauss_1 = 1 - np.exp(-(np.hypot(uu, vv))**2/(2*sigma_1**2))
img_filtered_d = np.fft.ifft2(img_f * gauss_1).real

### plot output
plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(img_filtered_c, cmap='gray')
plt.show()


### plot filter
# plt.figure(figsize=(12,7))
# plt.imshow(np.fft.ifftshift(gauss_diff), cmap='gray')
# plt.gca().set_title('bandpass')
# plt.show()

# np.savetxt('./outputs/algo1/gauss1-carotis.txt', img_filtered_c, delimiter='\t')