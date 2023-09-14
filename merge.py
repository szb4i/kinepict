import numpy as np
import matplotlib.pyplot as plt
from utils.img.scaler import scale

img_tiger = scale(plt.imread('./data/tiger.png')[:,:,2])
img_zebra = scale(plt.imread('./data/zebra.png')[:,:,2])
img_sum = img_tiger + img_zebra
img_tiger_f =np.fft.fft2(img_tiger)
img_zebra_f =np.fft.fft2(img_zebra)
img_output_f = img_tiger_f + img_zebra_f
img_output = np.fft.ifft2(img_output_f).real
plt.figure(figsize=(12,7))
plt.subplot(2,2,1)
plt.imshow(img_tiger, cmap='gray')
plt.subplot(2,2,2)
plt.imshow(img_zebra, cmap='gray')
plt.subplot(2,2,3)
plt.imshow(img_sum, cmap='gray')
plt.subplot(2,2,4)
plt.imshow(img_output, cmap='gray')
plt.show()