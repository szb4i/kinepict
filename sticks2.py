import numpy as np
import matplotlib.pyplot as plt

from utils.file.dva_reader import read_dva
from utils.kernels.sticks import get_sticks

img = plt.imread('./data/tree.jpg')
img = img[:,:,0]
img = img/img.max()
noise = np.random.normal(0, 0.1, img.shape)
img = img+noise
img[img<0] = 0
# img = read_dva('./data/23_kep_test/Carotis 100%/CAR09IM3')
img_padded = np.pad(img, pad_width=3, mode='edge')
img_padded_height = img_padded.shape[0]
img_padded_width = img_padded.shape[1]
img_padded_sliding_window = np.lib.stride_tricks.sliding_window_view(img_padded, (7,7))

### pick max of sticks
sticks = get_sticks()
img_transformed = np.zeros((img.shape[0], img.shape[1], sticks.shape[0]))
for i in range(0, sticks.shape[0]):
    img_transformed[:,:,i] = np.sum(img_padded_sliding_window*sticks[i], axis=(2,3))
img_transformed = np.amax(img_transformed, axis=(2))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(img_transformed, cmap='gray')
plt.show()