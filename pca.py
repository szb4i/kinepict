import matplotlib.pyplot as plt
import numpy as np
import time

from utils.file.dva_reader import read_dva
from utils.frangi.frangiFilter2D import FrangiFilter2D
from utils.file.dicom_reader import read_dicom
from skimage.filters import frangi, hessian
from skimage import morphology
from sklearn.decomposition import PCA
from utils.img.scaler import scale

# start = time.time()
# end = time.time()
# print(end - start)

# img_stack = read_dicom('./data/23_kep_test/Vena P/5_1_N')
img_stack = read_dicom('./data/23_kep_test/Vena I/1_1_N')
img_background = np.mean(img_stack[:,:,:3], axis=2)
img_dsa = np.amin(img_stack, axis=2) - np.amax(img_stack, axis=2)
img_dva = np.std(img_stack, axis=2, ddof=1)

# frangi_img, frangi_scale, frangi_direction = FrangiFilter2D(img, BlackWhite=False)
# https://scikit-image.org/docs/0.14.x/api/skimage.filters.html#skimage.filters.frangi
img_frangi = frangi(img_dsa, black_ridges=True)

### opening: eliminates single pixels (noise removal)
square = morphology.square(width=2)
img_morphological_opening = morphology.dilation(morphology.erosion(img_frangi, square))
### closing: fills holes (connecting vessels)
# disk = morphology.disk(radius=18)
# img_morphological_closing = morphology.erosion(morphology.dilation(img_frangi, disk))
# img_miklos = img_frangi - img_morphological_closing
### binary
img_morphological_opening[0.15 > img_morphological_opening] = 0
img_morphological_opening[0.15 < img_morphological_opening] = 1
img_morphological_opening = img_morphological_opening.astype(bool)

# img_frangi = frangi(img_dsa, black_ridges=True)
# plt.figure(figsize=(12,7))
# disk = morphology.disk(radius=2)
# img_morphological_opening = morphology.dilation(morphology.erosion(img_frangi, disk))
# plt.subplot(2,2,1)
# plt.imshow(img_morphological_opening, cmap='gray')
# disk = morphology.disk(radius=3)
# img_morphological_opening = morphology.dilation(morphology.erosion(img_frangi, disk))
# plt.subplot(2,2,2)
# plt.imshow(img_morphological_opening, cmap='gray')
# disk = morphology.disk(radius=4)
# img_morphological_opening = morphology.dilation(morphology.erosion(img_frangi, disk))
# plt.subplot(2,2,3)
# plt.imshow(img_morphological_opening, cmap='gray')
# disk = morphology.disk(radius=5)
# img_morphological_opening = morphology.dilation(morphology.erosion(img_frangi, disk))
# plt.subplot(2,2,4)
# plt.imshow(img_morphological_opening, cmap='gray')
# plt.show()

# img_frangi = frangi(img_dsa, black_ridges=True)
# plt.figure(figsize=(12,7))
# square = morphology.square(width=2)
# img_morphological_opening = morphology.dilation(morphology.erosion(img_frangi, square))
# plt.subplot(2,2,1)
# plt.imshow(img_morphological_opening, cmap='gray')
# square = morphology.square(width=3)
# img_morphological_opening = morphology.dilation(morphology.erosion(img_frangi, square))
# plt.subplot(2,2,2)
# plt.imshow(img_morphological_opening, cmap='gray')
# square = morphology.square(width=4)
# img_morphological_opening = morphology.dilation(morphology.erosion(img_frangi, square))
# plt.subplot(2,2,3)
# plt.imshow(img_morphological_opening, cmap='gray')
# square = morphology.square(width=5)
# img_morphological_opening = morphology.dilation(morphology.erosion(img_frangi, square))
# plt.subplot(2,2,4)
# plt.imshow(img_morphological_opening, cmap='gray')
# plt.show()

# square2 = morphology.square(width=2)
# img_morphological_opening_square2 = morphology.dilation(morphology.erosion(img_frangi, square2))
# square5 = morphology.square(width=5)
# img_morphological_opening_square5 = morphology.dilation(morphology.erosion(img_frangi, square5))
# plt.figure(figsize=(12,7))
# plt.imshow(img_morphological_opening_square2-img_morphological_opening_square5, cmap='gray')
# plt.show()

# img_frangi = frangi(img_dsa, black_ridges=True)
# plt.figure(figsize=(12,7))
# plt.subplot(2,2,1)
# disk = morphology.disk(radius=2)
# img_morphological_closing = morphology.erosion(morphology.dilation(img_frangi, disk))
# img_miklos = img_frangi - img_morphological_closing
# plt.imshow(img_miklos, cmap='gray')
# plt.subplot(2,2,2)
# disk = morphology.disk(radius=3)
# img_morphological_closing = morphology.erosion(morphology.dilation(img_frangi, disk))
# img_miklos = img_frangi - img_morphological_closing
# plt.imshow(img_miklos, cmap='gray')
# plt.subplot(2,2,3)
# disk = morphology.disk(radius=4)
# img_morphological_closing = morphology.erosion(morphology.dilation(img_frangi, disk))
# img_miklos = img_frangi - img_morphological_closing
# plt.imshow(img_miklos, cmap='gray')
# plt.subplot(2,2,4)
# disk = morphology.disk(radius=5)
# img_morphological_closing = morphology.erosion(morphology.dilation(img_frangi, disk))
# img_miklos = img_frangi - img_morphological_closing
# plt.imshow(img_miklos, cmap='gray')
# plt.show()


img_stack_flatten = np.reshape(img_stack, (img_stack.shape[0]*img_stack.shape[1], img_stack.shape[2]))

img_stack_flatten_vein_domain = img_stack_flatten[img_morphological_opening.flatten()]
threshold_variance_ratio = 1e-05
pca = PCA(n_components=img_stack_flatten_vein_domain.shape[1])
transformed = pca.fit_transform(img_stack_flatten_vein_domain)
transformed_copy = transformed.copy()
for i, variance_ratio in enumerate(pca.explained_variance_ratio_):
  # if variance_ratio < threshold_variance_ratio:
  if i > 2:
    transformed_copy[:, i:] = 0
    break
img_pca_filtered_vein_domain = pca.inverse_transform(transformed_copy)

img_stack_flatten_background_domain = img_stack_flatten[~img_morphological_opening.flatten()]
threshold_variance_ratio = 5e-05
pca = PCA(n_components=img_stack_flatten_background_domain.shape[1])
transformed = pca.fit_transform(img_stack_flatten_background_domain)
transformed_copy = transformed.copy()
for i, variance_ratio in enumerate(pca.explained_variance_ratio_):
  # if variance_ratio < threshold_variance_ratio:
  if i > 2:
    transformed_copy[:, i:] = 0
    break
img_pca_filtered_background_domain = pca.inverse_transform(transformed_copy)

img_stack_filtered = np.zeros((img_stack.shape[0], img_stack.shape[1], img_stack.shape[2]))
for k in range(0, img_stack.shape[2]):
  index_vein_domain = 0
  index_background_domain = 0
  for i in range(0, img_stack.shape[0]):
    for j in range(0, img_stack.shape[1]):
      if img_morphological_opening[i,j]:
        img_stack_filtered[i,j,k] = img_pca_filtered_vein_domain[index_vein_domain, k]
        index_vein_domain += 1
      else:
        img_stack_filtered[i,j,k] = img_pca_filtered_background_domain[index_background_domain, k]
        index_background_domain += 1

plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
plt.imshow(img_dva, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(np.std(img_stack_filtered, axis=2, ddof=1), cmap='gray')
plt.show()

# np.savetxt('./outputs/pca/img_dva' + '.txt', img_dva, delimiter='\t')
# np.savetxt('./outputs/pca/img_dsa' + '.txt', img_dsa, delimiter='\t')
# np.savetxt('./outputs/pca/img_dva_filtered' + '.txt', np.std(img_stack_filtered, axis=2, ddof=1), delimiter='\t')
# np.savetxt('./outputs/pca/img_dsa_filtered' + '.txt', np.amin(img_stack_filtered, axis=2) - np.amax(img_stack_filtered, axis=2), delimiter='\t')
