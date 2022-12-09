import numpy as np
from scipy.ndimage import gaussian_filter

def apply_method4(img):
  img_height = img.shape[0]
  img_width = img.shape[1]
  ### gaussian blur
  img_blured = gaussian_filter(img, sigma=3)
  ### wl operator
  C = 9*np.identity(9)+np.full((9,9), -1)
  normalizing_const = 1/(9**2)
  kernel_dx = 1
  kernel_dy = 1
  img_wl = np.zeros((img_height, img_width))
  for x in range(kernel_dy, img_height - kernel_dy):
    for y in range(kernel_dx, img_width - kernel_dx):
      region_flatten = img_blured[x - kernel_dx: x + kernel_dx + 1, y - kernel_dy: y + kernel_dy + 1].flatten()
      img_wl[x,y] = normalizing_const*(region_flatten.dot(C.dot(region_flatten)))
  ### gamma
  img_wl_gamma = img_wl**0.5
  return img_wl_gamma
