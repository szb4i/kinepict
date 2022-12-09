import numpy as np
from scipy.ndimage import gaussian_filter

def apply_method5(img):
  img_height = img.shape[0]
  img_width = img.shape[1]
  ### gaussian blur
  img_blured = gaussian_filter(img, sigma=3)
  ### edge enhancement
  kernel_dx = 1
  kernel_dy = 1
  img_edge = np.zeros((img_height, img_width))
  for x in range(kernel_dy, img_height - kernel_dy):
    for y in range(kernel_dx, img_width - kernel_dx):
      partition = img_blured[x - kernel_dx: x + kernel_dx + 1, y - kernel_dy: y + kernel_dy + 1]
      img_edge[x,y] = partition.max() - partition.min()
  return img_edge
