import numpy as np

from utils.kernels.laplacian_kernel import get_laplacian_kernel

def convolve2d_adaptive_laplacian(img):
  kernel = get_laplacian_kernel()
  kernel = np.flipud(np.fliplr(kernel))
  kernel_dx = kernel.shape[1] // 2
  kernel_dy = kernel.shape[0] // 2
  img_height = img.shape[0]
  img_width = img.shape[1]
  output = np.zeros((img_height, img_width))
  for x in range(1, img_height - 1):
    for y in range(1, img_width - 1):
      local_diff_sum = 0
      local_diff_max = 0
      local_max = img[x,y]
      local_min = img[x,y]
      for k in range(-kernel_dx, kernel_dx + 1):
        for l in range(-kernel_dy, kernel_dy + 1):
          current_kernel_pixel = img[x+k,y+l]
          local_max = current_kernel_pixel if current_kernel_pixel > local_max else local_max
          local_min = current_kernel_pixel if current_kernel_pixel < local_min else local_min
          local_diff = abs(img[x,y] - current_kernel_pixel)
          local_diff_sum += local_diff
          local_diff_max = local_diff if local_diff > local_diff_max else local_diff_max
      local_diff_sum = (1/9)*local_diff_sum
      kernel_local = (local_diff_sum/(local_diff_max))/kernel
      new_pixel_value = (kernel_local * img[x - kernel_dx: x + kernel_dx + 1, y - kernel_dy: y + kernel_dy + 1]).sum()
      # if new_pixel_value > local_max:
      #   new_pixel_value = local_max
      # elif new_pixel_value < local_min:
      #   new_pixel_value = local_min
      # else:
      #   print('nothing')
      output[x, y] = new_pixel_value
  return output
