import numpy as np

def convolve2d_adaptive(img, kernel):
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
      for k in range(-kernel_dx, kernel_dx + 1):
        for l in range(-kernel_dy, kernel_dy + 1):
          local_diff = abs(img[x,y] - img[x+k,y+l])
          local_diff_sum += local_diff
          local_diff_max = local_diff if local_diff > local_diff_max else local_diff_max
      kernel_local = (1/9)*abs((local_diff_sum/(local_diff_max+0.0001))/kernel)
      output[x, y] = (kernel_local * img[x - kernel_dx: x + kernel_dx + 1, y - kernel_dy: y + kernel_dy + 1]).sum()
  return output
