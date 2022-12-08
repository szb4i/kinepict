import numpy as np

def get_box_kernel(n = 3):
  return np.ones((n, n), dtype=np.float)/n**2