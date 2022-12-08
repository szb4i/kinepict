import numpy as np

def get_zone_plate_pattern(n_zones = 8.2, step = 0.0275):
  grid_size = np.arange(-n_zones, n_zones, step)
  xx, yy = np.meshgrid(grid_size, grid_size)
  pattern = (1/2*(1+np.cos(xx**2+yy**2))) * (np.hypot(xx, yy) < n_zones)
  return pattern