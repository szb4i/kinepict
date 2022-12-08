import numpy as np

def get_rectangle(outer_x=512, outer_y=512, inner_x=50, inner_y=50):
  grid_size_x = np.arange(-outer_x//2, outer_x//2, 1)
  grid_size_y = np.arange(-outer_y//2, outer_y//2, 1)
  xx, yy = np.meshgrid(grid_size_x, grid_size_y)
  rectangle_boolean = (abs(xx) < inner_x) & (abs(yy) < inner_y)
  rectangle_float = np.array(rectangle_boolean, dtype=float)
  return rectangle_float
