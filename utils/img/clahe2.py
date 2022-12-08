import numpy as np
import cv2



def clahe2(img):
  """
  source: https://stackoverflow.com/questions/43569566/adaptive-histogram-equalization-in-python
  """
  img_size=img.shape

  img_mod = np.zeros((600, 800))

  for i in range(0,img_size[0]-30):
      for j in range(0,img_size[1]-30):
          kernel = img[i:i+30,j:j+30]
          for k in range(0,30):
              for l in range(0,30):
                  element = kernel[k,l]
                  rank = 0
                  for m in range(0,30):
                      for n in range(0,30):
                          if(kernel[k,l]>kernel[m,n]):
                              rank = rank + 1
                  img_mod[i,j] = ((rank * 255 )/900)
  return img
