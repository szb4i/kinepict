import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import os

from utils.file.dva_reader import read_dva
from utils.img.scaler import scale


### single image
img = read_dva('./data/23_kep_test/Carotis 100%/CAR09IM3')
img = scale(img)
img_clahe_1 = exposure.equalize_adapthist(img,kernel_size=[img.shape[0]//8,img.shape[1]//8],clip_limit=0.00015,nbins=26200)
img_clahe_2 = exposure.equalize_adapthist(img,kernel_size=[img.shape[0]//15,img.shape[1]//15],clip_limit=0.00015,nbins=26200)

### multiple images
img_dir = '/data/23_kep_test/'
current_path = (os.path.abspath(os.path.dirname(__file__))).replace("\\","/")
img_dir_list = os.listdir(current_path + img_dir)
for i, dir_name in enumerate(img_dir_list):
  inner_img_dir = current_path + img_dir + dir_name
  image_list = os.listdir(inner_img_dir)
  for j, img_file_name in enumerate(image_list):
    if '.txt' == img_file_name[-4:]:
      continue
    img = read_dva(inner_img_dir + '/' + img_file_name)
    img = scale(img)
    ### different kernel sizes
    img_clahe_1 = exposure.equalize_adapthist(img,kernel_size=[50,50],clip_limit=0.00015,nbins=26200)
    img_clahe_2 = exposure.equalize_adapthist(img,kernel_size=[60,60],clip_limit=0.00015,nbins=26200)
    img_clahe_3 = exposure.equalize_adapthist(img,kernel_size=[70,70],clip_limit=0.00015,nbins=26200)
    img_clahe_4 = exposure.equalize_adapthist(img,kernel_size=[80,80],clip_limit=0.00015,nbins=26200)
    img_clahe_5 = exposure.equalize_adapthist(img,kernel_size=[90,90],clip_limit=0.00015,nbins=26200)
    img_clahe_6 = exposure.equalize_adapthist(img,kernel_size=[100,100],clip_limit=0.00015,nbins=26200)
    ### different clip limits
    # img_clahe_1 = exposure.equalize_adapthist(img,kernel_size=[70,70],clip_limit=0.00010,nbins=26200)
    # img_clahe_2 = exposure.equalize_adapthist(img,kernel_size=[70,70],clip_limit=0.00020,nbins=26200)
    # img_clahe_3 = exposure.equalize_adapthist(img,kernel_size=[70,70],clip_limit=0.00030,nbins=26200)
    # img_clahe_4 = exposure.equalize_adapthist(img,kernel_size=[70,70],clip_limit=0.00040,nbins=26200)
    # img_clahe_5 = exposure.equalize_adapthist(img,kernel_size=[70,70],clip_limit=0.00050,nbins=26200)
    # img_clahe_6 = exposure.equalize_adapthist(img,kernel_size=[70,70],clip_limit=0.00060,nbins=26200)
    plt.figure(figsize=(20,14))
    plt.suptitle(dir_name + '/' + img_file_name, size=16)
    plt.subplot(2,3,1)
    plt.imshow(img_clahe_1, cmap='gray')
    plt.subplot(2,3,2)
    plt.imshow(img_clahe_2, cmap='gray')
    plt.subplot(2,3,3)
    plt.imshow(img_clahe_3, cmap='gray')
    plt.subplot(2,3,4)
    plt.imshow(img_clahe_4, cmap='gray')
    plt.subplot(2,3,5)
    plt.imshow(img_clahe_5, cmap='gray')
    plt.subplot(2,3,6)
    plt.imshow(img_clahe_6, cmap='gray')
    plt.show()
    print('showing img: ' + dir_name + '/' + img_file_name)

### save
# np.savetxt('./outputs/img.txt', img, delimiter='\t')
# np.savetxt('./outputs/img_clahe_kernel_1.txt', img_clahe_1, delimiter='\t')
# np.savetxt('./outputs/img_clahe_kernel_2.txt', img_clahe_2, delimiter='\t')
# np.savetxt('./outputs/img_clahe_kernel_3.txt', img_clahe_3, delimiter='\t')
# np.savetxt('./outputs/img_clahe_kernel_4.txt', img_clahe_4, delimiter='\t')
# np.savetxt('./outputs/img_clahe_kernel_5.txt', img_clahe_5, delimiter='\t')
# np.savetxt('./outputs/img_clahe_kernel_6.txt', img_clahe_6, delimiter='\t')
