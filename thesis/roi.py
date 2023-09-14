import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import exposure

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.file.dva_reader import read_dva
from utils.img.scaler import scale

img_1 = read_dva('./data/23_kep_test/Prostate/US_9660096_24_1.IMA')
img_1 = scale(img_1)
img_clahe_1 = exposure.equalize_adapthist(img_1,kernel_size=[img_1.shape[0]/10,img_1.shape[1]/10],clip_limit=0.00015,nbins=26200)
img_1_roi_signal_in_vein_1 = [[697, 700], [638, 655]]
img_1_roi_signal_out_vein_1 = [[718, 721], [638, 655]]
img_1_roi_signal_in_vein_2 = [[455, 462], [874, 882]]
img_1_roi_signal_out_vein_2 = [[469, 476], [893, 890]]
img_1_roi_signal_in_vein_3 = [[355, 368], [616, 629]]
img_1_roi_signal_out_vein_3 = [[404, 417], [605, 618]]

img_2 = read_dva('./data/23_kep_test/Carotis 100%/CAR17IM0')
img_2 = scale(img_2)
img_clahe_2 = exposure.equalize_adapthist(img_2,kernel_size=[img_2.shape[0]/10,img_2.shape[1]/10],clip_limit=0.00015,nbins=26200)
img_2_roi_signal_in_vein_1 = [[529, 533], [79, 90]]
img_2_roi_signal_out_vein_1 = [[543, 547], [79, 90]]
img_2_roi_signal_in_vein_2 = [[413, 418], [241, 253]]
img_2_roi_signal_out_vein_2 = [[427, 432], [241, 253]]
img_2_roi_signal_in_vein_3 = [[259, 270], [297, 322]]
img_2_roi_signal_out_vein_3 = [[288, 299], [295, 320]]

fontsize = 18
plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
plt.gca().set_title('Prostate', fontdict={'fontsize': fontsize})
plt.gca().add_patch(Rectangle((img_1_roi_signal_in_vein_1[0][0],img_1_roi_signal_in_vein_1[1][0]),
                    img_1_roi_signal_in_vein_1[0][1] - img_1_roi_signal_in_vein_1[0][0],
                    img_1_roi_signal_in_vein_1[1][1] - img_1_roi_signal_in_vein_1[1][0],
                    edgecolor='blue',
                    facecolor='none',
                    lw=1))
plt.gca().add_patch(Rectangle((img_1_roi_signal_out_vein_1[0][0],img_1_roi_signal_out_vein_1[1][0]),
                    img_1_roi_signal_out_vein_1[0][1] - img_1_roi_signal_out_vein_1[0][0],
                    img_1_roi_signal_out_vein_1[1][1] - img_1_roi_signal_out_vein_1[1][0],
                    edgecolor='red',
                    facecolor='none',
                    lw=1))
plt.gca().add_patch(Rectangle((img_1_roi_signal_in_vein_2[0][0],img_1_roi_signal_in_vein_2[1][0]),
                    img_1_roi_signal_in_vein_2[0][1] - img_1_roi_signal_in_vein_2[0][0],
                    img_1_roi_signal_in_vein_2[1][1] - img_1_roi_signal_in_vein_2[1][0],
                    edgecolor='blue',
                    facecolor='none',
                    lw=1))
plt.gca().add_patch(Rectangle((img_1_roi_signal_out_vein_2[0][0],img_1_roi_signal_out_vein_2[1][0]),
                    img_1_roi_signal_out_vein_2[0][1] - img_1_roi_signal_out_vein_2[0][0],
                    img_1_roi_signal_out_vein_2[1][1] - img_1_roi_signal_out_vein_2[1][0],
                    edgecolor='red',
                    facecolor='none',
                    lw=1))
plt.gca().add_patch(Rectangle((img_1_roi_signal_in_vein_3[0][0],img_1_roi_signal_in_vein_3[1][0]),
                    img_1_roi_signal_in_vein_3[0][1] - img_1_roi_signal_in_vein_3[0][0],
                    img_1_roi_signal_in_vein_3[1][1] - img_1_roi_signal_in_vein_3[1][0],
                    edgecolor='blue',
                    facecolor='none',
                    lw=1))
plt.gca().add_patch(Rectangle((img_1_roi_signal_out_vein_3[0][0],img_1_roi_signal_out_vein_3[1][0]),
                    img_1_roi_signal_out_vein_3[0][1] - img_1_roi_signal_out_vein_3[0][0],
                    img_1_roi_signal_out_vein_3[1][1] - img_1_roi_signal_out_vein_3[1][0],
                    edgecolor='red',
                    facecolor='none',
                    lw=1))
plt.axis("off")
plt.imshow(img_clahe_1, cmap='gray')
plt.subplot(1,2,2)
plt.gca().set_title('Carotis', fontdict={'fontsize': fontsize})
plt.gca().add_patch(Rectangle((img_2_roi_signal_in_vein_1[0][0],img_2_roi_signal_in_vein_1[1][0]),
                    img_2_roi_signal_in_vein_1[0][1] - img_2_roi_signal_in_vein_1[0][0],
                    img_2_roi_signal_in_vein_1[1][1] - img_2_roi_signal_in_vein_1[1][0],
                    edgecolor='blue',
                    facecolor='none',
                    lw=1))
plt.gca().add_patch(Rectangle((img_2_roi_signal_out_vein_1[0][0],img_2_roi_signal_out_vein_1[1][0]),
                    img_2_roi_signal_out_vein_1[0][1] - img_2_roi_signal_out_vein_1[0][0],
                    img_2_roi_signal_out_vein_1[1][1] - img_2_roi_signal_out_vein_1[1][0],
                    edgecolor='red',
                    facecolor='none',
                    lw=1))
plt.gca().add_patch(Rectangle((img_2_roi_signal_in_vein_2[0][0],img_2_roi_signal_in_vein_2[1][0]),
                    img_2_roi_signal_in_vein_2[0][1] - img_2_roi_signal_in_vein_2[0][0],
                    img_2_roi_signal_in_vein_2[1][1] - img_2_roi_signal_in_vein_2[1][0],
                    edgecolor='blue',
                    facecolor='none',
                    lw=1))
plt.gca().add_patch(Rectangle((img_2_roi_signal_out_vein_2[0][0],img_2_roi_signal_out_vein_2[1][0]),
                    img_2_roi_signal_out_vein_2[0][1] - img_2_roi_signal_out_vein_2[0][0],
                    img_2_roi_signal_out_vein_2[1][1] - img_2_roi_signal_out_vein_2[1][0],
                    edgecolor='red',
                    facecolor='none',
                    lw=1))
plt.gca().add_patch(Rectangle((img_2_roi_signal_in_vein_3[0][0],img_2_roi_signal_in_vein_3[1][0]),
                    img_2_roi_signal_in_vein_3[0][1] - img_2_roi_signal_in_vein_3[0][0],
                    img_2_roi_signal_in_vein_3[1][1] - img_2_roi_signal_in_vein_3[1][0],
                    edgecolor='blue',
                    facecolor='none',
                    lw=1))
plt.gca().add_patch(Rectangle((img_2_roi_signal_out_vein_3[0][0],img_2_roi_signal_out_vein_3[1][0]),
                    img_2_roi_signal_out_vein_3[0][1] - img_2_roi_signal_out_vein_3[0][0],
                    img_2_roi_signal_out_vein_3[1][1] - img_2_roi_signal_out_vein_3[1][0],
                    edgecolor='red',
                    facecolor='none',
                    lw=1))
plt.axis("off")
plt.imshow(img_clahe_2, cmap='gray')
plt.tight_layout()
plt.show()
