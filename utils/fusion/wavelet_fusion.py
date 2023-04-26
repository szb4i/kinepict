### image fusion using wavelet transform
import pywt
import numpy as np
from utils.img.scaler import scale

# This function does the coefficient fusing according to the fusion method
def _fuse_coeff(cooef_1, cooef_2, method):
    if (method == 'mean'):
        cooef = (cooef_1 + cooef_2) / 2
    elif (method == 'min'):
        cooef = np.minimum(cooef_1, cooef_2)
    elif (method == 'max'):
        cooef = np.maximum(cooef_1, cooef_2)
    else:
        cooef = []
    return cooef

def fuse(img_1, img_2, fusion_method = 'mean'):
    """
    param img_1: 1st image to fuse
    param img_2: 2nd image to fuse
    param fusion_method: can be 'min' || 'max || 'mean' or anything else if implemented in _fues_coeff
    """

    # First: Do wavelet transform on each image
    wavelet = 'db1'
    cooef_1 = pywt.wavedec2(img_1, wavelet)
    cooef_2 = pywt.wavedec2(img_2, wavelet)

    # Second: for each level in both image do the fusion according to the desire option
    fused_cooef = []
    for i in range(len(cooef_1)-1):
        # The first values in each decomposition is the apprximation values of the top level
        if(i == 0):
            fused_cooef.append(_fuse_coeff(cooef_1[0], cooef_2[0], fusion_method))
        else:
            # For the rest of the levels we have tupels with 3 coeeficents
            c1 = _fuse_coeff(cooef_1[i][0], cooef_2[i][0], fusion_method)
            c2 = _fuse_coeff(cooef_1[i][1], cooef_2[i][1], fusion_method)
            c3 = _fuse_coeff(cooef_1[i][2], cooef_2[i][2], fusion_method)
            fused_cooef.append((c1,c2,c3))

    # Third: After we fused the cooefficent we need to transfor back to get the image
    img_fused = pywt.waverec2(fused_cooef, wavelet)

    # Forth: normmalize values
    img_fused = scale(img_fused)
    return img_fused