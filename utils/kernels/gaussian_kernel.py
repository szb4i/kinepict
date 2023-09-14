import numpy as np
   
def get_gaussian_kernel(l=3, sig=1., normalize=True):
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    if normalize:
        return kernel / np.sum(kernel)
    return kernel
