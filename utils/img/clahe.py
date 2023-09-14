from skimage import exposure

def get_clahe_img(img_input):
    return exposure.equalize_adapthist(img_input,kernel_size=[img_input.shape[0]/10,img_input.shape[1]/10],clip_limit=0.00015,nbins=26200)