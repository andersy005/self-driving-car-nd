import numpy as np
import cv2

# Helper Function for image preprocessing

def crop_img(img, top=30, bottom=24, left=0, right=0) :
    h,w,_ = img.shape
    return img[top:h-bottom,left:w-right]

def MinMaxNorm(image, a=-.5, b=.5) :
    Xmin, Xmax = np.min(image), np.max(image)
    return (image-Xmin)*(b-a)/(Xmax-Xmin)

def AbsNorm(image, a=-.5, b=0.5, col_min=0, col_max=255) :
    return (image-col_min)*(b-a)/(col_max-col_min)

def contrast_norm(image) :
    """
    Applies histogram equilization on Y channel of YUV
    and returns normalized BGR image
    """
    # convert to YUV color space
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    new_image[:,:,0] = cv2.equalizeHist(new_image[:,:,0])
    new_image = cv2.cvtColor(new_image, cv2.COLOR_YUV2BGR)
    return AbsNorm(new_image)

def resize_img(img, new_size=(200,66)) :
    return cv2.resize(img, new_size, interpolation = cv2.INTER_AREA)

# main function for image preprocessing
def preprocess(img) : 
    # crop image 
    new_img = crop_img(img)
    # contrast normalization 
    new_img = contrast_norm(new_img)
    # resize to 200x66 
    return resize_img(new_img)