import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle
from combined_thresh import pipeline


def perspective_transform(img):
    img_size = (img.shape[1], img.shape[0])
    
    src = np.float32(
              [[200, 720],
               [1100, 720],
               [595, 450],
               [685, 450]])
    
    dst = np.float32(
               [[300, 720],
               [980, 720],
               [300, 0],
               [980, 0]])
    
    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)
    
    warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
    unwarped = cv2.warpPerspective(img, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)
    
    return warped, unwarped, m, m_inv


if __name__ == '__main__':
    image = mpimg.imread('test_images/test5.jpg')
    
    # Choose a Sobel Kernel size
    ksize = 5
    
    with open('calibrate_camera.p', 'rb') as f:
        save_dict = pickle.load(f)
        
    mtx = save_dict['mtx']
    dist = save_dict['dist']
    
    # Undistort the image
    image = cv2.undistort(image, mtx, dist, None, mtx)
    
    gradx, grady, mag_binary, dir_binary, hls_binary, image = pipeline(image, ksize)
    
    warped, unwarped, m, m_inv = perspective_transform(image)
    
    # Visualize Perspective transform 
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))

    ax1.set_title('Source image')
    ax1.imshow(image, cmap='gray')

    ax2.set_title('Warped image')
    ax2.imshow(warped, cmap='gray')

    ax3.set_title('unwarped image')
    ax3.imshow(unwarped, cmap='gray')
    plt.savefig('output_images/perspective_trans.png')
    plt.show()