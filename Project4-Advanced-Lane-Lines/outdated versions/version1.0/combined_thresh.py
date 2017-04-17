import matplotlib
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import pickle



def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    thresh_min,thresh_max = thresh[0],thresh[1]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if orient=='x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    elif orient=='y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel)
    abs_sobelxy = np.sqrt(np.power(sobelx,2)+np.power(sobely,2))
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    grad_dir = np.absolute(np.arctan2(abs_sobely, abs_sobelx))
    dir_binary = np.zeros_like(grad_dir)
    dir_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    
    return dir_binary

def hls_thresh(img, thresh=(100, 255)):
    # convert RGB to HLS and threshold to binary image using
    # S channel
    
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def pipeline(image, ksize=3):
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(50,100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(50,100 ))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(50, 205))
    dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
    hls_binary = hls_thresh(image, thresh=(170, 255))
    combined = np.zeros_like(dir_binary)
    combined[(((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))) | (hls_binary == 1)] = 1
    
    return gradx, grady, mag_binary, dir_binary, hls_binary, combined


if __name__ == '__main__':
    image = mpimg.imread('test_images/test5.jpg')
    
    # Choose a Sobel Kernel size
    ksize = 5
    
    with open('calibrate_camera.p', 'rb') as f:
        save_dict = pickle.load(f)
        
    mtx = save_dict['mtx']
    dist = save_dict['dist']
    
    # Undistort the image
    image_undistort = cv2.undistort(image, mtx, dist, None, mtx)
    
    gradx, grady, mag_binary, dir_binary, hls_binary, combined = pipeline(image_undistort, ksize)
    
    
    # Plot the results
    fontsize=20
    f, (ax1, ax2,ax3,ax4, ax5) = plt.subplots(1, 5, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(gradx, cmap='gray')
    ax1.set_title('Thresholded Gradient x', fontsize=fontsize)
    ax2.imshow(grady, cmap='gray')
    ax2.set_title('Thresholded Gradient y', fontsize=fontsize)
    ax3.imshow(mag_binary, cmap='gray')
    ax3.set_title('Thresholded Grad. Mag.', fontsize=fontsize)
    ax4.imshow(dir_binary, cmap='gray')
    ax4.set_title('Thresholded Grad. Dir.', fontsize=fontsize)
    ax5.imshow(hls_binary, cmap='gray')
    ax5.set_title('Thresholded Color.', fontsize=fontsize)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig('output_images/thresh_pipeline.png')


    plt.figure()
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image_undistort)
    ax1.set_title('Original Image', fontsize=fontsize)
    ax2.imshow(combined, cmap='gray')
    ax2.set_title('Combined', fontsize=fontsize)
    plt.savefig('output_images/combined_thresh.png')
    plt.show()
    




