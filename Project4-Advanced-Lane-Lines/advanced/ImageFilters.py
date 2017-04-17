import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle
import os
from advanced.CalibrateCamera import CalibrateCamera


class ImageFilters():
    """ 
    Class that handles image analysis and image filtering operations
    """

    def __init__(self, camCal, debug=False):
        """ Initialize ImageFilter"""
        # set debugging
        self.debug = debug

        # Frame number
        self.currentFrame = None

        # Copy of the camera calibration parameters
        self.mtx, self.dist, self.img_size = camCal.get()

        # Image size
        self.x, self.y = self.img_size

        # Mid point in picture (by height)
        self.mid = int(self.y/2)

        # Current Image RGB
        self.currentImage = np.zeros((self.y, self.x, 3), np.float32)

        # Current Image Top half RGB
        self.currentSkyRGB = np.zeros((self.mid, self.x, 3), np.float32)

        # Current Image Bottom half RGB
        self.currentRoadRGB = np.zeros((self.mid, self.x, 3), np.float32)

        # Current Sky Luma Image
        self.currentSkyL = np.zeros((self.mid, self.x, 3), np.float32)

        # Current Road Luma Image
        self.currentRoadL = np.zeros((self.mid, self.x, 3), np.float32)

        # Current Edge (both left and right)
        self.currentRoadEdge = np.zeros((self.mid, self.x), np.uint8)
        self.currentRoadEdgeProjected = np.zeros((self.y, self.x, 3), np.uint8)

        # Current Edge(Right Only)
        self.currentRoadRightEdge = np.zeros((self.mid, self.x), np.uint8)
        self.currentRoadRightEdgeProjected = np.zeros(
            (self.y, self.x, 3), np.uint8)

        # Current Edge (Left Only)
        self.currentRoadLeftEdge = np.zeros((self.mid, self.x), np.uint8)
        self.currentRoadLeftEdgeProjected = np.zeros(
            (self.y, self.x, 3), np.uint8)

        # image stats
        self.skylrgb = np.zeros((4), np.float32)
        self.roadlrgb = np.zeros((4), np.float32)
        self.roadbalance = 0.0
        self.horizonFound = False
        self.roadhorizon = 0
        self.visibility = 0

        # Texture Image Info
        self.skyText = 'NOIMAGE'
        self.skyImageQ = 'NOIMAGE'
        self.roadText = 'NOIMAGE'
        self.roadImageQ = 'NOIMAGE'

        # Set up debugging diag screens
        if self.debug:
            self.diag1 = np.zeros((self.mid, self.x, 3), np.float32)
            self.diag2 = np.zeros((self.mid, self.x, 3), np.float32)
            self.diag3 = np.zeros((self.mid, self.x, 3), np.float32)
            self.diag4 = np.zeros((self.mid, self.x, 3), np.float32)

    def makehalf(self, image, half=0):
        """Define a function to chop a picture in half horizontally"""
        if half == 0:
            if len(image.shape) < 3:
                newimage = np.copy(image[self.mid:self.y, :])
            else:
                newimage = np.copy(image[self.mid:self.y, :, :])
        else:
            if len(image.shape) < 3:
                newimage = np.copy(image[0:self.mid, :])
            else:
                newimage = np.copy(image[0:self.mid, :, :])
        return newimage

    def makefull(self, image, half=0):
        """ Function that makes a half picture whole horizontally """
        if len(image.shape) < 3:
            newimage = np.zeros((self.y, self.x), np.uint8)

        else:
            newimage = np.zeros((self.y, self.x, 3), np.uint8)

        if half == 0:
            if len(image.shape) < 3:
                newimage[0:self.mid:self.y, :] = image
            else:
                newimage[self.mid:self.y, :, :] = image

        else:
            if len(image.shape) < 3:
                newimage[0:self.mid, :] = image
            else:
                newimage[0:self.mid, :, :] = image

        return newimage

    def image_only_yellow_white(self, image):
        """Functions that masks out yellow lane lines"""
        # Setup range to mask off everything except white and  yellow
        lower_yellow_white = np.array([140, 140, 64])
        upper_yellow_white = np.array([255, 255, 255])
        mask = cv2.inRange(image, lower_yellow_white, upper_yellow_white)
        return cv2.bitwise_and(image, image, mask=mask)

    def gaussian_blur(self, img, kernel_size=3):
        """ Function that applies Gaussian Noise Kernel """
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def canny(self, img, low_threshold=0, high_threshold=255, kernel_size=3):
        """ Function that applies Canny Edge detection"""
        img = image_only_yellow_white(img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur_gray = gaussian_blur(gray, kernel_size)
        return cv2.Canny(blur_gray, low_threshold, high_threshold)

    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        """
        Function that applies Sobel x or y, then takes an absolute value
        and applies a threshold.
        """

        # 1. Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 2. Take the derivative in  x or y given orient='x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        elif orient == 'y':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # 3.Take the absolute value of the derivative
        abs_sobel = np.absolute(sobel)

        # 4. Scale to 8-bit (0-255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

        # 5. Create a copy and apply the Thresholding
        # Create a mask of 1's where the scaled gradient magnitude
        #    is > thresh_min and < thresh_max
        ret, binary_output = cv2.threshold(
            scaled_sobel, thresh[0], thresh[1], cv2.THRESH_BINARY)

        # 6. Return the result
        return binary_output

    def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
        """ Function that applies Sobel x and y, the magnitude of the gradient and then applies
        a threshold."""

        # 1. convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 2. Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # 3. Find the magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)

        # 4. Scale to 8-bit (0-255) then convert to np.uint8
        gradmag = np.uint8(255*gradmag/np.max(gradmag))

        # 5. Create a binary mask where mag thresholds are met
        ret, mag_binary = cv2.threshold(
            gradmag, mag_thresh[0], mag_thresh[1], cv2.THRESH_BINARY)

        # 6. Return the result
        return mag_binary

    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
        """ Function that applies Sobel x and y, then computes the direction
        of the gradient and applies a threshold.
        """
        # 1. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 2. Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # 3. Calculate the direction of the gradient and Take absolute value
        with np.errstate(divide='ignore', invalid='ignore'):
            dirout = np.absolute(np.arctan(sobely/sobelx))

            # 4. Create a binary mask where direction thresholds are met
            dir_binary = np.zeros_like(dirout).astype(np.float32)
            dir_binary[(dirout > thresh[0]) & (dirout < thresh[1])] = 1

            # 5. Return this mask as binary_output image
        # update nan to number
        np.nan_to_num(dir_binary)

        # make it fit
        dir_binary[(dir_binary > 0) | (dir_binary < 0)] = 128
        return dir_binary.astype(np.uint8)

    def miximg(self, img1, img2, α=0.8, ß=1., λ=0.):
        """
        The result image is computed as follows:
        img1 * α + img2 * β + λ
        NOTE: img1 and img2 must be the same shape!
        """
        return cv2.addWeighted(img1.astype(np.uint8), α, img2.astype(np.uint8),ß, λ)

    def hls_s(self, img, thresh=(0, 255)):
        """ Function that thresholds the S-channel of HLS"""
        # 1. Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s = hls[:, :, 2]

        # 2. Apply the threshold to the S-channel
        retval, s_binary = cv2.threshold(s.astype('uint8'), thresh[
                                         0], thresh[1], cv2.THRESH_BINARY)

        # 3. Return a binary image of threshold result
        return s_binary

    def hls_h(self, img, thresh=(0, 255)):
        """Function that thresholds the H-channel of HLS"""

        # 1. Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        h = hls[:, :, 0]

        # 2. Apply the threshold to the S-channel
        retval, h_binary = cv2.threshold(h.astype('uint8'), thresh[
                                         0], thresh[1], cv2.THRESH_BINARY)

        # 3. Return a binary image of threshold result
        return h_binary

    def edges(self, side=3):
        """ Function that retrieves edges detected by the filter combinations.
        This function is used by other modules to locate current binary image.
        """

        # piece together iamges that we want to project
        img = np.zeros((self.y, self.x, 3), np.uint8)

        if side == 1:
            img[self.mid:self.y, :, :] = np.dstack((self.currentRoadLeftEdge,
                                                    self.currentRoadLeftEdge,
                                                    self.currentRoadLeftEdge))
        elif side == 2:
            img[self.mid:self.y, :, :] = np.dstack((self.currentRoadRightEdge,
                                                    self.currentRoadRightEdge,
                                                    self.currentRoadRightEdge))

        else:
            img[self.mid:self.y, :, :] = np.dstack((self.currentRoadEdge,
                                                    self.currentRoadEdge,
                                                    self.currentRoadEdge))

        return img

    def imageQ(self, image):
        """Function to check image quality"""
        # Undistort the image
        self.currentImage = cv2.undistort(
            image, self.mtx, self.dist, None, self.mtx).astype(np.float32)

        # Convert to YUV space
        self.yuv = cv2.cvtColor(
            self.currentImage, cv2.COLOR_RGB2YUV).astype(np.float32)

        # Get some statistics for the sky image
        self.currentSkyL = self.yuv[0:self.mid, :, 0]
        self.currentSkyRGB[:, :] = self.currentImage[0:self.mid, :]
        self.skylrgb[0] = np.average(self.currentSkyL[0:self.mid, :])
        self.skylrgb[1] = np.average(self.currentSkyRGB[0:self.mid, :, 0])
        self.skylrgb[2] = np.average(self.currentSkyRGB[0:self.mid, :, 1])
        self.skylrgb[3] = np.average(self.currentSkyRGB[0:self.mid, :, 2])

        # Get some statistics for the road image
        self.currentRoadL = self.yuv[self.mid:self.y, :, 0]
        self.currentRoadRGB[:, :] = self.currentImage[self.mid:self.y, :]
        self.roadlrgb[0] = np.average(self.currentRoadL[0:self.mid, :])
        self.roadlrgb[1] = np.average(self.currentRoadRGB[0:self.mid, :, 0])
        self.roadlrgb[2] = np.average(self.currentRoadRGB[0:self.mid, :, 1])
        self.roadlrgb[3] = np.average(self.currentRoadRGB[0:self.mid, :, 2])

        # Sky image condition
        if self.skylrgb[0] > 160:
            self.skyImageQ = 'Sky Image: overexposed'

        elif self.skylrgb[0] < 50:
            self.skyImageQ = 'Sky Image: underexposed'

        elif self.skylrgb[0] > 143:
            self.skyImageQ = 'Sky Image: normal bright'

        elif self.skylrgb[0] < 113:
            self.skyImageQ = 'Sky Image: normal dark'

        else:
            self.skyImageQ = 'Sky Image: normal'

        # Sky detected weather or lighting conditions
        if self.skylrgb[0] > 128:
            if self.skylrgb[3] > self.skylrgb[0]:
                if self.skylrgb[1] > 120 and self.skylrgb[2] > 120:
                    if(self.skylrgb[2] - self.skylrgb[1]) > 20.0:
                        self.skyText = 'Sky Condition: tree shaded'
                    else:
                        self.skyText = 'Sky Condition: cloudy'
                else:
                    self.skyText = 'Sky Condition: clear'
            else:
                self.skyText = 'Sky Condition: UNKNOWN SKYL > 128'

        else:
            if self.skylrgb[2] > self.skylrgb[3]:
                self.skyText = 'Sky Condition: surrounded by trees'
                self.visibility = -80
            elif self.skylrgb[3] > self.skylrgb[0]:
                if(self.skylrgb[2] - self.skylrgb[1]) > 10.0:
                    self.skyText = 'Sky Condition: tree shaded'
                else:
                    self.skyText = 'Sky Condition: very cloudy or under overpass'
            else:
                self.skyText = 'Sky Condition: UNKNOWN!'

        self.roadbalance = self.roadlrgb[0] / 10.0

        # Road image condition
        if self.roadlrgb[0] > 160:
            self.roadImageQ = 'Road Image: overexposed'
        elif self.roadlrgb[0] < 50:
            self.roadImageQ = 'Road Image: underexposed'
        elif self.roadlrgb[0] > 143:
            self.roadImageQ = 'Road Image: normal bright'
        elif self.roadlrgb[0] < 113:
            self.roadImageQ = 'Road Image: normal dark'
        else:
            self.roadImageQ = 'Road Image: normal'

    def horizonDetect(self, debug=False, thresh=50):
        """Function to detect the horizon using the Sobel magnitude operation"""
        if not self.horizonFound:
            img = np.copy(self.currentRoadRGB).astype(np.uint8)
            magch = self.mag_thresh(img, sobel_kernel=9, mag_thresh=(30, 150))
            horizonLine = 50
            while not self.horizonFound and horizonLine < int(self.y/2):
                magchlinesum = np.sum(
                    magch[horizonLine:(horizonLine+1), :]).astype(np.float32)
                if magchlinesum > (self.x*thresh):
                    self.horizonFound = True
                    self.roadhorizon = horizonLine + int(self.y/2)
                    if debug:
                        self.diag4[horizonLine:(horizonLine+1), :, 0] = 255
                        self.diag4[horizonLine:(horizonLine+1), :, 1] = 255
                        self.diag4[horizonLine:(horizonLine+1), :, 2] = 0

                else:
                    horizonLine += 1

    def balanceEx(self):
        """ Function to balance the image exposure for easier line detection.
        """
        # Separate each of the RGB color channels
        r = self.currentRoadRGB[:, :, 0]
        g = self.currentRoadRGB[:, :, 1]
        b = self.currentRoadRGB[:, :, 2]

        # Get the Y channel (luma) from the YUV color space
        # and make two copies

        yo = np.copy(self.currentRoadL[:, :]).astype(np.float32)
        yc = np.copy(self.currentRoadL[:, :]).astype(np.float32)

        # Use the balance factor calculated previously to calculate the
        # corrected Y
        yc = (yc/self.roadbalance)*8.0

        # make a copy and threshold it to maximum value 255
        lymask = np.copy(yc)
        lymask[(lymask > 255.0)] = 255.0

        # Create another mask to mask yellow road markings
        uymask = np.copy(yc) * 0

        # Subtract the thresholded mask from the corrected Y.
        # Now we just have the peaks
        yc -= lymask

        # If we are dealing with an overexposed image, cap its corrected Y to
        # 242
        if self.roadlrgb[0] > 160:
            yc[(b > 254) & (g > 254) & (r > 254)] = 242.0

        # if we are dealing with a darker image
        # try to pickup faint blue and cap them to 242.

        elif self.roadlrgb[0] < 128:
            yc[(b > self.roadlrgb[3]) & (yo > 160+(self.roadbalance*20))] = 242.0
        else:
            yc[(b > self.roadlrgb[3]) & (yo > 210+(self.roadbalance*10))] = 242.0

        # Mask yellow lane lines
        uymask[(b < self.roadlrgb[0]) & (r > self.roadlrgb[0])
               & (g > self.roadlrgb[0])] = 242.0

        # Combined the corrected road luma and the masked yellow
        yc = self.miximg(yc, uymask, 1.0, 1.0)

        # Mix it back to the original luma.
        yc = self.miximg(yc, yo, 1.0, 0.8)

        # Resize the image to get the lane lines to the bottom
        yc[int((self.y/72)*70):self.y, :] = 0
        self.yuv[self.mid:self.y, :, 0] = yc.astype(np.uint8)
        self.yuv[(self.y - 40): self.y, :, 0] = yo[(self.mid - 40): self.mid, :].astype(np.uint8)

        # Convert back to RGB
        self.currentRoadRGB = cv2.cvtColor(
            self.yuv[self.mid:self.y, :, :], cv2.COLORYUV2RGB)

    def applyFilter1(self, side=3):
        """ Filter1"""
        # Run the functions
        img = np.copy(self.currentRoadRGB).astype(np.uint8)
        gradx = self.abs_sobel_thresh(img, orient='x', thresh=(25, 100))
        grady = self.abs_sobel_thresh(img, orient='y', thresh=(50, 150))
        magch = self.mag_thresh(img, sobel_kernel=9, mag_thresh=(50, 150))
        dirch = self.dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))
        sch = self.hls_s(img, thresh=(88, 190))
        hch = self.hls_h(img, thresh=(50, 100))

        # Output "masked_lines" is a single channel mask
        shadow = np.zeros_like(dirch).astype(np.uint8)
        shadow[(sch > 0) & (hch > 0)] = 128

        # Create the Red Filter
        rEdgeDetect = img[:, :, 0] / 4
        rEdgeDetect = 255 - rEdgeDetect
        rEdgeDetect[(rEdgeDetect > 210)] = 0

        # Build the combination
        combined = np.zeros_like(dirch).astype(np.uint8)
        combined[((gradx > 0) | (grady > 0) | ((magch > 0) & (dirch > 0)) | (
            sch > 0)) & (shadow == 0) & (rEdgeDetect > 0)] = 35
        if(side & 1) == 1:
            self.currentRoadLeftEdge = np.copy(combined)
        if(side & 2) == 2:
            self.currentRoadRightEdge = np.copy(combined)
        if(side & 3) == 3:
            self.currentRoadEdge = combined

        # Build diag screen if in debug mode
        if self.debug:
            # Create diagnostic screen 1-3
            # Create a blank color channel for combinatio
            ignore_color = np.copy(gradx) * 0
            self.diag1 = np.dstack((rEdgeDetect, gradx, grady))
            self.diag2 = np.dstack((ignore_color, magch, dirch))
            self.diag3 = np.dstack((sch, shadow, hch))
            self.diag4 = np.dstack((combined, combined, combined)) * 4

    def applyFilter2(self, side=3):
        """ Filter2"""
        # Run the functions
        img = np.copy(self.currentRoadRGB).astype(np.uint8)
        gradx = self.abs_sobel_thresh(img, orient='x', thresh=(25, 100))
        grady = self.abs_sobel_thresh(img, orient='y', thresh=(50, 150))
        magch = self.mag_thresh(img, sobel_kernel=9, mag_thresh=(50, 150))
        dirch = self.dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))
        sch = self.hls_s(img, thresh=(88, 190))
        hch = self.hls_h(img, thresh=(50, 100))

        # Output "masked_lines" is a single channel mask
        shadow = np.zeros_like(dirch).astype(np.uint8)
        shadow[(sch > 0) & (hch > 0)] = 128

        # Create the Red Filter
        rEdgeDetect = img[:, :, 0] / 4
        rEdgeDetect = 255 - rEdgeDetect
        rEdgeDetect[(rEdgeDetect > 210)] = 0

        # Build the combination
        combined = np.zeros_like(dirch).astype(np.uint8)
        combined[((gradx > 0) | (grady > 0) | ((magch > 0) & (dirch > 0)) | (
            sch > 0)) & (shadow == 0) & (rEdgeDetect > 0)] = 35
        combined[(grady > 0) & (dirch > 0) & (magch > 0)] = 35
        if(side & 1) == 1:
            self.currentRoadLeftEdge = np.copy(combined)
        if(side & 2) == 2:
            self.currentRoadRightEdge = np.copy(combined)
        if(side & 3) == 3:
            self.currentRoadEdge = combined

        # Build diag screen if in debug mode
        if self.debug:
            # Create diagnostic screen 1-3
            # Create a blank color channel for combinatio
            ignore_color = np.copy(gradx) * 0
            self.diag1 = np.dstack((rEdgeDetect, gradx, grady))
            self.diag2 = np.dstack((ignore_color, magch, dirch))
            self.diag3 = np.dstack((sch, shadow, hch))
            self.diag4 = np.dstack((combined, combined, combined)) * 4

    def applyFilter3(self, side=3):
        """ Filter3"""
        # Run the functions
        img = np.copy(self.currentRoadRGB).astype(np.uint8)
        gradx = self.abs_sobel_thresh(img, orient='x', thresh=(25, 100))
        grady = self.abs_sobel_thresh(img, orient='y', thresh=(50, 150))
        magch = self.mag_thresh(img, sobel_kernel=9, mag_thresh=(30, 150))
        dirch = self.dir_threshold(img, sobel_kernel=15, thresh=(0.6, 1.3))
        sch = self.hls_s(img, thresh=(20, 100))
        hch = self.hls_h(img, thresh=(125, 175))

        # Output "masked_lines" is a single channel mask
        shadow = np.zeros_like(dirch).astype(np.uint8)
        shadow[(sch > 0) & (hch > 0)] = 128

        # Create the Red Filter
        rEdgeDetect = img[:, :, 0] / 4
        rEdgeDetect = 255 - rEdgeDetect
        rEdgeDetect[(rEdgeDetect > 210)] = 0

        # Build the combination
        combined = np.zeros_like(dirch).astype(np.uint8)
        combined[((gradx > 0) | (grady > 0) | ((magch > 0) & (dirch > 0)) | (
            sch > 0)) & (shadow == 0) & (rEdgeDetect > 0)] = 35
        if(side & 1) == 1:
            self.currentRoadLeftEdge = np.copy(combined)
        if(side & 2) == 2:
            self.currentRoadRightEdge = np.copy(combined)
        if(side & 3) == 3:
            self.currentRoadEdge = combined

        # Build diag screen if in debug mode
        if self.debug:
            # Create diagnostic screen 1-3
            # Create a blank color channel for combinatio
            ignore_color = np.copy(gradx) * 0
            self.diag1 = np.dstack((rEdgeDetect, gradx, grady))
            self.diag2 = np.dstack((ignore_color, magch, dirch))
            self.diag3 = np.dstack((sch, shadow, hch))
            self.diag4 = np.dstack((combined, combined, combined)) * 4

    def applyFilter4(self, side=3):
        """ Filter4"""
        # Run the functions
        img = np.copy(self.currentRoadRGB).astype(np.uint8)
        gradx = self.abs_sobel_thresh(img, orient='x', thresh=(30, 100))
        grady = self.abs_sobel_thresh(img, orient='y', thresh=(75, 150))
        magch = self.mag_thresh(img, sobel_kernel=9, mag_thresh=(30, 150))
        dirch = self.dir_threshold(img, sobel_kernel=15, thresh=(0.6, 1.3))
        sch = self.hls_s(img, thresh=(20, 100))
        hch = self.hls_h(img, thresh=(125, 175))

        # Output "masked_lines" is a single channel mask
        shadow = np.zeros_like(dirch).astype(np.uint8)
        shadow[(sch > 0) & (hch > 0)] = 128

        # Create the Red Filter
        rEdgeDetect = img[:, :, 0] / 4
        rEdgeDetect = 255 - rEdgeDetect
        rEdgeDetect[(rEdgeDetect > 210)] = 0

        # Build the combination
        combined = np.zeros_like(dirch).astype(np.uint8)
        combined[((magch > 0) & (dirch > 0)) | (
            (rEdgeDetect > 192) & (rEdgeDetect < 200) & (magch > 0))] = 35
        if(side & 1) == 1:
            self.currentRoadLeftEdge = np.copy(combined)
        if(side & 2) == 2:
            self.currentRoadRightEdge = np.copy(combined)
        if(side & 3) == 3:
            self.currentRoadEdge = combined

        # Build diag screen if in debug mode
        if self.debug:
            # Create diagnostic screen 1-3
            # Create a blank color channel for combinatio
            ignore_color = np.copy(gradx) * 0
            self.diag1 = np.dstack((rEdgeDetect, gradx, grady))
            self.diag2 = np.dstack((ignore_color, magch, dirch))
            self.diag3 = np.dstack((sch, shadow, hch))
            self.diag4 = np.dstack((combined, combined, combined)) * 4

    def applyFilter5(self, side=3):
        """ Filter5"""
        # Run the functions
        img = np.copy(self.currentRoadRGB).astype(np.uint8)
        gradx = self.abs_sobel_thresh(img, orient='x', thresh=(25, 100))
        grady = self.abs_sobel_thresh(img, orient='y', thresh=(50, 150))
        magch = self.mag_thresh(img, sobel_kernel=9, mag_thresh=(30, 150))
        dirch = self.dir_threshold(img, sobel_kernel=15, thresh=(0.5, 1.3))
        sch = self.hls_s(img, thresh=(20, 80))
        hch = self.hls_h(img, thresh=(130, 175))

        # Output "masked_lines" is a single channel mask
        shadow = np.zeros_like(dirch).astype(np.uint8)
        shadow[(sch > 0) & (hch > 0)] = 128

        # Create the Red Filter
        rEdgeDetect = img[:, :, 0] / 4
        rEdgeDetect = 255 - rEdgeDetect
        rEdgeDetect[(rEdgeDetect > 220)] = 0

        # Build the combination
        combined = np.zeros_like(dirch).astype(np.uint8)
        combined[(rEdgeDetect > 192) & (rEdgeDetect < 205) & (sch > 0)] = 35
        if(side & 1) == 1:
            self.currentRoadLeftEdge = np.copy(combined)
        if(side & 2) == 2:
            self.currentRoadRightEdge = np.copy(combined)
        if(side & 3) == 3:
            self.currentRoadEdge = combined

        # Build diag screen if in debug mode
        if self.debug:
            # Create diagnostic screen 1-3
            # Create a blank color channel for combinatio
            ignore_color = np.copy(gradx) * 0
            self.diag1 = np.dstack((rEdgeDetect, gradx, grady))
            self.diag2 = np.dstack((ignore_color, magch, dirch))
            self.diag3 = np.dstack((sch, shadow, hch))
            self.diag4 = np.dstack((combined, combined, combined)) * 4

    def setProjection(self, projected, side=3):
        """ set the Projection image"""
        if(side & 1) == 1:
            self.currentRoadLeftEdgeProjected = np.copy(projected)
        if(side & 2) == 2:
            self.currentRoadRightEdgeProjected = np.copy(projected)
        if side == 3:
            self.currentRoadEdgeProjected = np.copy(projected)

    def getProjection(self, side=3):
        """ get the projection image"""
        if side == 1:
            return self.currentRoadLeftEdgeProjected
        if side == 2:
            return self.currentRoadRightEdgeProjected
        return self.currentRoadEdgeProjected

    def drawHorizon(self, image):
        """ Draw the discovered horizon in the image """
        horizonLine = self.roadhorizon
        image[horizonLine:(horizonLine+1), :, 0] = 255
        image[horizonLine:(horizonLine+1), :, 1] = 255
        image[horizonLine:(horizonLine+1), :, 2] = 0
