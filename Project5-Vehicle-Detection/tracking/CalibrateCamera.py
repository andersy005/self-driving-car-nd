import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import pickle
import os
import re


class CalibrateCamera():
    """ Initialize - either go through and calculate the camera calibration if 
        no pickle file exists. If the pickle file exists, just load the file.
    """

    def __init__(self, calibration_dir, pickle_file):
        """ Initialize CalibrateCamera"""
        self.mtx = None
        self.dist = None
        self.img_size = None

        if not os.path.isfile(pickle_file):

            # Mapping each calibration image to a number of checkboard corners
            objp_dict = {
                1: (9, 5),
                2: (9, 6),
                3: (9, 6),
                4: (9, 6),
                5: (9, 6),
                6: (9, 6),
                7: (9, 6),
                8: (9, 6),
                9: (9, 6),
                10: (9, 6),
                11: (9, 6),
                12: (9, 6),
                13: (9, 6),
                14: (9, 6),
                15: (9, 6),
                16: (9, 6),
                17: (9, 6),
                18: (9, 6),
                19: (9, 6),
                20: (9, 6),
            }
            # Arrays to store object points and image points from all
            # the images
            objpoints = []  # 3D points in real world space
            imgpoints = []  # 2D points in image plane

            # Go through all images and find corners
            for k in objp_dict:
                nx, ny = objp_dict[k]

                # Prepare object points, like (0, 0, 0), (1, 0, 0) ...
                objp = np.zeros((nx * ny, 3), np.float32)
                objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

                # Termination criteria
                criteria = (cv2.TERM_CRITERIA_EPS +
                            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

                # Make a list of calibration images
                fname = calibration_dir+'/calibration{}.jpg'.format(str(k))
                image = cv2.imread(fname)
                img = np.copy(image)
                self.img_size = (img.shape[1], img.shape[0])

                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Find the chessboard corners
                ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

                # If found, add object points, image points (after refining
                # them)
                if ret == True:
                    objpoints.append(objp)
                    cv2.cornerSubPix(gray, corners, (11, 11),
                                     (-1, -1), criteria)
                    imgpoints.append(corners)

                    # Draw and display the corners
                    img = cv2.drawChessboardCorners(
                        img, (nx, ny), corners, ret)
                    # plt.imshow(img)
                    cv2.imwrite(
                        '../output_images/calibrated{}.png'.format(str(k)), img)
                else:
                    print('Warning: ret = {} for {}'.format(ret, fname))

                 # Calibrate camera and distort
                ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints, imgpoints, self.img_size, None, None)

            # done and found all chessboard corners.
            # Save results into a pickle file for later retrieval

            try:
                with open(pickle_file, 'w+b') as f:
                    print('Saving data to pickle file :{}....'.format(pickle_file))
                    save_dict = {'img_size': self.img_size,
                                 'mtx': self.mtx,
                                 'dist': self.dist}
                    pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)
                    print('Camera Calibration Data saved to', pickle_file)

            except Exception as e:
                print('Unbale to save data to ', pickle_file, ':', e)
                raise

        # If previously saved pickle file of the distortion correction has been found
        # Retrieve it
        else:
            try:
                with open(pickle_file, 'rb') as f:
                    pickle_data = pickle.load(f)
                    self.img_size = pickle_data['img_size']
                    self.mtx = pickle_data['mtx']
                    self.dist = pickle_data['dist']
                    del pickle_data
                    print('Camera Calibration data restored from', pickle_file)

            except Exception as e:
                print('Unable to restore camera calibration data from',
                      pickle_file, ':', e)
                raise

     # Get the camera calibration result that the rest of the pipeline
    def get(self):
        """ Function that returns the camera calibration parameters"""
        return self.mtx, self.dist, self.img_size

    def setImageSize(self, img_shape):
        """ If the source image is now smaller than the original calibration image, 
        just set it.
        """
        self.img_size = (img_shape[1], img_shape[0])

    def getall(self):
        """ Get all the camera calibration result.
        """
        return self.mtx, self.dist, self.img_size, self.rvecs, self.tvecs
