import matplotlib
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import pickle



def calibrate_camera():
	"""Function to calibrate the camera"""
	
	# Arrays to store object points and image points from all
	# the images
	objpoints = []   # 3D points in real world space
	imgpoints = []   # 2D points in image plane

	nx = 9
	ny = 6

	# Prepare object points, like (0, 0, 0), (1, 0, 0,) ...
	objp = np.zeros((nx * ny, 3), np.float32)
	objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
	# termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	images = glob.glob('camera_cal/*.jpg')

	# Go through all images and find corners
	for fname in images:
		# read in an image
		img = cv2.imread(fname)

		# convert to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Find the chess board corners
		ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
		# If found, add object points, image points (after refining them)
		if ret == True:
			objpoints.append(objp)
			cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
			imgpoints.append(corners)

			# Draw and display the corners
			img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
			#plt.imshow(img)
			cv2.imwrite('output_images/chess_corners.png', img)
		else:
			print('Warning: ret = {} for {}'.format(ret, fname))

	# calibrate camera and undistort a test image
	img = plt.imread('test_images/test2.jpg')
	img_size = (img.shape[1], img.shape[0])
	
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
	return mtx, dist

if __name__ == '__main__':
	mtx, dist = calibrate_camera()
	save_dict = {'mtx':mtx, 'dist':dist}
	with open('calibrate_camera.p', 'wb') as f:
		pickle.dump(save_dict, f)

	# Undistort example calibration image
	img = mpimg.imread('camera_cal/calibration5.jpg')
	dst = cv2.undistort(img, mtx, dist, None, mtx)

	# cv2.imwrite('output_images/undistort_calibration.png')
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
	ax1.imshow(img)
	ax1.set_title('Original Image', fontsize=30)
	ax2.imshow(dst)
	ax2.set_title('Undistorted Image', fontsize=30)
	plt.savefig('output_images/undistort_calibration.png')
	
	
 


			

	
