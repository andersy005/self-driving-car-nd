import numpy as np
import cv2
import pickle
import time
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from scipy.ndimage.measurements import label
from config import *
from features import *
from train import *


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
					xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
	# If x and/or y start/stop positions not defined, set to image size
	if x_start_stop[0] == None:
		x_start_stop[0] = 0
	if x_start_stop[1] == None:
		x_start_stop[1] = img.shape[1]
	if y_start_stop[0] == None:
		y_start_stop[0] = 0
	if y_start_stop[1] == None:
		y_start_stop[1] = img.shape[0]
	# Compute the span of the region to be searched
	xspan = x_start_stop[1] - x_start_stop[0]
	yspan = y_start_stop[1] - y_start_stop[0]
	# Compute the number of pixels per step in x/y
	nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
	ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
	# Compute the number of windows in x/y
	nx_windows = np.int(xspan/nx_pix_per_step) - 1
	ny_windows = np.int(yspan/ny_pix_per_step) - 1
	# Initialize a list to append window positions to
	window_list = []
	# Loop through finding x and y window positions
	# Note: you could vectorize this step, but in practice
	# you'll be considering windows one by one with your
	# classifier, so looping makes sense
	for ys in range(ny_windows):
		for xs in range(nx_windows):
			# Calculate window position
			startx = xs*nx_pix_per_step + x_start_stop[0]
			endx = startx + xy_window[0]
			starty = ys*ny_pix_per_step + y_start_stop[0]
			endy = starty + xy_window[1]

			# Append window position to list
			window_list.append(((startx, starty), (endx, endy)))
	# Return the list of windows
	return window_list


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
					spatial_size=(32, 32), hist_bins=32,
					hist_range=(0, 256), orient=9,
					pix_per_cell=8, cell_per_block=2,
					hog_channel=0, spatial_feat=True,
					hist_feat=True, hog_feat=True):

	#1) Create an empty list to receive positive detection windows
	on_windows = []
	#2) Iterate over all windows in the list
	for window in windows:
		#3) Extract the test window from original image
		test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
		#4) Extract features for that window using single_img_features()
		features = single_img_features(test_img, color_space=color_space,
							spatial_size=spatial_size, hist_bins=hist_bins,
							orient=orient, pix_per_cell=pix_per_cell,
							cell_per_block=cell_per_block,
							hog_channel=hog_channel, spatial_feat=spatial_feat,
							hist_feat=hist_feat, hog_feat=hog_feat)
		#5) Scale extracted features to be fed to classifier
		test_features = scaler.transform(np.array(features).reshape(1, -1))
		#6) Predict using your classifier
		prediction = clf.predict(test_features)
		#7) If positive (prediction == 1) then save the window
		if prediction == 1:
			on_windows.append(window)
	#8) Return windows for positive detections
	return on_windows


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
	# Make a copy of the image
	imcopy = np.copy(img)
	# Iterate through the bounding boxes
	for bbox in bboxes:
		# Draw a rectangle given bbox coordinates
		cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
	# Return the image copy with boxes drawn
	return imcopy


# Add heat to heatmap
def add_heat(heatmap, bbox_list):
	# Iterate through list of bboxes
	for box in bbox_list:
		# Add += 1 for all pixels inside each bbox
		# Assuming each "box" takes the form ((x1, y1), (x2, y2))
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

	# Return updated heatmap
	return heatmap


# Apply threshold to heat map
def apply_threshold(heatmap, threshold):
	# Zero out pixels below the threshold
	heatmap[heatmap <= threshold] = 0
	# Return thresholded map
	return heatmap


# Draw bounding boxes based on labels
def draw_labeled_bboxes(img, labels):
	# Iterate through all detected cars
	for car_number in range(1, labels[1]+1):
		# Find pixels with each car_number label value
		nonzero = (labels[0] == car_number).nonzero()

		# Identify x and y values of those pixels
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])

		# Define a bounding box based on min/max x and y
		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		bbox_w = (bbox[1][0] - bbox[0][0])
		bbox_h = (bbox[1][1] - bbox[0][1])

		# Filter final detections for aspect ratios, e.g. thin vertical box is likely not a car
		aspect_ratio = bbox_w / bbox_h  # width / height
		#print('ar: %s' % (aspect_ratio,))

		# Also if small box "close" to the car (i.e. bounding box y location is high),
		# then probaby not a car
		bbox_area = bbox_w * bbox_h

		if bbox_area < small_bbox_area and bbox[0][1] > close_y_thresh:
			small_box_close = True
		else:
			small_box_close = False

		# Combine above filters with minimum bbox area filter
		if aspect_ratio > min_ar and aspect_ratio < max_ar and not small_box_close and bbox_area > min_bbox_area:
			# Draw the box on the image
			cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)

	# Return the image
	return img


if __name__ == '__main__':
	svc = LinearSVC()
	X_scaler = StandardScaler()

	if use_pretrained:
		with open('model.p', 'rb') as f:
			save_dict = pickle.load(f)
		svc = save_dict['svc']
		X_scaler = save_dict['X_scaler']

		print('Loaded pre-trained model from model.p')
	else:
		print('Reading training data and training classifier from scratch')

		with open('data.p', 'rb') as f:
			data = pickle.load(f)
		cars = data['vehicles']
		notcars = data['non_vehicles']
		train(cars, notcars, svc, X_scaler)

		print('Training complete, saving trained model to model.p')

		with open('model.p', 'wb') as f:
			pickle.dump({'svc': svc, 'X_scaler': X_scaler}, f)

	# Display predictions on all test_images
	imdir = 'test_images'
	for image_file in os.listdir(imdir):
		image = mpimg.imread(os.path.join(imdir, image_file))
		draw_image = np.copy(image)

		windows = slide_window(image, x_start_stop=(0, 1280), y_start_stop=(500, 700),
						xy_window=(128, 128), xy_overlap=(pct_overlap, pct_overlap))
		windows = slide_window(image, x_start_stop=(100, 1180), y_start_stop=(400, 500),
						xy_window=(96, 96), xy_overlap=(pct_overlap, pct_overlap))

		hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
								spatial_size=spatial_size, hist_bins=hist_bins,
								orient=orient, pix_per_cell=pix_per_cell,
								cell_per_block=cell_per_block,
								hog_channel=hog_channel, spatial_feat=spatial_feat,
								hist_feat=hist_feat, hog_feat=hog_feat)

		window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

		plt.imshow(window_img)
		plt.show()

		# Calculate and draw heat map
		heatmap = np.zeros((720, 1280))  # NOTE: Image dimensions hard-coded
		heatmap = add_heat(heatmap, hot_windows)
		heatmap = apply_threshold(heatmap, heatmap_thresh)
		labels = label(heatmap)
		print(labels[1], 'cars found')
		plt.imshow(labels[0], cmap='gray')
		plt.show()

		# Draw final bounding boxes
		draw_img = draw_labeled_bboxes(np.copy(image), labels)
		plt.imshow(draw_img)
		plt.show()