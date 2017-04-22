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
from moviepy.editor import VideoFileClip
from HotWindows import HotWindows
from config import *
from features import *
from train import *
from windows import *


# Global variables for moviepy to work
hot_windows = HotWindows(num_frames)
svc = LinearSVC()
X_scaler = StandardScaler()

# MoviePy video annotation will call this function
def annotate_image(image):
	"""
	Annotate the input image with detection boxes
	Returns annotated image
	"""
	global hot_windows, svc, X_scaler

	draw_image = np.copy(image)

	windows = slide_window(image, x_start_stop=(100, 1180), y_start_stop=(400, 500),
						xy_window=(96, 96), xy_overlap=(pct_overlap, pct_overlap))

	new_hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
							spatial_size=spatial_size, hist_bins=hist_bins,
							orient=orient, pix_per_cell=pix_per_cell,
							cell_per_block=cell_per_block,
							hog_channel=hog_channel, spatial_feat=spatial_feat,
							hist_feat=hist_feat, hog_feat=hog_feat)

	# DEBUG
	window_img = draw_boxes(draw_image, new_hot_windows, color=(0, 0, 255), thick=6)
	#return window_img

	# Add new hot windows to HotWindows queue
	hot_windows.add_windows(new_hot_windows)
	all_hot_windows = hot_windows.get_windows()

	# Calculate and draw heat map
	heatmap = np.zeros((720, 1280))  # NOTE: Image dimensions hard-coded
	heatmap = add_heat(heatmap, all_hot_windows)
	heatmap = apply_threshold(heatmap, heatmap_thresh)
	labels = label(heatmap)

	# Draw final bounding boxes
	draw_img = draw_labeled_bboxes(np.copy(image), labels)

	return draw_img


def annotate_video(input_file, output_file):
	""" Given input_file video, save annotated video to output_file """
	global hot_windows, svc, X_scaler

	with open('model.p', 'rb') as f:
		save_dict = pickle.load(f)
	svc = save_dict['svc']
	X_scaler = save_dict['X_scaler']

	print('Loaded pre-trained model from model.p')

	video = VideoFileClip(input_file)
	annotated_video = video.fl_image(annotate_image)
	annotated_video.write_videofile(output_file, audio=False)


if __name__ == '__main__':
	# Annotate the video
	#annotate_video('test_video.mp4', 'test_out.mp4')
	#annotate_video('debug2.mp4', 'debug2_out.mp4')
	#annotate_video('debug3.mp4', 'debug3_out.mp4')
	annotate_video('../project_video.mp4', 'out.mp4')