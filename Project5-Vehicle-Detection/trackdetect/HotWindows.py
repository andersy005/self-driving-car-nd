import numpy as np
import cv2
from config import *


class HotWindows():
	"""
	Keep track of n previous hot windows
	Compute cumulative heat map over time
	self.windows is a queue of lists of bounding boxes,
	where the list can be of arbitrary size.
	Each element in the queue represents the list of
	bounding boxes at a particular time frame.
	"""
	def __init__(self, n):
		self.n = n
		self.windows = []  # queue implemented as a list

	def add_windows(self, new_windows):
		"""
		Push new windows to queue
		Pop from queue if full
		"""
		self.windows.append(new_windows)

		q_full = len(self.windows) >= self.n
		if q_full:
			_ = self.windows.pop(0)

	def get_windows(self):
		"""
		Concatenate all lists in the queue and return as
		one big list
		"""
		out_windows = []
		for window in self.windows:
			out_windows = out_windows + window
		return out_windows