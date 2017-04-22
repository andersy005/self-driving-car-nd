'''
Gather all images and labels and save them to pickle file
Saved file is a list of RGB image arrays, for vehicle and non-vehicle
Make sure your data is organized in the following directory structure:
data
	non-vehicles
	vehicles
Where 'non-vehicles' and 'vehices' contains the image data from Udacity
Then, run the following from the command line:
find data/non-vehicles -name '*.png' > nv_files.txt
find data/vehicles -name '*.png' > v_files.txt
'''

import numpy as np
import pickle 
import cv2


non_vehicles = []
vehicles = []


# Non-vehicles
with open('nv_files.txt', 'r') as f:
	lines = f.readlines()

for line in lines:
	line = line[:-1]  # strip trailing newline

	img = cv2.imread(line)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	non_vehicles.append(img)

# Vehicles
with open('v_files.txt', 'r') as f:
	lines = f.readlines()

for line in lines:
	line = line[:-1]  # strip trailing newline

	img = cv2.imread(line)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	vehicles.append(img)

# Save to pickle file
save_dict = {'non_vehicles': np.array(non_vehicles), 'vehicles': np.array(vehicles)}
with open('data.p', 'wb') as f:
	pickle.dump(save_dict, f)