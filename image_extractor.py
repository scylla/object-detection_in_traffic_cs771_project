# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4

import argparse
import datetime
import imutils
import time
import cv2
import os
import re
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-j", "--json", help="path to the json metadata")
ap.add_argument("-r", "--root", help="path to save the extracted data")

args = vars(ap.parse_args())

# parse the json metadata
import json
frame_dict  = {}

with open(args["json"]) as json_data:
    d = json.load(json_data)
    for item in d:
    	label = d[item]["label"]
    	for frame in d[item]["boxes"]:
    		if not frame in frame_dict:
    			frame_dict[frame] = []
    		if not d[item]["boxes"][frame]["occluded"] == 1 and not d[item]["boxes"][frame]["outside"] == 1:
    			temp_list = [d[item]["boxes"][frame]["xtl"], d[item]["boxes"][frame]["ytl"], d[item]["boxes"][frame]["xbr"], d[item]["boxes"][frame]["ybr"], label]
    			frame_dict[frame].append(temp_list)


# print frame_dict
root_path = args["root"]
_format = ".png"

# process video
camera = cv2.VideoCapture(args["video"])

count_p = 0
count_m = 0
count_b = 0
count_a = 0
count_r = 0
count_c = 0
count_nao = 0

label_person = "Person"
label_motorcycle = "Motorcycle"
label_bicyle = "Bicycle"
label_car = "Car"
label_autorickshaw = "Autorickshaw"
label_rickshaw = "Rickshaw"
label_nao = "NAO"

for fi in os.listdir(root_path + "/" + label_person):
	if re.search(_format , fi):
		count_p = count_p + 1

for fi in os.listdir(root_path + "/" + label_motorcycle):
	if re.search(_format , fi):
		count_m = count_m + 1

for fi in os.listdir(root_path + "/" + label_car):
	if re.search(_format , fi):
		count_c = count_c + 1

for fi in os.listdir(root_path + "/" + label_autorickshaw):
	if re.search(_format , fi):
		count_a = count_a + 1

for fi in os.listdir(root_path + "/" + label_rickshaw):
	if re.search(_format , fi):
		count_r = count_r + 1

for fi in os.listdir(root_path + "/" + label_bicyle):
	if re.search(_format , fi):
		count_b = count_b + 1

for fi in os.listdir(root_path + "/" + label_nao):
	if re.search(_format , fi):
		count_nao = count_nao + 1

count = 0

# safe to extract 

def safe_to_extract(x, y, denial_zone):
	for item in denial_zone:
		x_s, y_s = item[0]
		x_t, y_t = item[1]

		if (x > x_s and x < x_t) or (y > y_s and y < y_t):
			return False
		elif (x + 100 > x_s and x + 100 < x_t) or (y + 100 > y_s and y + 100 < y_t):
			return False
		else:
			continue

	return True		



# loop over the frames of the video
while True:

	(grabbed, frame) = camera.read()

	if not grabbed:
		break

	# a valid frame
	if not frame is None:
		print "in frame", count
		if str(count) in frame_dict:
			print "passed"
			cur_list = frame_dict[str(count)]
			for item in cur_list:
				x = item[0]
				y = item[1]
				h = item[3] - item[1]
				w = item[2] - item[0]

				roi = frame[y:y+h, x:x+w]
				label = item[4]
				if label == label_person:
					count_p = count_p + 1
					cv2.imwrite(root_path + "/" + label + "/" + str(count_p) + _format, roi)
				elif label == label_rickshaw:
					count_r = count_r + 1
					cv2.imwrite(root_path + "/" + label + "/" + str(count_r) + _format, roi)
				elif label == label_autorickshaw:
					count_a = count_a + 1
					cv2.imwrite(root_path + "/" + label + "/" + str(count_a) + _format, roi)
				elif label == label_car:
					count_c = count_c + 1
					cv2.imwrite(root_path + "/" + label + "/" + str(count_c) + _format, roi)
				elif label == label_bicyle:
					count_b = count_b + 1
					cv2.imwrite(root_path + "/" + label + "/" + str(count_b) + _format, roi)
				elif label == label_motorcycle:
					count_m = count_m + 1
					cv2.imwrite(root_path + "/" + label + "/" + str(count_m) + _format, roi)
				else :
					continue
					

		count = count + 1

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
print len(frame_dict)
print count_nao
print "Done!!!"