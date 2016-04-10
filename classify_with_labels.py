# USAGE
# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4

# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
from skimage.io import imread
from skimage.feature import hog
from sklearn.externals import joblib
import argparse as ap
from nms import nms
from config import *
from PIL import Image
from resizeimage import resizeimage
import numpy as np


# map for labels used
label_map = {
	1 : "Person",
	2 : "Motorcycle",
	3 : "Bicycle",
	4 : "Car",
	5 : "Rickshaw",
	6 : "Autorickshaw"
}


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file", required=True)
ap.add_argument("-a", "--min-area", type=int, default=900, help="minimum area size")
ap.add_argument("-l", "--max-area", type=int, default=100000, help="maximum area size")
ap.add_argument('-t', "--tempfolder", help="Folder for temp files", required=True)
ap.add_argument('-m', "--modelpath", help="Path to model used", required=True)
ap.add_argument("-j", "--json", help="path to the json metadata")



fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')

video  = cv2.VideoWriter('just_classifier_wrt_boxes.avi', fourcc, 25, (640,480), True)

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

 # loaded dictionary   			

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	camera = cv2.VideoCapture(0)
	time.sleep(0.25)

# otherwise, we are reading from a video file
else:
	camera = cv2.VideoCapture(args["video"])

# initialize the first frame in the video stream
firstFrame = None

_object_type = "Car"
_temp_path = args["tempfolder"]
_model_path = args["modelpath"]

# Load the classifier
clf = joblib.load(_model_path)

# function to do the prediction 
def getLabel(image_path):

	image = Image.open(image_path)
	cover = resizeimage.resize_cover(image ,[100, 100], validate=False)
	cover.save(_temp_path + "/" + _object_type + "_t.png", image.format)

	im = imread(_temp_path + "/" + _object_type + "_t.png", as_grey=True)

	fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
	pred = clf.predict([fd])

	return (pred, clf.decision_function(fd))


car_count = 0

frame_count = 0
# loop over the frames of the video
while True:

	# grab the current frame and initialize the occupied/unoccupied
	# text
	(grabbed, frame) = camera.read()
	text = "Testing Video"

	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if not grabbed:
		break

	# draw the text and timestamp on the frame
	cv2.putText(frame, text, (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

	
	# label items when reading positions from JSON
	if not frame is None:
		if str(frame_count) in frame_dict:
			cur_list = frame_dict[str(frame_count)]
			print "in frame"
			for item in cur_list:
				x_l = item[0]
				y_l = item[1]
				h_l = item[3] - item[1]
				w_l = item[2] - item[0]
				roi = frame[y_l:y_l+h_l, x_l:x_l+w_l]
				cv2.imwrite(_temp_path + "/" + _object_type + ".png", roi)
				output, label_dis = getLabel(_temp_path + "/" + _object_type + ".png")
				margin_dis = label_dis[0][output-1]
				if margin_dis > 0.8:
					car_count += 1
					cv2.rectangle(frame, (x_l, y_l), (x_l + w_l, y_l + h_l), (0, 255, 0), 2)
					cv2.putText(frame, label_map[output[0]] + " " + str(margin_dis), (x_l + 8 ,y_l + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (250, 200, 200), 2)
				temp = imutils.resize(frame, width=640, height= 480)
				video.write(temp)	

	frame_count = frame_count + 1
	cv2.imshow("Testing from feeds", frame)
	
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
video.release()
camera.release()
cv2.destroyAllWindows()
print car_count, " cars detected!!!"