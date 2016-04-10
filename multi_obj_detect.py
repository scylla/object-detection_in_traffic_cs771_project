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

fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')

video  = cv2.VideoWriter('generated.avi', fourcc, 15, (640,480), True)

args = vars(ap.parse_args())

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

	# resize the frame, convert it to grayscale, and blur it
	# frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue

	# compute the absolute difference between the current frame and
	# first frame
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=5)
	(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)

	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < args["min_area"] or cv2.contourArea(c) > args["max_area"]:
			continue

		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		
		roi = frame[y:y+h, x:x+w]
		cv2.imwrite(_temp_path + "/" + _object_type + ".png", roi)
		output, label_dis = getLabel(_temp_path + "/" + _object_type + ".png")
		
		print output[0], label_dis[0][output-1], len(output), label_map[output[0]]

		margin_dis = label_dis[0][output-1]

		if margin_dis > 0.8:
			car_count += 1
			if output[0] == 2 and margin_dis < 1.5:
				continue
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			cv2.putText(frame, label_map[output[0]] + " " + str(margin_dis), (x + 8 ,y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (250, 200, 200), 2)
			frame = imutils.resize(frame, width=640, height= 480)
			video.write(frame)

	# draw the text and timestamp on the frame
	cv2.putText(frame, text, (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

	# show the frame and record if the user presses a key
	cv2.imshow("Security Feed", frame)
	# cv2.imshow("Thresh", thresh)
	# cv2.imshow("Frame Delta", frameDelta)
	
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
video.release()
camera.release()
cv2.destroyAllWindows()
print car_count, " cars detected!!!"