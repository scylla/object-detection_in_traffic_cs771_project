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
import argparse
from skimage.transform import pyramid_gaussian

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=900, help="minimum area size")
ap.add_argument("-l", "--max-area", type=int, default=100000, help="maximum area size")
ap.add_argument('-t', "--tempfolder", help="Folder for temp files", required=True)
ap.add_argument('-m', "--modelpath", help="Path to model used", required=True)


# map for labels used
label_map = {
	1 : "Person",
	2 : "Motorcycle",
	3 : "Bicycle",
	4 : "Car",
	5 : "Rickshaw",
	6 : "Autorickshaw"
}

# function to do the prediction 
def getLabel(image_path):

	image = Image.open(image_path)
	cover = resizeimage.resize_cover(image ,[100, 100], validate=False)
	cover.save(_temp_path + "/" + _object_type + "_t.png", image.format)

	im = imread(_temp_path + "/" + _object_type + "_t.png", as_grey=True)

	fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
	pred = clf.predict([fd])

	return (pred, clf.decision_function(fd))


def sliding_window(image, window_size, step_size):
    for y in xrange(0, image.shape[0], step_size[1]):
        for x in xrange(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])	


args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	camera = cv2.VideoCapture(0)
	time.sleep(0.25)

# otherwise, we are reading from a video file
else:
	camera = cv2.VideoCapture(args["video"])

fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
video  = cv2.VideoWriter('generated.avi', fourcc, 10, (640,480), True)

_object_type = "Car"
_temp_path = args["tempfolder"]
_model_path = args["modelpath"]

# Load the classifier
clf = joblib.load(_model_path)
firstFrame = None

fgbg = cv2.BackgroundSubtractorMOG(history=5, nmixtures=3, backgroundRatio=0.5, noiseSigma=0.7)

count = 0
car_count = 0


# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied text
	(grabbed, frame) = camera.read()
	text = "Unoccupied"

	# if the frame could not be grabbed, then we have reached the end of the video
	if not grabbed:
		break

	# resize the frame, convert it to grayscale, and blur it
	# frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
	
	gray = fgbg.apply(gray, learningRate=0.1)

	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue

	thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	
	# dilate
	thresh = cv2.dilate(thresh,kernel,iterations = 13) 

	# contours
	(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_NONE)
	
	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < args["min_area"] or cv2.contourArea(c) > args["max_area"]:
			continue	

		M = cv2.moments(c)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])

		(x, y, w, h) = cv2.boundingRect(c)

		roi = frame[y:y+h, x:x+w]

		cv2.imwrite(_temp_path + "/" + _object_type + ".png", roi)
		output, label_dis = getLabel(_temp_path + "/" + _object_type + ".png")
		
		print output[0], label_dis[0][output-1], len(output), label_map[output[0]]

		margin_dis = label_dis[0][output-1]

		

		print "stats ", M, cX, cY

		if margin_dis > 0.2:
			car_count += 1
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			cv2.putText(frame, label_map[output[0]] + " " + str(margin_dis), (x + 8 ,y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 150, 150),2)
			# out = imutils.resize(frame, width=640, height= 480)
			# video.write(out)

	
	cv2.imshow("Security Feed", frame)

	# show the frame and record if the user presses a key
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
video.release()
camera.release()
cv2.destroyAllWindows()