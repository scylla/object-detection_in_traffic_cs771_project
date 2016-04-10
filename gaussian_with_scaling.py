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
from imutils.object_detection import non_max_suppression

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


# for detecting faces
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
recognizer = cv2.createLBPHFaceRecognizer()


min_wdw_sz = (100, 100)
step_size = (20, 20)

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


# scale and pyramid search
def scale_pyramid(im_path):

    detections = []
    downscale = 1.5    
    
    # The current scale of the image
    scale = 0

    im = imread(im_path, as_grey=True)
    	
    top_2 = []

    # Downscale the image and iterate
    for im_scaled in pyramid_gaussian(im, downscale=downscale):

        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break

        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue

            # Calculate the HOG features
            fd = hog(im_window, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
            pred = clf.predict(fd)
            dec_score = clf.decision_function(fd)

            if(dec_score[0][pred - 1] > 0.5):
            	new_tuple = (x, y, dec_score[0][pred - 1], int(min_wdw_sz[0]*(downscale**scale)),
                	int(min_wdw_sz[1]*(downscale**scale)), pred)

            	if len(top_2) < 2:
            		top_2.append(new_tuple)
            	else:
            		if new_tuple[2] > top_2[0][2] and pred != top_2[0][5]:
            			top_2[0] = new_tuple
            		elif new_tuple[2] > top_2[1][2] and pred != top_2[0][5]:
            			top_2[1] = new_tuple
            		else:
            			continue
            
        scale+=1.25

    return top_2

# function ends


args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	camera = cv2.VideoCapture(0)
	time.sleep(0.25)

# otherwise, we are reading from a video file
else:
	camera = cv2.VideoCapture(args["video"])

fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
video  = cv2.VideoWriter('MOG_scaling_nov.avi', fourcc, 10, (640,480), True)

_object_type = "Car"
_temp_path = args["tempfolder"]
_model_path = args["modelpath"]

# Load the classifier
clf = joblib.load(_model_path)
firstFrame = None

fgbg = cv2.BackgroundSubtractorMOG(history=2, nmixtures=5, backgroundRatio=0.6, noiseSigma=0.7)

count = 0
car_count = 0
frame_count = 0

# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied text
	(grabbed, frame) = camera.read()
	text = "Unoccupied"

	# if the frame could not be grabbed, then we have reached the end of the video
	if not grabbed:
		break

	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=800)
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

		(x, y, w, h) = cv2.boundingRect(c)
		roi = frame[y:y+h, x:x+w]

		cv2.imwrite(_temp_path + "/" + _object_type + ".png", roi)

		detections = scale_pyramid(_temp_path + "/" + _object_type + ".png")

		items_in_win = 0

		detections = detections[:2]

		label_shown = ""	

		offset_val = 0

		for item in detections:

			margin_dis = item[2]
			h_new = item[3]
			w_new = item[4]
			label = item[5][0]
			
			if margin_dis > 0.8:
				label_shown = label_map[label] + " :: " + label_shown
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
				cv2.putText(frame, label_map[label] + " " + str(margin_dis), (x + 2 ,y + 16 + offset_val), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 200, 200),1)    		
				offset_val = offset_val + 12
				temp = imutils.resize(frame, width=640, height= 480)
				video.write(temp)

			items_in_win += 1

		offset_val = 0

	cv2.imshow("Testing Video", frame)
	frame_count = frame_count + 1
	if frame_count > 1000:
		break

	# show the frame and record if the user presses a key
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
video.release()
camera.release()
cv2.destroyAllWindows()