
import argparse
import datetime
import imutils
import time
import cv2
import os
import re
from PIL import Image
from resizeimage import resizeimage

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to the input folder")
ap.add_argument("-o", "--output", help="path to the output folder")
ap.add_argument("-s", "--suffix", help="suffix for image")

args = vars(ap.parse_args())
_format = ".png"
_root_path = args["input"]
_output_path = args["output"]
_suffix = args["suffix"]

count_p = 0

for fi in os.listdir(_root_path):
	if re.search(_format , fi):
		image = Image.open(_root_path + fi)
		cover = resizeimage.resize_cover(image ,[100, 100], validate=False)
		cover.save(_output_path + "/" + _suffix + fi, image.format)
		count_p = count_p + 1
		if count_p > 5000:
			break

print count_p, "Images scaled!!!"