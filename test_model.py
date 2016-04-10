# Import the required modules
from skimage.transform import pyramid_gaussian
from skimage.io import imread
from skimage.feature import hog
from sklearn.externals import joblib
import cv2
import argparse as ap
from nms import nms
from config import *
from PIL import Image
from resizeimage import resizeimage
import numpy as np


if __name__ == "__main__":

    # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('-i', "--image", help="Path to the test image", required=True)
    parser.add_argument('-o', "--objectcategory", help="Object type to test for", required=True)
    parser.add_argument('-t', "--tempfolder", help="Folder for temp files", required=True)

    args = vars(parser.parse_args())

    _image_path = args["image"]
    _object_type = args["objectcategory"]
    _temp_path = args["tempfolder"]

    image = Image.open(_image_path)
    cover = resizeimage.resize_cover(image ,[100, 100], validate=False)
    cover.save(_temp_path + "/" + _object_type + ".png", image.format)
    
    # Read the image
    im = imread(_temp_path + "/" + _object_type + ".png", as_grey=True)

    # Load the classifier
    clf = joblib.load(model_path + "/" + _object_type + "/" + _object_type + ".model")

    fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)

    pred = clf.predict([fd])

    if pred == 1:
        print  "Detected " + _object_type
        print  "Confidence Score {} \n".format(clf.decision_function(fd))
    else :
        print "missed"
        print  "Confidence Score {} \n".format(clf.decision_function(fd))
