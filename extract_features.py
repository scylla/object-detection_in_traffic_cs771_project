# Import the functions to calculate feature descriptors
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib

# To read file names
import argparse as ap
import glob
import os
from config import *

if __name__ == "__main__":
    # Argument Parser
    parser = ap.ArgumentParser()
    parser.add_argument('-d', "--datapath", help="Path to data folder",
            required=True)
    parser.add_argument('-f', "--featurepath", help="Path to feature folder",
            required=True)
    parser.add_argument('-u', "--descriptor", help="Descriptor to be used -- HOG",
            default="HOG")
    parser.add_argument('-o', "--objectcategory", help="Object category", required=True)

    args = vars(parser.parse_args())

    des_type = args["descriptor"]
    _data_path = args["datapath"]
    _feature_path = args["featurepath"]
    _object_type = args["objectcategory"]
    _pos_im_path = _data_path + "/" + _object_type + "P"
    _neg_im_path = _data_path + "/" + _object_type + "N"
    _pos_f_path = _feature_path + "/" + _object_type + "P"
    _neg_f_path = _feature_path + "/" + _object_type + "N"

    _pos_im_path = _data_path + "/" + _object_type
    _pos_f_path = _feature_path + "/" + _object_type

    print _pos_f_path, _pos_im_path

    # If feature directories don't exist, create them
    if not os.path.isdir(_pos_f_path):
        os.makedirs(_pos_f_path)

    # If feature directories don't exist, create them
    # if not os.path.isdir(_neg_f_path):
    #     os.makedirs(_neg_f_path)

    count = 0

    print "Calculating the descriptors for the positive samples and saving them"
    for im_path in glob.glob(os.path.join(_pos_im_path, "*")):
        im = imread(im_path, as_grey=True)
        if des_type == "HOG":
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
            print len(fd)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(_pos_f_path, fd_name)
        joblib.dump(fd, fd_path)
        count = count + 1
        if count > 5000:
            break


    print "Positive features saved in {}".format(_pos_f_path)

    # print "Calculating the descriptors for the negative samples and saving them"
    # for im_path in glob.glob(os.path.join(_neg_im_path, "*")):
    #     im = imread(im_path, as_grey=True)
    #     if des_type == "HOG":
    #         fd = hog(im,  orientations, pixels_per_cell, cells_per_block, visualize, normalize)
    #     fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
    #     fd_path = os.path.join(_neg_f_path, fd_name)
    #     joblib.dump(fd, fd_path)

    # print "Negative features saved in {}".format(_neg_f_path)

    print "Completed calculating features from training images"
