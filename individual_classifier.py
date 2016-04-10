# Import the required modules
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import argparse as ap
import glob
import os
from config import *

if __name__ == "__main__":
    # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('-f', "--featurepath", help="Path to the features root directory", required=True)
    parser.add_argument('-c', "--classifier", help="Classifier to be used", default="LIN_SVM")
    parser.add_argument('-o', "--objecttype", help="Object category")
    parser.add_argument('-m', "--modelpath", help="Path for model")

    args = vars(parser.parse_args())

    _feature_root = args["featurepath"]
    _object_tye = args["objecttype"]
    _model_path = args["modelpath"] + "/" + _object_tye

    _pos_f_path = _feature_root + "/" + _object_tye + "P"
    _neg_f_path = _feature_root + "/" + _object_tye + "N"


    label_map = {
        "Person" : 1,
        "Motorcycle" : 2,
        "Bicycle" : 3,
        "Car" : 4,
        "Rickshaw" : 5,
        "Autorickshaw" : 6
    }
    
    # If model directory don't exist, create
    if not os.path.isdir(_model_path):
        os.makedirs(_model_path)

    # Classifiers supported
    clf_type = args['classifier']

    fds = []
    labels = []

    # Load the positive features
    for feat_path in glob.glob(os.path.join(_pos_f_path,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(1)

    # Load the negative features
    for feat_path in glob.glob(os.path.join(_neg_f_path,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(0)

    if clf_type is "LIN_SVM":
        clf = LinearSVC()
        print "Training a Linear SVM Classifier"
        clf.fit(fds, labels)
        joblib.dump(clf, _model_path + "/" + _object_tye + ".model")
        print "Classifier saved to {}".format(model_path)
