# Import the required modules
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import argparse as ap
import glob
import os
from config import *
import pickle
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import preprocessing
import numpy as np


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

    # If model directory don't exist, create
    if not os.path.isdir(_model_path):
        os.makedirs(_model_path)

    # Classifiers supported
    clf_type = args['classifier']

    fds = []
    labels = []

    feature_data = open(_feature_root, "rb")
    features_dict = pickle.load(feature_data)

    x_train = features_dict["x_train"]
    labels = features_dict["label_train"]

    for item in x_train:
        fds.append(item[0])

    # X = np.array(fds)
    # X_scaled = preprocessing.scale(X)
    # scaler = preprocessing.StandardScaler().fit(X)

    if clf_type is "LIN_SVM":
        # clf = LinearSVC()
        clf = LinearSVC()
        print "Training a Linear SVM Classifier"
        clf.fit(fds, labels)
        joblib.dump(clf, _model_path + "/" + _object_tye + ".model")
        print "Classifier saved to {}".format(model_path)

    print "testing accuracy"
    predicted_labels = clf.predict(fds)
    print "accuracy:", accuracy_score(labels, predicted_labels)


























