# Import the required modules
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import argparse as ap
import glob
import os
from config import *
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.cluster import KMeans

if __name__ == "__main__":
    # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('-f', "--featurepath", help="Path to the file of features list", required=True)
    parser.add_argument('-m', "--modelpath", help="Path for model")

    args = vars(parser.parse_args())

    _feature_root = args["featurepath"]
    _model_path = args["modelpath"]

    # Load the classifier
    clf = joblib.load(_model_path)

    fds = []
    labels = []

    datar = pd.read_csv(_feature_root, header=None, sep=r"\s+")

    # print len(datar)

    for item in xrange(len(datar)):
        fd = joblib.load(datar[0][item])
        fds.append(fd)
        labels.append(datar[1][item])
    
    predicted_labels = clf.predict(fds)

    print "confusion matrix : "
    print confusion_matrix(labels, predicted_labels)
    print "-----------------------------------------------------------------"
    print classification_report(labels, predicted_labels)
    print "-----------------------------------------------------------------"
    print "accuracy:", accuracy_score(labels, predicted_labels)
    print "-----------------------------------------------------------------"
    



