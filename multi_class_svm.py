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
from sklearn.metrics import accuracy_score
from sklearn import mixture
import numpy as np
from sklearn.cluster import KMeans


if __name__ == "__main__":

    # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('-f', "--featurepath", help="Path to the file of features list", required=True)
    parser.add_argument('-c', "--classifier", help="Classifier to be used", default="LIN_SVM")
    parser.add_argument('-o', "--objecttype", help="Object category", default="Common")
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

    # load fetures
    feature_list = []
    label_list = []

    datar = pd.read_csv(_feature_root, header=None, sep=r"\s+")

    for item in xrange(len(datar)):
        fd = joblib.load(datar[0][item])
        fds.append(fd)
        labels.append(datar[1][item])
    
    print "data loaded" 

    if clf_type is "LIN_SVM":
        
        # np.random.seed(1)
        # clf = mixture.GMM(n_components=6, covariance_type="tied")
        clf = KMeans(n_clusters=6)

        # clf = LinearSVC()
        print "Training a Linear SVM Classifier"
        clf.fit(fds, labels)
        joblib.dump(clf, _model_path + "/" + _object_tye + ".model")
        print "Classifier saved to {}".format(model_path)

        print "Testing on self"
        # predicted_labels = clf.predict(fds)
        # print "accuracy:", accuracy_score(labels, predicted_labels)



