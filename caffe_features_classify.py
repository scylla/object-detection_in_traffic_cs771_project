import numpy as np
import sys, time, glob
import pandas as pd
import os
import re
import argparse

caffe_root =  "/home/ubuntu/caffe/"
sys.path.insert(0, caffe_root + 'python')

import caffe
from sklearn.metrics import accuracy_score
from random import shuffle
from sklearn import svm
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to the input file")
ap.add_argument("-o", "--output", help="output file")

args = vars(ap.parse_args())
_input_file = args["input"]
_output_file = args["output"]


def init_net():
	net = caffe.Classifier(caffe_root  + 'models/bvlc_reference_caffenet/deploy.prototxt',
	                       caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
	# net.set_phase_test()
	# net.set_mode_cpu()
	# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
	# net.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))  # ImageNet mean
	# net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
	# net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
	net.set_mode_gpu()
	return net

def get_features(file, net):
	
	#print "getting features for", file
	scores = net.predict([caffe.io.load_image(file)])
	feat = net.blobs['fc6'].data[4][:,0, 0]
	feature = feat.copy()
	return feature

def shuffle_data(features, labels):
	new_features, new_labels = [], []
	index_shuf = range(len(features))
	shuffle(index_shuf)
	for i in index_shuf:
	    new_features.append(features[i])
	    new_labels.append(labels[i])

	return new_features, new_labels

def get_dataset(net, datar):
	
	feature_list = []
	label_list = []

	for item in xrange(len(datar)):
		feature_list.append(get_features(datar[0][item], net))
		label_list.append(datar[1][item])
	
	return (feature_list, label_list)

net = init_net()

datar = pd.read_csv(_input_file, header=None, sep=r"\s+")
train_x, label_x = get_dataset(net, datar)

final_dict = {
	"x_train" : train_x,
	"label_train" : label_x
}

out_file = open(_output_file, "wb")
pickle.dump(final_dict, out_file)

print "data saved!!!"

