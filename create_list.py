import pandas as pd
import os
import re
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to the input folder")
ap.add_argument("-r", "--rootprefix", help="root prefix for absolute path")
ap.add_argument("-o", "--output", help="output file")

args = vars(ap.parse_args())

_input_path = args["input"]
_root_path = args["rootprefix"]
_output_path = args["output"]

label_map = {
	"Person" : 1,
	"Motorcycle" : 2,
	"Bicycle" : 3,
	"Car" : 4,
	"Rickshaw" : 5,
	"Autorickshaw" : 6
}

file_path = []
file_label = []


for label in label_map:
	
	dir_path = _input_path + "/" + label
	files = os.listdir(dir_path)

	for fi in files:
		if re.search('.feat$' , fi):
			print fi
			file_path.append(_root_path + "/" + label + "/" + fi)
			file_label.append(label_map[label])


path_list = list(zip(file_path, file_label))
df = pd.DataFrame(data = path_list, columns=['Path', 'label'])
df.to_csv(_output_path, header=None, index=None, sep=' ', mode='a')

print df
