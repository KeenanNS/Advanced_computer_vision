import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import requests
import urllib
import os
from PIL import Image
from kaggle.api.kaggle_api_extended import KaggleApi

from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

system(kaggle datasets download -d ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes)
system(unzip crop-and-weed-detection-data-with-bounding-boxes.zip)

img_path = 'data/agri_0_%.jp*g' % idx
txt_path = 'data/agri_0_%.txt' % idx
labels = []



converted_data_train = {
    'image_name': [],
    'x_min': [],
    'y_min': [],
    'x_max': [],
    'y_max': [],
    'class_name': [],
}

converted_data_test = {
    'image_name': [],
    'x_min': [],
    'y_min': [],
    'x_max': [],
    'y_max': [],
    'class_name': [],
}

converted_data = {
    'image_name': [],
    'x_min': [],
    'y_min': [],
    'x_max': [],
    'y_max': [],
    'class_name': [],
}
for idx in range (num_entries):
	numbers = []
	file = open(txt_path, 'a')
	bounding_boxes = file.readlines()
	for i in range len(bounding_boxes):
		string = bounding_boxes[i] 
		for num in string.split():
			if num.isdigit():
				numbers.append(int(num))
	converted_data['image_name'].append(txt_path)
	converted_data['x_min'].append()

	

