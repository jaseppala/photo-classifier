from tensorflow.keras import models
from data import get_image_dict
import os
import cv2
import numpy as np
import shutil

pipeline = models.load_model('blur_detection_model')

path = os.path.join('..', 'raw_data', 'test')

img_dict = {file:0 for file in os.listdir(path) if not os.path.isdir(os.path.join(path, file))}

for file in os.listdir(path): 
    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):                                    # get every file in the folder
        img = cv2.imread(os.path.join(path, file))                  # load the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res = cv2.resize(gray, dsize=(100, 100))             # make it RGB (cv2 uses BGR)
        img_dict[file] = res

blurry_dump = os.path.join(path,'blurry')

if not os.path.exists(blurry_dump):
    os.mkdir(blurry_dump)

for k in img_dict.keys():
    pred = pipeline.predict(np.expand_dims(img_dict[k], axis = 0))
    if pred > 0.5:
        shutil.move((os.path.join(path, k)), (os.path.join(path, 'blurry', k)))