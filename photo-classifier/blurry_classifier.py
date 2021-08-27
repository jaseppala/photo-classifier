from tensorflow.keras import models
from data import get_image_dict
import os
import cv2
import numpy as np
import shutil

pipeline = models.load_model('blur_detection_model.h5')

path = os.path.join('..', 'raw_data', 'test')

img_dict = get_image_dict(path)

blurry_dump = os.path.join(path,'blurry')

if not os.path.exists(blurry_dump):
    os.mkdir(blurry_dump)

for k in img_dict.keys():
    pred = pipeline.predict(np.expand_dims(img_dict[k], axis = 0))
    if pred > 0.5:
        shutil.move((os.path.join(path, k)), (os.path.join(path, 'blurry', k)))