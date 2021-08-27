import cv2
import numpy as np
from data import load_data, get_data_gcp, get_image_dict
from tensorflow.keras import models
import os
import shutil

if __name__ == '__main__':

    model = models.load_model('cnn_screenshots.h5')

    path = os.path.join('..', 'raw_data', 'test')

    img_dict = {file:0 for file in os.listdir(path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))}

    for file in os.listdir(path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):                                    # get every file in the folder
            img = cv2.imread(os.path.join(path, file))                  # load the image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            res = cv2.resize(gray, dsize=(150, 150))
            res = res / 255.
            res = np.expand_dims(res, axis = -1)             # make it RGB (cv2 uses BGR)
            img_dict[file] = res

    screenshots_dump = os.path.join(path,'screenshots')

    if not os.path.exists(screenshots_dump):
        os.mkdir(screenshots_dump)

    #print(img_dict.keys())

    for k in img_dict.keys():
        pred = model.predict(np.expand_dims(img_dict[k], axis=0))
        if pred > 0.5:
            shutil.move((os.path.join(path, k)), (os.path.join(screenshots_dump, k)))














    #memes1 = load_data(path = memes_path_1, how = 'many', asarray = False)
    #print(len(memes1))
    #memes = load_data(path = memes_path_2, how = 'many', n_img = 4500)
    #print(len(memes2))
    #memes = memes1.extend(memes2)
    #print(len(memes))

    #memes = np.array(memes)
    #memes

# not_memes = load_data(nonmemes_path, n_img = 4500)

#print(memes.shape)
#print(not_memes.shape)
