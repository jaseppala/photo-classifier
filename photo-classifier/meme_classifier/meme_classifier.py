import cv2
import numpy as np
from meme_data import load_data, get_data_gcp
from tensorflow import keras
import os

#memes_path_1 = os.path.join(os.getcwd(), 'data', 'train', 'train')
#memes_path_2 = os.path.join(os.getcwd(), 'data', 'valid', 'valid')

#nonmemes_path = os.path.join(os.getcwd(), 'data', 'InstaNY100K', 'img_resized', 'newyork')

def memes_Xy(not_memes, memes):
    X_train = np.concatenate((not_memes[:4000], memes[:4000]))
    X_test = np.concatenate((not_memes[4000:], memes[4000:]))
    
    y1 = np.array([[0] for x in range(4500)])
    y2 = np.array([[1] for x in range(4500)])

    y_train = np.concatenate((y1[:4000], y2[:4000]))
    y_test = np.concatenate((y1[4000:], y2[4000:]))

    return X_train, X_test, y_train, y_test

#get_data_gcp('InstaNY100K.zip')

if __name__ == '__main__':  


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
    



