from google.cloud import storage
from zipfile import ZipFile

import os
from os.path import isdir
import cv2
import numpy as np

GCP_FILE_NAME = 'memes_dataset.zip'

def get_data_gcp(GCP_FILE_NAME):
    client = storage.Client(project = 'le-wagon-ds-bootcamp-318909')
    bucket = client.get_bucket('lewagon-photo-classifier')
    blob = bucket.get_blob(GCP_FILE_NAME)
    blob.download_to_filename(os.path.join(os.getcwd(), GCP_FILE_NAME))
    
    with ZipFile(os.path.join(os.getcwd(), GCP_FILE_NAME), 'r') as zipObj:
        zipObj.extractall('data')

def load_data(path, how = 'one', grayscale = True, asarray = True, n_img = 'all'):
    """loads all images into an array

    path: path to the folder in which the images or folders full of images are
    how: 'one' to load files from one folder, 'many' if images are in subfolders (default 'one')
    grayscale: pictures are stored in grayscale if True, in RGB if False (default 'True')
    asarray: return a np.array. Returns a list if False (default 'True')
    n_img: number of images to be loaded from the path (all by default)

    Returns:
        np.array of n_img pictures
        or list of n_img pictures if asarray = False
    """    
    X = []

    if type(n_img) == int:
        i = 0

    if how == 'one':
        for file in os.listdir(path):                              # get every file in the folder
            img = cv2.imread(os.path.join(path, file))                  # load the image
            if grayscale:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                res = cv2.resize(gray, dsize=(100, 100))             # make it RGB (cv2 uses BGR)
                X.append(res)
            else:
                clr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X.append(clr)
            if type(n_img) == int:
                    i += 1
                    if i == n_img:
                        if asarray:
                            return np.array(X)
                        else:
                            return X

    if how == 'many':
        for folder in os.listdir(path):
            if isdir(os.path.join(path, folder)): 
                for file in os.listdir(os.path.join(path, folder)):                              # get every file in the folder
                    img = cv2.imread(os.path.join(path, folder, file))                  # load the image
                    if grayscale:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)             # make it RGB (cv2 uses BGR)
                        res = cv2.resize(gray, dsize=(100, 100)) 
                        X.append(res)
                    else:
                        clr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        X.append(img)
                    
                    if type(n_img) == int:
                        i += 1
                        if i == n_img:
                            if asarray:
                                return np.array(X)
                            else:
                                return X

    if asarray:
        return np.array(X)
    else:
        return X

    
def preprocess_data(X):
    X = X / 255
    X = np.expand_dims(X, axis = -1)
    return X


