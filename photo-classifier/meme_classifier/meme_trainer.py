from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from meme_classifier import memes_Xy
from meme_data import load_data
from meme_data import preprocess_data

import numpy as np

import os

memes_path_2 = os.path.join(os.getcwd(), 'data', 'valid', 'valid')

nonmemes_path = os.path.join(os.getcwd(), 'data', 'InstaNY100K', 'img_resized', 'newyork')

memes = load_data(path = memes_path_2, how = 'many', n_img = 4500)

not_memes = load_data(nonmemes_path, n_img = 4500)

X_train, X_test, y_train, y_test = memes_Xy(not_memes, memes)

X_train_scaled = preprocess_data(X_train)

X_test_scaled = preprocess_data(X_test)

new_random_train_indices = np.random.permutation(len(X_train_scaled))
X_train_reshuffled = X_train_scaled[new_random_train_indices]
y_train_reshuffled = y_train[new_random_train_indices]

def initialize_meme_classifier():
    model = Sequential()
    # layer 1
    model.add(layers.Conv2D(16, (8,8), input_shape=(100,100,1), 
                            padding='same', activation='relu'))
    model.add(layers.Conv2D(16, (4,4), input_shape=(100,100,1), 
                            padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    # layer 2
    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    # layer 3
    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    # layer 4
    model.add(layers.Conv2D(32, (2,2), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    # flatten layer
    model.add(layers.Flatten())
    # Dense layer
    model.add(layers.Dense(10, activation='relu'))
    # last layer
    model.add(layers.Dense(1, activation='sigmoid'))
    # compile
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                 )
    return model   

model = initialize_meme_classifier()

es = EarlyStopping(patience = 5, restore_best_weights= True)

history = model.fit(X_train_reshuffled, y_train_reshuffled, validation_split = 0.2,
          epochs=200,  
          batch_size=16, 
          verbose=1,
         callbacks = [es])

model.evaluate(X_test_scaled, y_test)

def save_model(model):
        """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
        HINTS : use joblib library and google-cloud-storage"""

        # saving the trained model to disk is mandatory to then beeing able to upload it to storage
        # Implement here
        model.save('meme_classifier')
        print("saved model.joblib locally")

        # Implement here
        #client = storage.Client()

        #bucket = client.bucket(BUCKET_NAME)

        #blob = bucket.blob(STORAGE_LOCATION)

        #blob.upload_from_filename('model.joblib')
        #print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")

save_model(model)


