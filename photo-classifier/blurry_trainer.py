from data import get_data_gcp, load_data

import os
import numpy as np

from tensorflow.keras import layers, Sequential, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras

#get_data_gcp('blurry_sharp_dataset.zip')

path = '../raw_data'

blurry = load_data(os.path.join(path, 'defocused_blurred'))

sharp = load_data(os.path.join(path, 'sharp'))

#print(blurry.shape, sharp.shape)

def create_blurry_pipe():

    model_pipe = Sequential([
    layers.Reshape((100, 100, 1), input_shape=(100, 100)),
    layers.experimental.preprocessing.Rescaling(scale=1./255.),
    layers.Conv2D(32, (3,3), padding='same', activation="relu"),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Conv2D(64, (3,3), padding='same', activation="relu"),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPool2D(4,4),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(16,activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
    ])

    model_pipe.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                 )
    return model_pipe

pipe = create_blurry_pipe()

X_train = np.concatenate((sharp[:300], blurry[:300]))
X_test = np.concatenate((sharp[300:], blurry[300:]))

y1 = np.array([[0] for x in range(350)])
y2 = np.array([[1] for x in range(350)])

y_train = np.concatenate((y1[:300], y2[:300]))
y_test = np.concatenate((y1[300:], y2[300:]))

es = EarlyStopping(patience = 10, restore_best_weights = True)
history = pipe.fit(X_train, y_train, validation_split = 0.2,
          batch_size=32, # Too small --> no generalization. Too large --> compute slowly
          epochs=100,
          callbacks=[es],
          verbose=1
         )


pipe.save('blur_detection_model')