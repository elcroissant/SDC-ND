import csv
import cv2
import numpy as np

lines = []

with open('../../../data_track_1_320x240/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = [] 
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('\\')[-1]
    current_path = '../../../data_track_1_320x240/IMG/' + filename
    image = cv2.imread(current_path)
    image_flipped = np.fliplr(image)
    images.append(image)
    images.append(image_flipped)
    measurement = float(line[3])
    measurement_flipped = -measurement
    measurements.append(measurement)
    measurements.append(measurement_flipped)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
model = Sequential()
model.add(Lambda( lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6, 5, 5, border_mode='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), border_mode='valid'))
model.add(Convolution2D(16, 5, 5, border_mode='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), border_mode='valid'))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)
model.save('model.h5')

    
