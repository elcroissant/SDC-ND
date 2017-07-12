import csv
import cv2
import numpy as np

car_images = []
steering_angles = []

def car_images_extend(img_center, img_left, imge_right):
    car_images.append(img_center)
    car_images.append(img_left)
    car_images.append(img_right)
    car_images.append(np.fliplr(img_center))
    car_images.append(np.fliplr(img_left))
    car_images.append(np.fliplr(img_right))


def steering_angles_extend(steering_center, steering_left, steering_right):
    steering_angles.append(steering_center)
    steering_angles.append(steering_left)
    steering_angles.append(steering_right)
    steering_angles.append(-steering_center)
    steering_angles.append(-steering_left)
    steering_angles.append(-steering_right)


with open("../../../data_track_1_320x240/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        steering_center = float(row[3])

        # create adjusted steering measurements for the side camera images
        correction = 0.2 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        path = "../../../data_track_1_320x240/IMG/" # fill in the path to your training IMG directory
        
        img_center = cv2.imread(path + row[0].split('\\')[-1])
        img_left = cv2.imread(path + row[1].split('\\')[-1])
        img_right = cv2.imread(path + row[2].split('\\')[-1])
	
        # add images and angles to data set
        #car_images.extend(img_center, img_left, img_right)
        car_images_extend(img_center, img_left, img_right)
        #steering_angles.extend(steering_center, steering_left, steering_right)
        steering_angles_extend(steering_center, steering_left, steering_right)


X_train = np.array(car_images)
y_train = np.array(steering_angles)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

model = Sequential()
model.add(Lambda( lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)
model.save('model.h5')

    
