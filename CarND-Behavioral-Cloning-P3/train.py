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

    
