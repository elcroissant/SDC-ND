import csv
import cv2
import numpy as np


class CarInfo:
    def __init__(self):
        self.car_images = []
        self.steering_angles = []

    def car_images_extend(self, img_center, img_left, img_right):
        self.car_images.append(img_center)
        self.car_images.append(img_left)
        self.car_images.append(img_right)
        self.car_images.append(np.fliplr(img_center))
        self.car_images.append(np.fliplr(img_left))
        self.car_images.append(np.fliplr(img_right))
    
    def steering_angles_extend(self, steering_center, steering_left, steering_right):
        self.steering_angles.append(steering_center)
        self.steering_angles.append(steering_left)
        self.steering_angles.append(steering_right)
        self.steering_angles.append(-steering_center)
        self.steering_angles.append(-steering_left)
        self.steering_angles.append(-steering_right)
    

def collect_data(csv_path, img_path):
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            steering_center = float(row[3])
    
            # create adjusted steering measurements for the side camera images
            correction = 0.2 # this is a parameter to tune
            steering_left = steering_center + correction
            steering_right = steering_center - correction
    
            # read in images from center, left and right cameras
            path = img_path # fill in the path to your training IMG directory
            
            img_center = cv2.imread(path + row[0].split('\\')[-1])
            img_left = cv2.imread(path + row[1].split('\\')[-1])
            img_right = cv2.imread(path + row[2].split('\\')[-1])
    
            # add images and angles to data set
            car_info.car_images_extend(img_center, img_left, img_right)
            car_info.steering_angles_extend(steering_center, steering_left, steering_right)

car_info = CarInfo()	

collect_data("../../data_track_1_adv/driving_log.csv", "../../data_track_1_adv/IMG/")
collect_data("../../data_track_1_adv_revert/driving_log.csv", "../../data_track_1_adv_revert/IMG/")
collect_data("../../data_track_1_adv_recovery/driving_log.csv", "../../data_track_1_adv_recovery/IMG/")



X_train = np.array(car_info.car_images)
y_train = np.array(car_info.steering_angles)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
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
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)
model.save('model.h5')

    
