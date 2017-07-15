# class Track Info is used to store generated track info 
class TrackInfo:
    def __init__(self):
        self.car_images = []
        self.steering_angles = []

    def car_images_extend(self, img_center, img_left, img_right):
        self.car_images.append(img_center)
        # additional views from left and right cameras
        self.car_images.append(img_left)
        self.car_images.append(img_right)
        # this part is for horizontal flip to augment data
        self.car_images.append(np.fliplr(img_center))
        self.car_images.append(np.fliplr(img_left))
        self.car_images.append(np.fliplr(img_right))
    
    def steering_angles_extend(self, steering_center, steering_left, steering_right):
        self.steering_angles.append(steering_center)
        # this part corresponds to left and right cameras
        # steering left and right angles added in the generator 
        self.steering_angles.append(steering_left)
        self.steering_angles.append(steering_right)
        # this part corresponds to horizontal flip 
        # if you flip you also have to change direction of the angle you steer to
        self.steering_angles.append(-steering_center)
        self.steering_angles.append(-steering_left)
        self.steering_angles.append(-steering_right)

    def reset(self):
        self.car_images = []
        self.steering_angles = []
    
import os
import csv

# data collection for generatlization purpose
# driving_log_1.csv - data gathered during 3x laps of center line driving
# driving_log_2.csv - data gathered during reverse driving
# driving_log_3.csv - data gathered during recovery driving (steering from edge to middle)
# note, images from 3 driving excercises were all pushed to the IMG directory in the root
samples = []
with open('driving_log_1.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

with open('driving_log_2.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

with open('driving_log_3.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# samples are split between train and validation in proposion of 80 and 20 accordingly
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


import numpy as np
import cv2
import sklearn
from random import shuffle

def generator(samples, batch_size):
   # create empty arrays to contain batch of features and labels#
   track_info = TrackInfo()
   num_samples = len(samples)
   while True:
   # shuffle data to get different samples arrangment for different epochs
      shuffle (samples)
      for offset in range(0, num_samples, batch_size):
          # extract batch-size number of samples from provided samples  
          batch_samples = samples[offset:offset+batch_size]
          # reset track info every batch to operate only on the size of batch_size samples
          track_info.reset()
          for batch_sample in batch_samples:
                img_path = 'IMG/'
                # read central image and corresponding left nad right image
                center_image = cv2.imread(img_path +  batch_sample[0].split('\\')[-1])
                left_image = cv2.imread(img_path +  batch_sample[1].split('\\')[-1])
                right_image = cv2.imread(img_path +  batch_sample[2].split('\\')[-1])
                # read out given steering angle for central image 
                # and generate corrected angles for left and right images
                center_angle = float(batch_sample[3])
                correction = 0.2
                left_angle = center_angle + correction
                right_angle = center_angle - correction
                # pass data to track_info, where horizontal flip data is added to each image
                track_info.car_images_extend(center_image, left_image, right_image)
                track_info.steering_angles_extend(center_angle, left_angle, right_angle)

          # retrive data from track_info and create numpy array before passing it to yield
          X_train = np.array(track_info.car_images)
          y_train = np.array(track_info.steering_angles)

          # shuffle output data to generize 
          yield sklearn.utils.shuffle(X_train, y_train)
          #yield (X_train, y_train)

# specifying batch_size to be 20 we are telling generator to produce 20 * 6 images 
# (center, left, right, center_flip, left_flip, right_flip)
train_generator = generator(train_samples, batch_size=20)
validation_generator = generator(validation_samples, batch_size=20)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

def PrepareInitModel():
    model = Sequential()
    model.add(Lambda( lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    # trim image to only see section with road
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    return model

# this is NVidia model the three additional dropout layers to reduce overfitting
def NvidiaModel():
    model = PrepareInitModel();
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
    return model

model = NvidiaModel()

model.compile(loss='mse', optimizer='adam')

# we are augmenting data by views from left and right cameras and adding flip images for each picture
# which give us x6 as many images compared to original list
number_train_samples = len(train_samples) * 6 
model.fit_generator(train_generator, number_train_samples, validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')

    
