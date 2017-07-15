**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center.jpg "Center camera (big)"
[image3]: ./examples/recovery1.jpg "Recovery Image"
[image4]: ./examples/recovery2.jpg "Recovery Image"
[image5]: ./examples/recovery3.jpg "Recovery Image"
[image6]: ./examples/normal.jpg "Normal Image"
[image7]: ./examples/flip.jpg "Flipped Image"
[image8]: ./examples/left_multicamera.jpg "Left camera"
[image9]: ./examples/center_multicamera.jpg "Center camera"
[image10]: ./examples/right_multicamera.jpg "Right camera"
[image11]: ./examples/revert.jpg "Revert driving"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

PYTHON scripts:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* video.py for making video from IMG dictionary

MODELS: 
* model.h5 containing a trained convolution neural network with data of 3 laps of central driving, revert and recovery driving with shuffling in the generator 
*	model.h5_gen_all_no_shuffling containing a trained convolution neural network with data of 3 laps of central driving, revert and recovery driving with NO shuffling in the generator 
* model.h5_gen_3laps_no_shuffling containing a trained convolution neural network with data of 3 laps of central driving with NO shuffling in the generator
* model.h5_track_1_adv_3laps containing a trained convolution neural network with data of	3 laps of center driving, revert and recovery driving with NO generator
*	model.h5_track_1_adv_recover 	containing a trained convolution neural network with data of recovery driving with NO generator
* model.h5_track_1_adv_revert 	containing a trained convolution neural network with data of revert driving with NO generator

VIDEOS: 
*	video.mp4 - video corresponding to model.h5
* run5_gen_3laps_revert_recovery.mp4 - video corresponding to model.h5_gen_all_no_shuffling
*	run6_gen_3laps_no_shuffling.mp4 - video corresponding to model.h5_gen_3laps_no_shuffling
*	model.h5_track_1_adv_3laps.mp4 - video corresponding to model.h5_track_1_adv_3laps

REPORTS:
* writeup.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with three 5x5 and two 3x3 filter sizes and depths between 24 and 64 (model.py lines 125-129) followed by flatten layer and four fully connected layers. The three first convolutions have a stride of 2x2. Between fully connected layers dropout layers have been added to reduce overfitting. Each convolution layer ends by RELU layer to introduce nonlinearity (code lines 125-129). The data is normalized in the model using a Keras lambda layer (code line 117). The data is cropped to focus on significant part of the image, meaning road and not surroudings like trees, lakes etc (code line 119).  

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 132, 134, 136). 
The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 58-59, 105-106, 147). 
The model was trained and validated on extended data sets like views from additional cameras (code lines 83-84), horizontal flip images (code lines 13-15) to help with left turn bias and corresponding opposite sign of the steering measurements (code lines 25-27).
The model was trained and validated with date from multiple runs, recovery driving like steering from edge to the middle or revert clock-wise driving (code lines 42-55) 
The model takes advatage of the generator (code lines 67-106) producing 6x as many samples as input batch size, efectivelly giving 120 samples per batch (batch size * number of generated images). 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. The appropriate videos have been attached to visualize some of the steps done on the way, see paragraph #1. 
NOTE. The model generator uses shuffling method over whole samples set at the start of the generation and shuffling method over output batch samples at the end. However it appeard that this method doesn't work well enough for the training. My first guess is that the model doesn't work better on shuffled data because such data has no longer information about consecutive frames from videos. It may however generize better. This is something that needs to be anylize more deeply.     

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 142).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and clock-wise (revert) driving.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to collect as much data as possible by driving a car in the simulator, build a model, train and validate it and the verify results on simulator. Then if necessary augment data, use effective techniques to remove possible biases like left turn bias, non-central driving and similar.   

My first step was to use a convolution neural network model similar to the Nvidia model I thought this model might be appropriate because it was mentioned in the lecture as one of the possible choises to start with. I believe any convolution-based topology would work well, at least after approprite tuning of the hyper parameters or by adding additional convolutional layers to extract more features or increasing depths of the existing ones and adding dropout layer to reduce overfitting.  

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it takes advantage of dropout layers between all fully connected layers for classification. 

Then I trained and validated the model on different data sets from multiple runs of center line driving, steering back to center or other direction driving. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track like different surface on the bridge or on the sharp bends or other places where not enough data was provided. To improve the driving behavior in these cases I had to provide more data and crop the image to focus only on significat part of the image and not on the irrelevent ones like trees, rocks, rivers etc. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture


The final model architecture (model.py lines 115-138) consis of a convolution neural network with the following layers and layer sizes: 

Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 158, 24)   1824        cropping2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       convolution2d_4[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           convolution2d_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           211300      flatten_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 100)           0           dense_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_1[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 50)            0           dense_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dropout_2[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 10)            0           dense_3[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dropout_3[0][0]
====================================================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:
![alt text][image2]

I also took advantage of multicamera views being recorded for wich I applied 0.2 constant correction for the steering angle. Here are examples of the left, center and right imagas:
![alt text][image8]
![alt text][image9]
![alt text][image10]

With that data set prepared I trained the model I got the following results: 
Train on 21835 samples, validate on 5459 samples
Epoch 1/4
21835/21835 [==============================] - 112s - loss: 0.0333 - val_loss: 0.0213
Epoch 2/4
21835/21835 [==============================] - 113s - loss: 0.0294 - val_loss: 0.0192
Epoch 3/4
21835/21835 [==============================] - 114s - loss: 0.0278 - val_loss: 0.0200
Epoch 4/4
21835/21835 [==============================] - 114s - loss: 0.0265 - val_loss: 0.0194


I then recorded the vehicle going revert direction. Here is an example: 
![alt text][image11]
Here are the results from model training on 3 laps of center driving and one lap of revert driving:
Train on 29030 samples, validate on 7258 samples
Epoch 1/4
29030/29030 [==============================] - 150s - loss: 0.0327 - val_loss: 0.0350
Epoch 2/4
29030/29030 [==============================] - 151s - loss: 0.0284 - val_loss: 0.0366
Epoch 3/4
29030/29030 [==============================] - 151s - loss: 0.0270 - val_loss: 0.0346
Epoch 4/4
29030/29030 [==============================] - 152s - loss: 0.0258 - val_loss: 0.0355

To my surprise after adding revert road driving the car started to run faster and kind of recklessly.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to how to steer back to the center (normal way of driging). These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]
![alt text][image5]

And here is the output from the model driving. 
Train on 50448 samples, validate on 12612 samples
Epoch 1/4
50448/50448 [==============================] - 261s - loss: 0.0463 - val_loss: 0.0739
Epoch 2/4
50448/50448 [==============================] - 262s - loss: 0.0423 - val_loss: 0.0702
Epoch 3/4
50448/50448 [==============================] - 262s - loss: 0.0404 - val_loss: 0.0734
Epoch 4/4
50448/50448 [==============================] - 262s - loss: 0.0384 - val_loss: 0.0718

In the next step I decided to add dropout layers between all fully connected layers with the connection loss to be 50%. 
Though, that didn't help a lot as you can see below:
Train on 50448 samples, validate on 12612 samples
Epoch 1/4
50448/50448 [==============================] - 260s - loss: 0.0526 - val_loss: 0.0684
Epoch 2/4
50448/50448 [==============================] - 263s - loss: 0.0471 - val_loss: 0.0700
Epoch 3/4
50448/50448 [==============================] - 262s - loss: 0.0452 - val_loss: 0.0700
Epoch 4/4
50448/50448 [==============================] - 263s - loss: 0.0436 - val_loss: 0.0711

To augment the data sat, I also flipped images and angles thinking that this would help in generalization. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 8408 number of data points. I then preprocessed this data by generator to produce additional data achieving eventually 6x more, meaning 50448 images and corresponding streering angle. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2. I used an adam optimizer so that manually training the learning rate wasn't necessary.
