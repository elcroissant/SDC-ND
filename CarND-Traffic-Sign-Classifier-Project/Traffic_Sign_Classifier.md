#**Traffic Sign Recognition** 

##Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.png "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/TrafficSigns/1.jpg "Traffic Sign 1"
[image5]: ./examples/TrafficSigns/2.jpg "Traffic Sign 2"
[image6]: ./examples/TrafficSigns/3.jpg "Traffic Sign 3"
[image7]: ./examples/TrafficSigns/4.jpg "Traffic Sign 4"
[image8]: ./examples/TrafficSigns/5.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/elcroissant/SDC-ND/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used python and numpy library to calculate summary statistics of the traffic signs data set. Below you can find results and methods used to calcucate/retrieve data.

* The size of training set is 34799 (retrieve form X_train_un.shape[0] (same can be got by len(X_train_un))
* The size of the validation set is 4410 (retrieve from X_valid_un.shape[0] (same can be got by len(X_valid_un))
* The size of test set is 12630 (retrived from X_test_un.shape[0] (same can be got by len(X_test_un))
* The shape of a traffic sign image is (32,32,3) (retrived from X_train_un.shape[1:4])
* The number of unique classes/labels in the data set is 43 (counted by np.unique(y_train).size (same can be calculated by len(set(y_train)))

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed. From that distribution we can easily read that some of the traffic signs come out much often then the others. This most likely will lead to the misclasification problem of the small classes. One of the solutions to overcome it could be to augment those less frequent classes by adding images from the provided dataset but with additional deformation like shifting, rotating, bluring and similar. That would help for the rest of the classes as well as it is always good to increase number of training examples to make test phase reach better accuracy (smaller loss function). 

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

As a first step, I decided to convert the images to grayscale to save memory and to speed up training.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a second step, I normalized the image data because the range of values of raw data can vary widely. In best case scenario we should make sure that each feature contributes proportionally to the metric classifier uses.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Basically it is LeNet-based architecture with additional layers added and convolution capacity increased.  The final model consist of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 15x15x64. 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 13x13x64	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x64. 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 4x4x64	|
| RELU					|												|
| Flatten	      	| output 1024 				|
| Fully connected		| output 120        									|
| Dropout | keep_prob: .75 |
| Fully connected		| output 84        									|
| Dropout | keep_prob: .75 |
| Fully connected		| output 43        									|
| Softmax				|         									| 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer with batch size to be 64 images and learning rate equal to 0.001. Dropout keep probability was set to 75% for the two first (out of three) fully connected layers. 
The model was trained for 30 Epochs achieving the following results:
*Train Accuracy = 1.000
*Validation Accuracy = 0.977 
*Test Accuracy = 0.959

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100
* validation set accuracy of 97.7% 
* test set accuracy of 96%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

The first architecture that was tried was LeNet and I decided to tune it iteratively and make LeNet-based one for myself. What makes me choose this architecture was relative simplicity of the architecture and similar character of the problem with MNIST. My idea was to increase capacity (number of kernels) to extract more basic features from the images and then decrease kernel size to focus on smaller parts of the image. We have much more examples compared to MNIST and more output classes, but in general the problem is very similar, especially when we convert data to grayscale. 

* What were some problems with the initial architecture?
Accuracy was the mojor problem which most likely came from capacity of the network nad overfitting. 

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

Some iterative steps done while tuning were as follows: 

#0: Increased no of Epochs to be 30
#1: Added Normalization
Validation Accuracy = 0.934
Test Accuracy = 0.913

#2: Reduced kernel size for the first convolution 5x5 => 3x3
Validation Accuracy = 0.944
Test Accuracy = 0.929

#3: Increased capacity of the first convolution 6 => 64 (no of kernels)
Validation Accuracy = 0.949
Test Accuracy = 0.941

#4: Reduced kernel size for the second convolution 5x5 => 3x3 and its capacity to the same extent as first convolution: 16=>64
Validation Accuracy = 0.964
Test Accuracy = 0.944

#5: Decreased BATCH SIZE 128=>64
Validation Accuracy = 0.964
Test Accuracy = 0.954

#6: Added Dropout to prevent overfitting and optimize the performance.
Validation Accuracy = 0.978
Test Accuracy = 0.96

* Which parameters were tuned? How were they adjusted and why?
BATCH SIZE: decreased to the level of 64 images per batch
EPOCHS: increased to 30 to allow the model for further convergence

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

- Increased number of convolution kernels to extract more features. The idea was to force the network to find additional features in the data so that classes could be better distinguished from each other.  
- Decreased kernel size to focus on smaller part of the image. Too big kernels tries to find more global pattern, but traffic signs seems to be very similar so we should focus more on local features. The provided traffic signs pictures were small in size 32x32x1 thus I decided to decrease kernel size as well.   
- Dropout added to fully connected latyers to overcome overfitting. 

If a well known architecture was chosen:
* What architecture was chosen?
LeNet

* Why did you believe it would be relevant to the traffic sign application?
I believe I would find better architecture for this problem, however I found traffic sign recognition problem to be very similar to MNIST and what comes to my mind when I think about MNIST is LeNet. 

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
If Train, Validation and Test accuracies are all high it means final model works well with the given problem.
-Train Accuracy = 1.000
-Validation Accuracy = 0.977 
-Test Accuracy = 0.959

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] 

The first image might be difficult to classify because is partly hidden and there are twigs which introduce some clutter to the picture. The second picture may also be hard to recognize as not all edges are visible on the picture. The third picture of stop sign was taken at an angle, so this also may be little tricky to classify. The forth picture may cause troble due to displacement. The last but least will cause problme for the classifier because it's partly hidden.  

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image					| Prediction							| 
|:---------------------:|:-------------------------------------:| 
| Pedestrians 			| General caution						| 
| Turn left ahead    	| No entry								|
| Stop					| Stop									|
| Stop					| Traffic signals 		 				|
| Beware of ice/snow	| Slippery road   						|


The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%. This was far below the accuracy on the test set which was 95.9%. Though, we should not expect any better predition for those images without training the model with augmented data.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.

For the first image, the model is quite sure that this is a General cautions (probability of 53%), but it is Pedestrians sign in fact which is the secend choice of the model. If we look more closesly those two signs are pretty much similar which explain the error a little bit. 

| Probability         	|     Prediction		| 
|:---------------------:|:---------------------:| 
| .53         			| General caution 		| 
| .33     				| Pedestrians 			|
| .10					| Traffic signals 		|
| .01	      			| Keep right			|
| .01				    | Speed limit (60km/h)	|


For the second image the model is almost 100% sure that the sign is No entry, but in fact the model is totally wrong, because it it's Turn left sign. Well, partly wrong if we exemine the difference of those two by looking at the pictures.  

| Probability         	|     Prediction						| 
|:---------------------:|:-------------------------------------:| 
| 1.00         			| No entry  							| 
| .00     				| General caution 						|
| .00					| Speed limit (30km/h)					|
| .00	      			| End of all speed and passing limits	|
| .00				    | Roundabout mandatory  				|

For the third image, the model is pretty much sure (with 59% certainty) this is Stop sign and that's correct.  

| Probability         	|     Prediction						| 
|:---------------------:|:-------------------------------------:| 
| .59         			| Stop  								| 
| .22     				| Road work 							|
| .13					| General caution						|
| .00	      			| Double curve					 		|
| .00				    | Turn left ahead      					|

   
For the forth image, the model is not sure at all and it is totally wrong. There is no right answer within 5 top guesses. 

| Probability         	|     Prediction	        			| 
|:---------------------:|:-------------------------------------:| 
| .22         			| Traffic signals 						| 
| .17     				| Speed limit (20km/h) 					|
| .12					| Go straight or left					|
| .12	      			| Road narrows on the right				|
| .09				    | Turn right ahead      				|

[  8.74413788e-01   9.27516222e-02   3.11537068e-02   1.53641799e-03
    7.57986418e-05]]
    [23 21 29 30 11]]

For the fifth image the model has no clue what is on the picture, however funny enough predicted Slippery road and real Beware of ice/snow are very reletad.   

| Probability         	|     Prediction	        				| 
|:---------------------:|:-----------------------------------------:| 
| .87         			| Slippery road 							| 
| .09     				| Double clurve 							|
| .03					| Bicycles crossing							|
| .00	      			| Beware of ice/snow				 		|
| .00				    | Right-of-way at the next intersection		|


