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
[image4]: ./examples/TrafficSigns/00005.PNG "Traffic Sign 1"
[image5]: ./examples/TrafficSigns/00006.PNG "Traffic Sign 2"
[image6]: ./examples/TrafficSigns/00007.png "Traffic Sign 3"
[image7]: ./examples/TrafficSigns/00008.png "Traffic Sign 4"
[image8]: ./examples/TrafficSigns/00009.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

| Criteria | Specification met | Status | 
| Submission files | The project submission includes all required files | Done | 
| Dataset Summary | The submission includes a basic summary of the data set. | Done | 
| Exploratory Visualization | The submission includes an exploratory visualization on the dataset. | Done | 
| Preprocessing | The submission describes the preprocessing techniques used and why these techniques were chosen. | Done |
| Model Architecture | The submission provides details of the characteristics and qualities of the architecture, including the type of model used, the number of layers, and the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged. | Done |
| Model Training | The submission describes how the model was trained by discussing what optimizer was used, batch size, number of epochs and values for hyperparameters. | Done |
| Solution Approach | The submission describes the approach to finding a solution. Accuracy on the validation set is 0.93 or greater. | Done | 
| Acquiring New Images | The submission includes five new German Traffic signs found on the web, and the images are visualized. Discussion is made as to particular qualities of the images or traffic signs in the images that are of interest, such as whether they would be difficult for the model to classify. | Done |
| Performance on New Images | The submission documents the performance of the model when tested on the captured images. The performance on the new images is compared to the accuracy results of the test set. | Done |
| Model Certainty - Softmax Probabilities | The top five softmax probabilities of the predictions on the captured images are outputted. The submission discusses how certain or uncertain the model is of its predictions. | Done | 

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
*Validation Accuracy = 0.980 
*Test Accuracy = 0.959

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of 98% 
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
-Validation Accuracy = 0.980 
-Test Accuracy = 0.959

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image									| Prediction							| 
|:-------------------------------------:|:-------------------------------------:| 
| Right-of-way at the next intersection | Right-of-way at the next intersection | 
| Speed limit (120km/h)    				| Speed limit (20km/h) 					|
| Road work								| Road work								|
| Stop									| Stop					 				|
| Bumpy Road							| Bumpy Road      						|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 95.9%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Right-of-way at the next intersection (probability of 0.26), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Right-of-way at the next intersection 		| 
| .00     				| Priority road 								|
| .00					| Beware of ice/snow							|
| .00	      			| Double curve					 				|
| .00				    | Roundabout mandatory      					|

The model is 100% sure that the first image is Right-of-way at the next intersection. 

For the second image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .95         			| Speed limit (20km/h) 							| 
| .04     				| Children crossing 							|
| .00					| Speed limit (120km/h)							|
| .00	      			| General caution					 			|
| .00				    | Roundabout mandatory      					|

The model chose wrong sign with 95% certainty. It was pretty close to the target (visually) but failed.

For the third image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Road work 									| 
| .00     				| General caution 								|
| .00					| Dangerous curve to the right)					|
| .00	      			| Speed limit (70km/h)					 		|
| .00				    | Go straight or right      					|

The model 99.95% sure that the given traffic sign was Road work which is correct.  

    
For the forth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .87         			| Stop 											| 
| .11     				| Speed limit (60km/h) 							|
| .00					| Speed limit (50km/h)							|
| .00	      			| Turn left ahead					 			|
| .00				    | Speed limit (80km/h      						|

The model was pretty much certain that the traffic sign was a Stop, which is correct.
    
For the fifth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Bumpy road 									| 
| .00     				| Keep right 									|
| .00					| Stop											|
| .00	      			| Bicycles crossing				 				|
| .00				    | Traffic signals     							|

100% certainty for the last image. 
