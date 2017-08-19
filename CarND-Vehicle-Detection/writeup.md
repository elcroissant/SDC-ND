
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.jpg
[image3]: ./output_images/window_search.png
[image4]: ./output_images/scale1_4_threshold5.jpg
[image5]: ./output_images/scale1_threshold0.jpg
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

---
Writeup / README

Histogram of Oriented Gradients (HOG)

1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 3rd code cell of the IPython notebook (i.e. P5.ipynb). The first example of usege can be found in the 4th cell of the IPython notebook.   

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of two of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
The corresponding code can be found in the 2nd cell of the IPython notebook (i.e. P5.ipynb)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and my final set is 

color_space = 'YUV'
- YUV appeared to be much better than RGB, HSV, LUV, HLS and gave pretty much the same results as YCrCb. Both YUV adn YCrCb seem to be good candidates. 

orient = 8  # HOG orientations 
- increased HOG orienteation parameter gave better accuracy for linear clasiffier results, though, learning time increased singnificantly, thus decided to stay with 8 orientations 

pix_per_cell = 4 # HOG pixels per cell
- decreased pix_per_cel parameter didn't cause significant degradation in detection of the correct bboxes and helped with decreasing image processing time 

cell_per_block = 1 # HOG cells per block
- decreasing cell_per_block parameter didn't cause singnificate degradation in detecting correct bboxes and helped with decreasign image processing time

hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
- processing channels separetaly didn't give acceptable results, thus decided to go with ALL 

spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
- spatially binned color and histograms of color in the feature vector made detecting black color car more robust, most likely dataset is not covering it good enough, so this is area to explore/verify with some additional datasets

I decided to use spatial, histogram and HOG featrues all together as they achieved pretty much good results for linear clasification. 

3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using 8792 examples of car images and 8968 examples of non-car images. Feature vector lenght, which contains spatial, histogram and HOG features, was 6960. It takes ~300 seconds to extract features and ~8 seconds to traing SVC. After extracting features has been normalized and then split up into randomized training and test data sets with the ratio to be 80% vs 20% appropriately. The achieved accuracy on the test set was ~98.7%. Example predition of 10 classes took around 0.003 seconds. The corresponding code can be found going through all the cells between 6th and 10th in the P5.ipynb jupyter notebook)

Sliding Window Search

1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to use hog-subsampling window search which uses number of cell to step instead of overlap. The cooresponding code can be found in the 13th cell of the P5.ipynb jupyter notebook. The simple example of how it works can be found in the 14th cell of the same jupyter notebook.

![alt text][image3]

In the example above scale equal to 1.5 has been used with the cells_per_step parameter set to 2. In order to find best scale I had to used bbox heatmap to remove false postives. For details, have a look at the cell no 15. I then made some checks with different scale and heatmap threshold. From that excercise, I learned scale equal to 1 with threshold equal to 0 works best in terms of bounding box fit, but it was very slow. Very similar results I achieved also with scale equal to 1.4 and threshold equal to 5. See pictures below.

SCALE = 1 and THRESHOLD = 0
![alt text][image4]

SCALE = 1.4 and THRESHOLD = 5
![alt text][image5]


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

