## Writeup Template

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.jpg "Undistorted"
[image2]: ./output_images/distortion_correction.jpg "Road Transformed"
[image3]: ./output_images/threshold_output.jpg "Binary Example"
[image4]: ./output_images/perspective_transform_output.jpg "Warp Example"
[image5]: ./output_images/sliding_windows.jpg "Sliding windows"
[image6]: ./output_images/output.jpg "Pipeline output"
[video1]: ./project_video_out.mp4 "Video"


### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first two code cells of the IPython notebook located in "./P4.ipynb" 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
![alt text][image1]
At the end of this step camera matrix and distortion coefficients are stored in the cal_pickle.p file for later use.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The code for this step is contained in the 3rd code cell of the IPython notebook located in "./P4.ipynb". At the begining of this step data like camera matrix and distortion coefficients are restored from cal_pickle.p file. cv2.undistort function is then used with this data to apply distortion correction onto each image. Here are results for the tests files provided with the project:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for this step is contained in the 6th code cell of the IPython notebook located in "./P4.ipynb". I used a combination of gradient and channel thresholds to generate a binary image. For gradient threshold I used Sobel x operator with sx_thresh=(90, 150), which helped to detect most of the white lines. Then I used S channel threshold from HLS color space with s_thresh=(170, 255) to detect yellow lines. On top of that I added V channel threshold from HSV color space with v_thresh=(220,255) as it appeard to be useful to magnitude some of the dashed lines.  Here are examples of my output for this step. 

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a functions called `warp_img()` and `warp_road_img()` , which appear in 8th code cell of the Ipython notebook located in ".P4.ipynb".  The `warp_img()` function takes as inputs an image (`img`), source (`src_coords`) and destination (`dst_coords`) points as well as (`matrix_transform`) flag. The latter is to control which operation we are triggering, wheather it is perspective transform or inverse perspective transform.  The `warp_road_img()` defines source and destination points in the following manner:


```python
   src = np.float32([[214,720], [580,460], [704,460], [1086,720]])
    offset = [50,0]
    
    _top_left=np.array([src[0,0],0])
    _top_right=np.array([src[3,0],0])
    
    dst = np.float32([src[0]+offset, _top_left+offset, _top_right-offset, src[3]-offset])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 214, 720      | 264, 720      | 
| 580, 460      | 264, 0        |
| 704, 460      | 1036, 0       |
| 1086, 720     | 1036, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. Here are examples of my output for this step:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for this step is contained in the 10th code cell of the IPython notebook located in "./P4.ipynb" 

After calibration, thresholding and a perspective transform applied to road image, we have a binary image where the lane lines stand out clearly. What we can do with that to identify lane-line pixes is as follows: we can take a histogram of the bottom half of the image then find the peak of the left and right halves. The peaks are our starting points for where to search for the lines. From that point I used sliding window to find and follow the lines. Here are example of my output for this step:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for this step is contained in the 14th code cell of the IPython notebook located in "./P4.ipynb"

According to the formula of radius of curvature for second order polynomial provided in the lecture we need first and second derivatives to be calculated. We then put the values directly to the formula. 

In addition, to calculate distance from the center we assume that the center of the camera is exactly in the middle of the image, then we calculate x-positions (i.e. 'left_pix_base', 'right_pix_base') for both left and right lines/fits for the given y to be the bottom of the picture. We then calculate mean avarage and rescale it by meter per pixels in x dimention. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in 14th code cells.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the problem I faced when developing the project was to correctly filter out false lane lines. My initial approach for that was to play with gradient and color/channel thresholding and tune values of thresholds in such a way that all test images together would work with the same settings/thresholds. This approach, however, will not cover all the situation on the road. The good example here would be different surface illumination or driving at night. The possible ways how to overcome that would be to use deep Learning techniques to determine drivable area
