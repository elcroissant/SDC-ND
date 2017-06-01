# **Finding Lane Lines on the Road** 

## Writeup Template

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I applied Gaussian smoothing, then I used Canny function to detect edges, masked out irrelevent parts of the image, applied Hough transform to detect line segments and converted them into two single lines. Finally, I produced an image which consist of the orginal image with those two aforementioned lines drew on it.  

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by adding two new helper functions, one for major segments identification and one for extending segment to line. Identification of major segments was done based on the lenght between segments points. There were two such segments, on for negative slope (right line) and the other one for positive slope (left line).    

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when detecting line segment had different slop than targeted road line. As there is no validation of the slope, but checking only weather it is negative or positive, the solution is not guarded against artefacts on the roads which resemble road lines such as horizontal long lines.    

Another shortcoming could be when recognizing not existed lane lines. In that case, the code would choose two longest detected segments, one with negative and one with positive slope, without asking if they are valid.   

a few more, like behaving on crossroads, wroking at night, hadling road narrows and many more...

### 3. Suggest possible improvements to your pipeline

A possible improvement would be to add helper function to validate the range of the slope, like for instance we typically don't expect horizontal lines on the road. I will try to add that today to check if it helps for the optional video. 

Another potential improvement could be to extend draw function so that it would fit the real curve of the lane lines.
