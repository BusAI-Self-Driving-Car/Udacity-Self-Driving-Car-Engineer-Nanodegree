# **Finding Lane Lines on the Road**

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on the work in a written report


[//]: # (Image References)

[imageOriginal]: ./test_images_intermediate/Original.png "Original"
[imageGrayscale]: ./test_images_intermediate/Gray.png "Grayscale"
[imageCanny]: ./test_images_intermediate/Canny.png "Canny edges"
[imageHough]: ./test_images_intermediate/Hough.png "Hough lines"
[imageBlended]: ./test_images_intermediate/Blended.png "Blended image"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of the following steps:
* Read in an image (or a frame from a video)

![alt text][imageOriginal]

* Convert the color image to gray-scale. This summarizes the intensities of all three
color channels to a single grayscale intensity channel, making it easier to apply
subsequent processing steps.

![alt text][imageGrayscale]

* Apply Canny-edge detection based on finding the peaks of the image gradient. A
cruder way encountered in earlier lessons was to do intensity thresholding for
extracting the lane-lines. But this works only for very ideal road conditions -- very
good contrast between asphalt and lane-lines.

* Applying a Gaussian blur to the image before applying Canny-edge detection helps
prevent the detection of false positives due to image noise.

* Apply a region-of-interest mask to narrow down the set of edges detected in the
image to potential lane-lines on the road.

![alt text][imageCanny]

* Extract hough-lines from the edges. Hough-method is a robust way of extracting lines
from potentially broken Canny edges. It transforms the edge pixels to Hough parameter
space and determines a line in the original space based on intersections in the Hough
parameter-space. We can explicitly set a min. threshold on the no. of sine curves that
should intersect in the Hough grid-cell for the corresponding pixels to be considered
collinear.

![alt text][imageHough]

* In order to draw a single line on the left and right lanes, in the **draw_lines()**
function, I first separated the left and the right line segments based on their slopes
(negative slope for the left lane segments and positive for the right ones). Here, one
has to remember that the origin of the Image is considered to be at the top-left
corner, not at the bottom-left as we are usually accustomed to. To determine a single
slope and y-offset (intercept) for the lane-lines, I used the median() operation on the
slopes and offsets of the left and right lane sets. One could also use the mean, but I
found median to be more robust. This is because median chooses values in the
value-middle of the set and does not get affected by severe outliers in either
direction. Mean() could get biased by outliers. Perhaps there are also other ways of
more robustly finding a single slope/offset pair, that I could not try due to time-
limitations.

Extracted lane-lines blended together with the original image:

![alt text][imageBlended]

### 2. Identify potential shortcomings with your current pipeline
The pipeline I described above may not always work in several situations:
* If there is low contrast between the asphalt and the lane-markings due to:
  * cloudy weather, water on the road, shadows from e.g., trees or long trucks
* If there are false positives due to Asphalt repair lines
* If we see extreme curves, where it is difficult to extract long-enough straight line
 segments
* If there is water or snow on the windshield, and the camera does not have a  clear
view of the road.

### 3. Suggest possible improvements to your pipeline
* A possible improvement would be to use a contrast enhancement algorithm before
applying edge detection to make the pipeline more robust.
* Detect that we may be in an extreme curve and adapt the hough line parameters to
extract shorter line segments.
* In a curve, use the curvilinear motion of the car to predict the shape of lane-lines.

### 4. Real-life issues not considered in the above problem
* If not mounted robustly enough, the camera may lose extrinsic calibration w.r.t the
car co-ordinate system.
* The above problem tackles a situation where we are already on the road and see the
lane-markings in the region--of-interest we define. The above solution does not help
when that is not the case, e.g. while we are driving from the side of the road on to
the road.
