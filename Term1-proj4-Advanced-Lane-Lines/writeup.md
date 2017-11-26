## Writeup 

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

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

*Rather than cross-referencing line numbers in the code, I will mention functions which implement a specific functionality. Code can change rapidly even after project submission, rendering any line-numbers mentioned in this README invalid.*

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file `camera_calibration.py`.

Here we determine the camera matrix, which captures the transformation between real-world 3D coordinates of objects and their corresponding 2D image coordinates. For this purpose we use a checkerboard as our object, since it has a simple pattern with good contrast and known dimensions. We use the internal corners of the checkerboard to determine the 3D world and 2D image coordinates. 

Since the checkerboard is flat, we can set its z-coordinates to 0, and for every checkerboard image, we can assume the same real-world checkerboard object with the same (x, y, z) coordinates. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time all checkerboard corners are detected in a test image.  

`imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful checkerboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to a test image using the `cv2.undistort()` function and obtained the following result.

[imageUndistortCheckerboard]: ./output_images/undistort_checkerboard.png "Undistorted checkerboard"
![alt text][imageUndistortCheckerboard]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Once the camera calibration is available from the previous step, it can be used to undistort real-world test images using the `cv2.undistort()` function.

Notice how the deer-warning road-sign appears flatter in the undistorted image:

[imageUndistortRoadImage]: ./output_images/undistort_road_img.png "Undistorted road image"
![alt text][imageUndistortRoadImage]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

 I used a combination of gradient (x-direction) and HLS colorspace thresholding on the S-channel. See `binarize_frame()` in `image_binarization.py` and the illustrations in the following images. 

 I used the sobel operator to calculate the x- or y-gradients (opencv function `cv2.Sobel()`). The magnitude and direction gradients were derived from the x- and y-gradients. After experimenting somewhat with these gradients and their combinations, I concluded that just the x-gradient was sufficient to achieve a good performance with the project-video. Since the lane-lines are more or less vertical in the camera images, the x-gradient captures them most clearly. 

With hints from the project guidelines and after experimenting with the HLS colorspace, I found that the S (saturation)-channel captured the lane-lines quite well. It was independent of lane-line color (yellow/white) and pretty robust against contrast changes on the road-surface, e.g. due to shadows.

My final image binarization routine uses a combination of x-gradient and S-channel thresholding. This combined thresholding is illustrated in the images below:

[image1XGradientThreshold]: ./output_images/1-x-gradient-threshold.png "Undistorted road image"
![alt text][image1XGradientThreshold]

[image2SThreshold]: ./output_images/2-S-threshold.png "Undistorted road image"
![alt text][image2SThreshold]

[image3CombinedGradientSThreshold]: ./output_images/3-combined-gradient-S-threshold.png "Undistorted road image"
![alt text][image3CombinedGradientSThreshold]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes the functions `get_src_dst_vertices()`, `get_perspective_transform()`, and `warp_image_to_top_down_view()` in file `perspective_transformation.py`. 

The source and destination points required for determining the perspective transformation are obtained from the `get_src_dst_vertices()`, where after many trials, they were hardcoded to:

```python
src = np.float32([[567, 470],[717, 470],[1110, 720],[200, 720]])

offset, mult = 100, 3    
dst = np.float32([[mult * offset, offset],
                 [img_size[0] - mult * offset, offset],
                 [img_size[0] - mult * offset, img_size[1]],
                 [mult * offset, img_size[1]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 567, 470      | 300, 100      | 
| 717, 470      | 980, 100      |
| 1110, 720     | 980, 720      |
| 200, 720      | 300, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

[image1PerspectiveTransformStraight]: ./output_images/1-perspective-transform-straight.png "Perspective transform for straight lane lines"
![alt text][image1PerspectiveTransformStraight]

Here is the perspective transform applied to curved lane-lines:

[image2PerspectiveTransformCurved]: ./output_images/2-perspective-transform-curved.png "Perspective transform for curved lane lines"
![alt text][image2PerspectiveTransformCurved]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The input to this stage is a binary image with a top-down (bird's eye) view. 

For "detecting" lane-lines for the first time (see `detect_lane_lines()` in `lane_lines.py`): I determine x-coordinates in the image that most likely coincide with the left and the right lane lines, by looking at the peaks of the histogram taken along the x-axis at the bottom of the image. Then I build rectangular search windows around these x-coordinates and retrieve the pixel positions for the lane-line pixels (See `get_lane_indices()` and `get_lane_pixel_positions()` in `lane_lines.py`). With the x and y pixel positions thus determined, a second-order polynomial fit is determined for the lane-lines using the function `Line.update_line_fit()` in `lane_lines.py`.

An example of such a polynomial-fit (Ax^2 + Bx + C), with the rectangular search windows:

```python
left_fit: [  5.93193079e-06   2.29126109e-02   2.94748692e+02]
right_fit: [ -4.70246069e-06  -2.74283879e-02   9.98546569e+02]
```

[imageSecondOrderPolyfitDetection]: ./output_images/second-order-polyfit-detection.png "Second order polynomial fit on detected lane-lines"
![alt text][imageSecondOrderPolyfitDetection]

Once the lane-lines are detected at the beginning, I track them over the subsequent frames by specifying a search window around the polynomial fit determined previously (see `track_lane_lines()` in `lane_lines.py`). This saves us an exhaustive search from bottom-to-top of the image as required during the line detection described above. 

In fact for tracking lines across frames, I don't use a single previous fit, rather an average over some previous fits to place the search window. Also, for plotting the lane-line on the output image, I use the average fit which includes the current fit. This low-pass filtering is done in the hope that the wobbliness or flutteriness of lines is reduced. Here is an example image of tracked lane-lines:

[imageSecondOrderPolyfitTracking]: ./output_images/second-order-polyfit-tracking.png "Second order polynomial fit on tracked lane-lines"
![alt text][imageSecondOrderPolyfitTracking]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature of the lane and the offset of the car w.r.t. the lane center are calculated in the functions `write_curvature_text_to_image()` and `write_lane_offset_text_to_image()`, respectively, in `lane_lines.py`.

**Radius of curvature** The x and y coordinates of the lane-lines are translated to their metric values using appropriate scaling factors provided in the project guidelines. Then a second-order polynomial fit is calculated on these metric values. The radius of curvature is then determined as described at https://www.intmath.com/applications-differentiation/8-radius-curvature.php. Finally, I average the radius for the left and right lines to determine the radius of curvature for the lane.

**Car offset w.r.t. lane-center** This value is determined by taking the difference between the x-coordinates of the midpoint of the determined lane-lines and the center of the image. This assumes that the camera is mounted exactly along the center-axis of the car. 

```python
lane_mid_x = x_left + (x_right - x_left)/2
offset = x_meter_per_pixel * (img_mid_x - lane_mid_x)
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This is implemented by the function `project_lane_lines_to_road()` in `lane_lines.py`. Here is an example of my result on a test image:

[imageLaneIdentified]: ./output_images/lane-identified.png "Lane identified"
![alt text][imageLaneIdentified]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./out_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As can be seen from the images below, the image-binarization captures a lot of confounding details, which may turn out badly for the line-fitting. Perhaps a Convolutional Neural Network would be more robust here, picking up its own features relevant to the problem and ignoring non-essential details. 

In this image, the shadows on the road show up heavily in the binarized image:

[imageProblem1]: ./output_images/problem1.png "Problem1"
![alt text][imageProblem1]

Also, my frame-to-frame line-tracking seems to fail for sharp curves in the challenge videos. This is because the search window predicted from previous frames excludes sharply curving lane-lines deeper in the image. A solution to this could be to use the radius of curvature to correct the predicted search window. For a sharp curve, this could help us to "bend" the search window, so that the curved lane-lines are captured deeper into the image. 

From the harder challenge video:

[imageProblem2]: ./output_images/problem2.png "Problem2"
![alt text][imageProblem2]

We can visualize the output at various stages of the pipeline with test images. However, working with a video where we track lines across frames is a bit different, and cannot be visualized easily. In the future, I would like to implement a Picture-in-Picture inset to the output video to visualize, e.g. how well the gradient and color thresholding are performing. This would make it easier to tune the image-binarization algorithm, while looking at the output video simutaneously. 