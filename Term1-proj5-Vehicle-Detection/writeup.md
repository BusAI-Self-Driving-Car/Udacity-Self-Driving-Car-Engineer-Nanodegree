## Writeup

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Note: for those first two steps don't forget to normalize the features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)


[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

HOG feature extraction is implemented in the function `extract_hog_features()` in `feature_extracters.py`). This function transforms the input image to the HSV colorspace and eventually calls `skimage.feature.hog()` to extract the HOG features. 

Features were extracted from images of vehicle and non-vehicle classes. One example of each class is shown below.

[image1]: ./examples/car_not_car.png
![alt text][image1]

#### 2. Explain how you settled on your final choice of HOG parameters.

I explored different parameter settings for HOG feature extraction and converged to the parameter set below. This set had the best test-accuracy of 0.98 for a linear SVM clssifier, and seemed to aid car detection best with few false positives. 

``` python
# HOG feature params
'use_gray_img':False,
'hog_channel':'ALL',
'hog_cspace':'HSV',
'hog_n_orientations': 9,
'hog_pixels_per_cell': 8,
'hog_cells_per_block': 2,
'hog_subsampling_max': 3
```

Here is an example of feature extraction using the above parameter values:

[image2]: ./output_images/HOG_example.png
![alt text][image2]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In addition to the HOG features which capture object shape, I used color features to capture object appearance. The color features consisted of a downsampled image and histograms of intensity values in the individual channels of the image. I used the following parameter set for extracting these features:

```python
'color_cspace':'HSV',
'color_spatial_size':(32, 32),
'color_hist_bins':32,
'color_hist_range':(0, 256)
```

I found that using color features in addition to HOG features further improved the test accuracy of the linear SVM classifier while reducing the false positive rate further.

For each training (car / non-car) image, the HOG- and color-features are concatenated to form a one single feature vector. These feature vectors are stacked to form the matrix `X`, and labels car or non-car corresponding to each feature vector  are stored in a vector of `label`s (see function `X, labels = get_training_data()` in `classifiers.py`). 

Then, a linear SVM classifier is fitted to the training data in the function `fit_svm(X, labels)` in file `classifiers.py`. Before training the classifier, the columns of the stacked feature vectors are normalized using `sklearn.preprocessing.StandardScaler()` in order to avoid any particular feature dominating the others by sheer scale. 

The training data is split into training (80 %) and test data (20 %) and randomy shuffled. An accuracy score for the classification is calculated on the test data using `svc.score()`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Initially I implemented a mechanism where a list of windows (x, y-coordinates) was generated for a region of interest in an image, given window size and overlap (`slide_window()` function in `sliding_windows.py`).  This windows list was then filtered for windows containing cars using the function `search_windows()` in `sliding_windows.py`. The `search_windows()` function extracted HOG and color features for each window and classified the window as either containing a car or not. This process was computationally very expensive, since the HOG features were extracted separately for each window. 

To reduce this computationaly complexity, I altered the above mechanism so that the HOG features are extracted only once for a region of interest in an image. Later during window classification, only the portion of the large HOG feature array inside that window is considered. This refined mechanism is implemented in the function `find_cars()` in file `sliding_windows.py`.

[image3]: ./output_images/all_windows_multiscale.png
![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

In the Section on Histogram of Oriented Gradients (HOG) above, I mentioned that I extracted the HOG- and the color features all on images transformed to the HSV colorspace. This seemed to work well with the test-images at first, however the performance on the project-video was verz unsatisfactory. Hence I decided to try using the `YCrCb` colorspace. 

Here are some example images:

[image4]: ./output_images/hotwindows_cars_heatmap.png
![alt text][image4]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./out_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

