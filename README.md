# The Udacity Self-Driving Car Engineer Nanodegree projects

## Index

#### [Term 1 - Project 1: Lane-Line Detection](Term1-proj1-LaneLines)

In this project, a pipeline that finds lane lines on the road is developed. Computer Vision / image-processing techniques like Gaussian blurring, Canny edge detction and hough line extraction are applied for this purpose.

<p align="center">
 <a href=""><img src="./Term1-proj1-LaneLines/test_videos_output/solidYellowLeft.gif" alt="Overview" width="50%" height="50%"></a>
</p>

#### [Term 1 - Project 2: Traffic-Sign Classication](Term1-proj2-Traffic-Sign-Classifier)

In this project, a Convolutional Neural Network leaning on the LeNet architecture is used to classify traffic signs. 

#### [Term 1 - Project 3: Behavioral cloning](Term1-proj3-Behavioral-Cloning)

Here a CNN model is employed for cloning human driving behavior. The developed model is able to successfully navigate both tracks in the Udacity simulator without driving the car off-track even once. 

<p align="center">
 <a href=""><img src="./Term1-proj3-Behavioral-Cloning/videos/track2.gif" alt="Overview" width="50%" height="50%"></a>
</p>

#### [Term 1 - Project 4: Advanced Lane-Line Detection and Tracking](Term1-proj4-Advanced-Lane-Lines)

Advanced Computer Vision techniques are used in this project to detect and track lane-lines more robustly than in [Term 1 - Project 1: Lane-Line Detection](Term1-proj1-LaneLines). 

Camera calibration and distortion coefficients are extracted from calibration images and used to correct raw images before further processing.

Thresholding on color transformed images and that on gradients of the images is combined to create binary images. A perspective transform is applied to rectify the binary images to the "birds-eye view". Lane pixels are detected in the bird's-eye view and lines are fit to find the lane boundaries. The images are warped back to the original view from the camera. 

Once the lane-lines are detected at the beginning, they are tracked over the subsequent frames by specifying a search window around the polynomial fit determined previously.

<p align="center">
 <a href=""><img src="./Term1-proj4-Advanced-Lane-Lines/videos/out_project_video.gif" alt="Overview" width="50%" height="50%"></a>
</p>

#### [Term 1 - Project 5: Vehicle Detection and Tracking](Term1-proj5-Vehicle-Detection)

In this project, HOG features which capture object shape, and color features which capture object appearance, are used to detect and track vehicles over subsequent frames in a video. A linear SVM classifier is used on windows sliding over an input frame to classify the window as containing a car or not.  

<p align="center">
 <a href=""><img src="./Term1-proj5-Vehicle-Detection/videos/out_video.gif" alt="Overview" width="50%" height="50%"></a>
</p>



## Tips and tricks
* TensorFlow: Grow GPU memory as required by the program (otherwise cudnn errors!):
[Link to guthub issue](https://github.com/tensorflow/tensorflow/issues/6698)
```python
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config=config, ...)
```
