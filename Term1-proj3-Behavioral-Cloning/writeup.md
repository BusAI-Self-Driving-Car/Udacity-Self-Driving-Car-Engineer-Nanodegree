# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

[imageCenterLaneDriving]: ./examples/center_2017_11_07_17_27_36_336.jpg "Center-lane driving"
[imageFlipped]: ./examples/flipped.png "Images flipped vertically"
[imageSteeringAngleHistBiasedTo0]: ./examples/histogram_biased_to_0.png "Steering angle distribution biased towards 0.0 deg."
[imageSteeringAngleHistEqualized]: ./examples/histogram_equalized.png "Steering angle distribution equalized"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [`config.py`](config.py) contains the parameter settings for all parts of the project
* [`data_visualization.py`](data_visualization.py) contains functions for visualizing images from the dataset and distribution of steering angles
* [`model.py`](model.py) contains NN model-definitions and code for training the model
* [`drive.py`](drive.py) can be used for driving the car in autonomous mode
* [`model.h5`](model.h5) contains trained CNN-based NVidia model for autonomous cars
* [`video.mp4`](video.mp4) shows a successful drive by the above model on track1 of the Udacity self-driving car simulator
* [`writeup_report.md`](writeup_report.md) contains a description of my approach to tackling this project

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Model exploration

I started with a simple model containing a single convolution layer (see get_model_single_layer() in model.py line 139), and gradually introduced more complexity. In the next step, I introduced just one extra image-cropping layer at the beginning (see get_model_single_layer_cropped() in model.py line 118). Next, I tried the LeNet architecture I had used in the previous project (see get_model_lenet() in mode.py line 84). 

With every step, the model performed better than before. I finally converged onto the NVidia model (see get_model_nvidia_arch() in model.py line 24), with which I got good model fit as well as good performance in the autonomous mode on the simulator tracks. 

This model consists of CNNs with 5x5 and 3x3 filter sizes and depths between 24 and 64. The model includes RELU layers to introduce nonlinearity (code line 46), and the data is normalized in the model using a Keras lambda layer (code line 42). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 160). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 190).

#### 4. Appropriate training data

I developed data preprocessing and augmentation strategies in parallel to model-exploration. Training data was successively extended as required so that the NN model was able to keep the vehicle driving on the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to successively introduce more complexity (convolutional layers) into the NN model, while extending/refining the training data until a satisfactory performance was achieved -- no over- or under-fitting, low mean-squared-error, and good performance on the simulator tracks in autonomous mode. 

I started out with a simple CNN with a single convolutional layer, mainly out of curiosity to see how well it would perform. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I had quite high mean-squared-errors (MSE), indicating that the model was significantly underfitting. Then I successively increased the number of convolutional layers until I had much lower MSE than before. But the autonomous-mode performance on track 2 in the simulator wasn't that good; the model seemed to overfit the training data. So I introduced dropout layers in the CNN, after which the model generalized quite well to track2.

I continuously evaluated track performance in autonomous mode in the simulator. With the NVidia CNN architecture, the car got quite far on track1, but wasn't able to navigate sharp turns so well. Also, if it went even a little off-track, it wasn't able to return back to the track. To improve the driving behavior in these cases, I augmented/extended the training data with cases specifically targeted at sharp turns and recovery from off-track. See more details below.

At the end of the process, the vehicle was able to drive autonomously around both tracks in the simulator without leaving the road.

#### 2. Final Model Architecture

The final model architecture (see get_model_nvidia_arch() in model.py lines 24) consisted of a convolution neural network with the following layers and layer sizes. It was based on the NVidia architecture presented in the paper https://arxiv.org/pdf/1604.07316v1.pdf.

| Layer         		|     Description	        					| I/P dims. | O/P dims | 
|:---------------------:|:----------------------------:|:---------:|:--------:| 
| Cropping & normalization   	| top, bottom = 60, 20 crop | 160x320x3 RGB image | 80x320x3 |
| 2D Convolution 5x5, ReLU   	| 2x2 stride, valid padding | 80x320x3 | 38x158x24 |
| Dropout    	| -- | -- | -- |
| 2D Convolution 5x5, ReLU   	| 2x2 stride, valid padding | 38x158x24 | 17x77x36 |
| Dropout    	| -- | -- | -- |
| 2D Convolution 5x5, ReLU   	| 2x2 stride, valid padding | 17x77x36 | 7x37x48 |
| Dropout    	| -- | -- | -- |
| 2D Convolution 3x3, ReLU   	| 2x2 stride, valid padding | 7x37x48 | 5x35x64 |
| Dropout    	| -- | -- | -- |
| 2D Convolution 3x3, ReLU   	| 2x2 stride, valid padding | 5x35x64 | 3x33x64 |
| Dropout    	| -- | -- | -- |
| Flattening  	    	| Flatten prev. output into single array | 3x33x64 | 6336 |
| Fully connected, ReLU		| Combines all outputs from previous layer together | 6336 | 100 |
| Dropout    	| -- | -- | -- |
| Fully connected, ReLU		| -- | 100 | 50 |
| Dropout    	| -- | -- | -- |
| Fully connected, ReLU		| -- | 50 | 10
| Dropout    	| -- | -- | -- |
| Fully connected, ReLU		| -- | 10 | 1 (steering angle prediction) 

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded Recorded a training dataset from the first track. The car more or less always in the center of the road. There were no corner cases (car drifting off-road).  Here is an example image of center lane driving:

![alt text][imageCenterLaneDriving]

I developed data preprocessing and data augmentation strategies in parallel to model-exploration. My first try was with the above recorded data and a single convolution layer. With this model, the car just rotated anticlockwise all the time, didn't drive along the road at all. 

The tracks are a circuit where by default. One drives counterclockwise leading to a strong bias towards left-ward (negative) steering angles in the data. To correct this imbalance, one could drive the same track in the other direction. However, I simply flipped the images from the counterclockwise dataset vertically and negated the corresponding steering angles to simulate driving in the other direction. Here are examples of images that have been flipped:

![alt text][imageFlipped]

I also augmented data further with the L/R camera images to teach the car to steer to the center if it moves towards the side of the road. The car was now able to at least drive a little while on the road, but still went off-road at sharp turns, and couldn't recover. I then added a new dataset consisting specifically of side-to-center recovery for both tracks. This led to improved turning behavior on sharp turns. These images show what a typical recording of recovery looks like:

![alt text][image3]
![alt text][image4]
![alt text][image5]

At very sharp turns however, the car still didn't steer as sharply as I would have liked it to, and in some cases went off track. One possible reason for this could be the large proportion of data points where the steering angle was 0.0. 

![alt text][imageSteeringAngleHistBiasedTo0]

To equalize this histogram, I dropped data with steering angles==0. with a probability of 0.9. This made the sharpest turns on the simulator tracks possible for the model.

![alt text][imageSteeringAngleHistEqualized]

--
After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.


✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― 
#### For the future

* Next steps
 * Preprocessing 
  * colorspace transformation(?) 
  * tune steering angle correction
