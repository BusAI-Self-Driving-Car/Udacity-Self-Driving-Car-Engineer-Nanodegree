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
[imageFlipped]: ./examples/center_2017_11_07_17_27_36_336-flipped.jpg "Image flipped vertically"
[imageSideToCenter1]: ./examples/side_to_center_2017_11_13_10_49_37_318.jpg "Roadside-to-center training -- 1"
[imageSideToCenter2]: ./examples/side_to_center_2017_11_13_10_49_37_590.jpg "Roadside-to-center training -- 2"
[imageSideToCenter3]: ./examples/side_to_center_2017_11_13_10_49_37_931.jpg "Roadside-to-center training -- 3"
[imageSteeringAngleHistBiasedTo0]: ./examples/histogram_biased_to_0.png "Steering angle distribution biased towards 0.0 deg."
[imageSteeringAngleHistEqualized]: ./examples/histogram_equalized.png "Steering angle distribution equalized"

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

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
* [`writeup.md`](writeup.md) contains a description of my approach to tackling this project

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Solution Design Approach

The overall strategy for deriving a model architecture was to successively introduce more complexity (convolutional layers) into the NN model, while extending/refining the training data until a satisfactory performance was achieved. Good performance means no over- or under-fitting, low mean-squared-error, and staying on-track in the simulator in autonomous mode. 

I started out with a simple CNN with a single convolutional layer, mainly out of curiosity to see how well it would perform. I split my image and steering angle data into a training and validation set and trained this model. I had quite high mean-squared-errors (MSE), indicating that the model was significantly underfitting. Then I successively increased the number of convolutional layers until I had much lower MSE than before.

I continuously evaluated track performance in autonomous mode in the simulator, by running the model through the simulator and ensuring that the vehicle could stay on the track. With the NVidia CNN architecture (https://arxiv.org/pdf/1604.07316v1.pdf), the car got quite far on track1, but wasn't able to navigate sharp turns so well. Also, if it went even a little off-track, it wasn't able to return back to the track. To improve the driving behavior in these cases, I augmented/extended the training data with cases specifically targeted at sharp turns and recovery from off-track. See more details below.

At the end of the process, the vehicle was able to drive autonomously around both tracks in the simulator without leaving the road.


### Model Architecture

#### 1. Model exploration

I started with a simple model containing a single convolution layer (see `get_model_single_layer()` in model.py line 139), and gradually introduced more complexity. In the next step, I introduced just one extra image-cropping layer at the beginning (see `get_model_single_layer_cropped()` in model.py line 118). Next, I tried the LeNet architecture I had used in the previous project (`see get_model_lenet()` in mode.py line 84). 

With every step, the model performed better than before. I finally converged onto the NVidia model (paper: https://arxiv.org/pdf/1604.07316v1.pdf; see `get_model_nvidia_arch()` in model.py line 24), with which I got good model fit as well as good performance in the autonomous mode on the simulator tracks. 

#### 2. Final Model Architecture

The final model architecture (see `get_model_nvidia_arch()` in model.py lines 24) consisted of a convolution neural network with the following layers and layer sizes. It was based on the NVidia architecture presented in the paper https://arxiv.org/pdf/1604.07316v1.pdf.

It includes RELU layers to introduce nonlinearity (model.py line 46), and the data is normalized in the model using a Keras lambda layer (model.py line 42). it contains dropout layers in order to reduce overfitting. 

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


### Creation of the Training Set & Training Process

#### 1. Data collection

To capture good driving behavior, first I recorded Recorded a training dataset from the first track in the simulator. The car more or less always in the center of the road. There were no corner cases (car drifting off-road).  Here is an example image of center lane driving:

![alt text][imageCenterLaneDriving]

I developed data preprocessing and data augmentation strategies in parallel to model-exploration. My first try was with the above recorded data and a single convolution layer. With this model, the car just rotated anticlockwise all the time, didn't drive along the road at all. 

The tracks are circuits, where by default one drives anticlockwise leading to a strong bias towards left-ward (negative) steering angles in the data. To correct this imbalance, one could drive the same track in the other direction. However, I simply flipped the images from the anticlockwise dataset vertically and negated the corresponding steering angles to generate additional data. Here are examples of images that have been flipped:

![alt text][imageCenterLaneDriving] ![alt text][imageFlipped]

I also augmented data further with the L/R camera images to teach the car to steer to the center if it moves towards the side of the road. The car was now able to at least drive a little while on the road, but still went off-road at sharp turns, and couldn't recover. I then added a new dataset consisting specifically of side-to-center recovery for both tracks. These images show what a typical recording of such a recovery looks like:

![alt text][imageSideToCenter1] ![alt text][imageSideToCenter3]

At very sharp turns however, the car still didn't steer as sharply as I would have liked it to, and in some cases went off track. One possible reason for this could be the large proportion of data points where the steering angle was 0.0. 

![alt text][imageSteeringAngleHistBiasedTo0]

To equalize this histogram, I dropped data with steering angles 0.0 with a probability of 0.9. This made the sharpest turns on the simulator tracks possible for the model.

![alt text][imageSteeringAngleHistEqualized]

The data was randomly shuffled before splitting it into training data (80 %) and validation data (20%). In total, I had (1564 * 2 flipped-images * 3 cameras) = 9384 training data points and (392 * 2 flipped-images * 3 cameras) = 2352 validation data points. 

#### 2. Preprocessing

The data preprocessing I employed was quite simple, and consists of two steps:
* Cropping the images from the top and from the bottom to focus on the road surface. This crops off the car dashboard at the bottom of the image and some scene imagery irrelevant for the NN model (trees, far away hills, etc.)
* Normalizing the data to the range [-0.5, 0.5]

The preprocessing is part of the NN model itself (see model.py `get_model_nvidia_arch()` lines 41--42). Making it a part of the model itself ensures that we have the same preprocessing used for the training/validation also available while driving in autonomous mode using the model. 

#### 3. Training

I found the best number of training epochs to be 12, as the MSE loss for training and validation monotonically decreased until epoch no. 12. The loss was comparable for training and validation at the end of 12 epochs, but diverged after that. I used an adam optimizer, so manually training the learning rate was not necessary.


✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― 
#### For the future

* Next steps
  * Preprocessing 
    * colorspace transformation (?) 
    * tune steering angle correction
