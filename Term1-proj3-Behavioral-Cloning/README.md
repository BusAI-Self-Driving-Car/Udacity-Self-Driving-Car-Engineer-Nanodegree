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
[imageCenterLaneDriving2]: ./examples/center_2017_11_07_17_27_40_602.jpg "Center-lane driving 2"
[imageFlipped]: ./examples/center_2017_11_07_17_27_36_336-flipped.jpg "Image flipped vertically"
[imageCropped]: ./examples/center_2017_11_07_17_27_36_336-cropped.jpg "Cropped image"
[imageSideToCenter1]: ./examples/side_to_center_2017_11_13_10_49_37_318.jpg "Roadside-to-center training -- 1"
[imageSideToCenter2]: ./examples/side_to_center_2017_11_13_10_49_37_590.jpg "Roadside-to-center training -- 2"
[imageSideToCenter3]: ./examples/side_to_center_2017_11_13_10_49_37_931.jpg "Roadside-to-center training -- 3"
[imageSteeringAngleHistBiasedTo0]: ./examples/histogram_biased_to_0.png "Steering angle distribution biased towards 0.0 deg."
[imageSteeringAngleHistEqualized]: ./examples/histogram_equalized.png "Steering angle distribution equalized"

<p align="center">
 <a href=""><img src="./videos/track2.gif" alt="Overview" width="50%" height="50%"></a>
</p>

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

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Solution Design Approach

The overall strategy for deriving a model architecture was to successively introduce more complexity (convolutional layers) into the NN model, while extending/refining the training data until a satisfactory performance was achieved. Good performance means no over- or under-fitting, low mean-squared-error, and staying on-track in the simulator in autonomous mode. 

I started out with a simple CNN with a single convolutional layer, mainly out of curiosity to see how well it would perform. I split my image and steering angle data into a training and validation set and trained this model. I had quite high mean-squared-errors (MSE), indicating that the model was significantly underfitting. Then I successively increased the number of convolutional layers until I had much lower MSE than before.

I continuously evaluated track performance in autonomous mode in the simulator, by running the model through the simulator and ensuring that the vehicle could stay on the track. With the NVidia CNN architecture (https://arxiv.org/pdf/1604.07316v1.pdf), the car got quite far on track1, but wasn't able to navigate sharp turns so well. Also, if it went even a little off-track, it wasn't able to return back to the track. To improve the driving behavior in these cases, I augmented/extended the training data with cases specifically targeted at sharp turns and recovery from off-track. My approach is described in further detail below.

At the end of the process, the vehicle was able to drive autonomously around both tracks in the simulator without leaving the road.


### Model Architecture

#### 1. Model exploration

I started with a simple model containing a single convolution layer (see `get_model_single_layer()` in model.py line 139), and gradually introduced more complexity. In the next step, I introduced just one extra image-cropping layer at the beginning (see `get_model_single_layer_cropped()` in model.py line 118). Next, I tried the LeNet architecture I had used in the previous project (`see get_model_lenet()` in mode.py line 84). 

With every step, the model performed better than before. I finally converged onto the NVidia model (paper: https://arxiv.org/pdf/1604.07316v1.pdf; see `get_model_nvidia_arch()` in model.py line 24), with which I got good model fit as well as good performance in the autonomous mode on the simulator tracks. 

#### 2. Final Model Architecture

The final model architecture (see `get_model_nvidia_arch()` in model.py lines 24) consisted of a convolution neural network with the layers and layer sizes as shown in the table below. It is based on the NVidia architecture presented in the paper https://arxiv.org/pdf/1604.07316v1.pdf.

The data is normalized in the model using a Keras lambda layer (model.py line 42). The model includes RELU layers to introduce nonlinearity (model.py line 46), and  it contains dropout layers in order to reduce overfitting. 

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

To capture good driving behavior, first I recorded Recorded a training dataset from the first track in the simulator. The car more or less always in the center of the road. There were no corner cases (car drifting off-road).  Here are example images of center lane driving:

![alt text][imageCenterLaneDriving] ![alt text][imageCenterLaneDriving2]

I developed data preprocessing and data augmentation strategies in parallel to model-exploration. My first try was with the above recorded data and a single convolution layer. With this model, the car just rotated anticlockwise all the time, didn't drive along the road at all. 

This behavior can be explained as follows. The tracks are circuits, where by default one drives anticlockwise leading to a strong bias towards left-ward (negative) steering angles in the data. To correct this imbalance, one could drive the same track in the other direction. However, I simply flipped the images from the anticlockwise dataset vertically and negated the corresponding steering angles to generate additional data. Here are examples of images that have been flipped:

![alt text][imageCenterLaneDriving] ![alt text][imageFlipped]

I also augmented data further with the L/R camera images to teach the car to steer to the center if it moves towards the side of the road. The car was now able to at least drive a little while on the road, but still went off-road at sharp turns, and couldn't recover. I then added a new dataset consisting specifically of side-to-center recovery for both tracks. These images show what a typical recording of such a recovery looks like:

![alt text][imageSideToCenter1] ![alt text][imageSideToCenter3]

At very sharp turns however, the car still didn't steer as sharply as I would have liked it to, and in some cases went off track. One possible reason for this could be the large proportion of training data points where the steering angle was 0.0. To equalize this histogram and reduce bias towards angles 0.0, I dropped training data with steering angles 0.0 with a probability of 0.9 (see `load_data.py` lines 20--28). This selective training data led to a model that was able to navigate the sharpest turns on the simulator tracks.

![alt text][imageSteeringAngleHistBiasedTo0] ![alt text][imageSteeringAngleHistEqualized]

The data was randomly shuffled before splitting it into training data (80 %) and validation data (20%). In total, I had (1564 * 2 flipped-images * 3 cameras) = 9384 training data points, and (392 * 2 flipped-images * 3 cameras) = 2352 validation data points. 

#### 2. Preprocessing

The data preprocessing I employed was quite simple, and consists of two steps:
* Cropping the images from the top and from the bottom to focus on the road surface. This crops off the car dashboard at the bottom of the image and some scene imagery irrelevant for the NN model (trees, far away hills, etc.)
* Normalizing the data to the range [-0.5, 0.5]

Left: original image, Right: cropped image focusing on the road.

![alt text][imageCenterLaneDriving] ![alt text][imageCropped]

The preprocessing is part of the NN model itself (see model.py `get_model_nvidia_arch()` lines 41--42). Making it a part of the model itself ensures that we have the same preprocessing used for the training/validation also available while driving in autonomous mode using the model. 

#### 3. Training

I found the best number of training epochs to be 12, as the MSE loss for training and validation monotonically decreased until epoch no. 12. The loss was comparable for training and validation at the end of 12 epochs, but diverged after that. I used an adam optimizer, so manually training the learning rate was not necessary.

-- Original Udacity text below --

✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― 

# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

