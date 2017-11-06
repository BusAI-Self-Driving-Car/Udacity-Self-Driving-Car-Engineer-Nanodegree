# **Traffic Sign Recognition** 

**Building a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"

[image4]: ./traffic_signs/17_no_entry.jpg "Traffic Sign 1"
[image5]: ./traffic_signs/12_priority_road.jpg "Traffic Sign 2"
[image6]: ./traffic_signs/14_stop.jpg "Traffic Sign 3"
[image7]: ./traffic_signs/13_yield.jpg "Traffic Sign 4"
[image8]: ./traffic_signs/01_speed_limit_30.jpg "Traffic Sign 5"
[image9]: ./traffic_signs/38_keep_right.jpg "Traffic Sign 6"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/wetoo-cando/sdcnd/blob/master/Term1-proj2-Traffic-Sign-Classifier/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The dataset was loaded from pickle-files, and directly provided images and corresponding labels 
for training, validation, and testing. Wherever required, I used the numpy library for the 
following analysis.

* The training set consists of 34799 images
* The validation set consists of 4410 images
* The test set consists of 12630 images

* The shape of a traffic sign image is (32, 32, 3) -- 32x32 RGB images
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the traffic signs are distributed in the training and validation data. The distibution is quite uneven with a lot more of the speed-limit signs being represented as compared to the other signs. I think the distribution represents the frequency with which traffic signs are encountered in reality quite well.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

#### Grayscaling
To keep the pipeline as simple as possible at the beginning and to be able to train faster, I transformed the color images to grayscale. I was also curious to see whether the goal of >0.93 validation accuracy was achievable just with grayscale images (it is achievable). Note however, that color may play an important role for the classifier to be able to distinguish between, e.g. min. speed 60 km/h and max. 60 km/h traffic signs in the German repertoire.


#### Contrast enhancement
A lot of images in the dataset have bad contrast, e.g. due to bad lighting conditions when the image was taken. To make information in the image more readily accessible to classifier, I enhanced the contrast by equalizing the image intensity histogram using the `cv2.equalizeHist()` function.


#### Feature standardization
As a last step, I mean-corrected and normalized the image data so as to prevent the optimizer from getting stuck in local mimima.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I took over my LeNet architecture from the previously developed LeNet lab (https://github.com/wetoo-cando/sdcnd/blob/master/Term1-cnn-examples/CarND-LeNet-Lab/LeNet-Lab-Solution.py), and modified it until the desired validation accuracy was achieved.


| Layer         		|     Description	        					| I/P dims. | O/P dims | 
|:---------------------:|:----------------------------:|:---------:|:--------:| 
| 2D Convolution 5x5   	| 1x1 stride, valid padding | 32x32 i/p gray image | 28x28x6 |
| 2D max-pooling    	| 2x2 stride, 2x2 window | 28x28x6 | 14x14x6 |
| 2D Convolution 5x5   	| 1x1 stride, valid padding | 14x14x6 | 10x10x16 |
| 2D max-pooling    	| 2x2 stride, 2x2 window | 10x10x16 | 5x5x16 |
| Flattening  	    	| Flatten prev. output into single array | 5x5x16 | 400 |
| Fully connected		| Combines all outputs from previous layer together | 400 | 120 |
| Fully connected		| -- | 120 | 84 |
| Fully connected		| -- | 84 | 43 (no. of unique signs) 

RELU activation was included after every layer to capture non-linear behavior. Dropout layers followed the RELUs to prevent overfitting during training. 
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam-optimizer. Furthermore, I tuned the following parameter values to achieve a validation accuracy > 0.93:
* keep_probability=0.85 for training
* EPOCHS = 25, 
* BATCH_SIZE = 128, and 
* learning_rate = 0.001/1.4 to prevent the optimization from bouncing around

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.942
* test set accuracy of 0.922

#### If an iterative approach was chosen, what was the first architecture that was tried and why was it chosen?

I chose the LeNet-5 architecture for the following reasons:
* I had experience with this architecture previously from this term's "LeNet Lab" lesson. 
* The code was already available from my solution to that lesson, and it had performed well on a very similar problem -- the MNIST handwriting classfication.

* What were some problems with the initial architecture?

The main issue was overfitting. I had pretty high training accuracies, but could not get the validation accuracy to the required target of 0.93.

#### How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

A simple solution to the data overfitting problem was to include dropout layers after every RELU activation.

#### Which parameters were tuned? How were they adjusted and why?

In an attempt to increase the validation accuracy, I initially changed the RELU activation to sigmoid. Although it did improve the accuracy, after a lot of trial and error, I converged on the following architectural changes and parameter values that gave consistently high accuracies for training, validation, as well as testing:
* Reverted from sigmoid activation to ReLU activation units
* Introduced dropout layers after every activation layer in the LeNet architecture to prevent overfitting
* keep_probability=0.85 for training
* EPOCHS = 25, and learning_rate = 0.001/1.4 to prevent the optimization from bouncing around


#### What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? 

The traffic sign classification problem has been addressed in literature quite often, and CNNs have shown a very good performance at it (http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). CNNs are known to classify images without explicitly being programmed to look for any particular features. Also, they execute fast on low-cost GPUs. With that background information, it made sense to begin with the CNN-based LeNet architecture to tackle this problem.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are some German traffic signs that I found on the web (github.com/frankkanis/CarND-Traffic-Sign-Classifier-Project/tree/master/new_signs):

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]

The first image might be difficult to classify because of the viewpoint from the side. 

The second image might be difficult to classify because of its incompleteness towards the bottom.

The third image might be difficult to classify because it's somewhat squashed, but it still has a good picture quality. 

The fourth image should not be that difficult to classify -- almost full frontal view and good lighting conditions and contrast. 

The fifth image should also not be that difficult to classify, but the sign itself occupies only around 1/4th of the image. 

The sixth image might be difficult to classify because of the wear visible on the sign. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 17_no_entry     		| 17_no_entry   								| 
| 12_priority_road		| Roundabout mandatory 							|
| 14_stop				| 14_stop										|
| 13_yield	      		| 13_yield						 				|
| 01_speed_limit_30		| 02_speed_limit_50      						|
| 38_keep_right		 	| 38_keep_right 								|


The model was able to correctly guess 4 of the 6 traffic signs. I would not calculate a percentage over just 6 images. The sample size is too small. Also, running the program multiple times, the correct classified images varies between 3/6 to 6/6. If expressed as a percentage accuracy, just one additional correct or incorrect classification changes the percentage by ~16% !

The test-set on the other hand, achieves an accuracy of 0.922 over 12630 images. 

The accuracy comparison will only make sense when the number of unseen images somewhat approaches the number of images in the test-set, at least in the order of magnitude.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.


For the first image -- 17_no_entry, the correct sign is identified with pretty high certainty.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .48         			| 17_no_entry   								| 
| .14     				| Right of way at next intersection				|
| .07					| No passing									|
| .07	      			| End of all speed/passing limits				|
| .07				    | Roundabout mandatory      					|


For the second image -- 12_priority_road, the classification is completely wrong, perhaps due to the incompleteness of the traffic sign in the image. The correct class does appear at the second place though. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .35         			| Roundabout mandatory   						| 
| .21     				| Priority road									|
| .16					| children crossing								|
| .06	      			| Go straight or left			 				|
| .05				    | Ahead only									|


For the third image -- 14_stop, the classifier is fully certain that it got the result right, and it did!

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.         			| Stop sign   									| 
| .0     				| 												|
| .0					| |
| .0	      			| |
| .0				    | |


For the fourth image -- 13_yield, the correct sign is identified with pretty high relative certainty.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .77         			| 13 yield   									| 
| .14     				| No passing for heavy vehicles					|
| .05					| speed limit 60								|
| .03	      			| No passing					 				|
| .01				    | speed limit 80      							|


For the fifth image -- 01_speed_limit_30, although the result is wrong, the classifier at least seems to think it's looking at a speed-limit sign. It seems to confuse the numbers 30, 50, and 80.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98         			| speed limit 50 								| 
| .01     				| speed limit 80 								|
| .01					| double curve									|
| .0	      			| 					 				|
| .0				    |       							|


For the sixth image -- 38_keep_right, again, the classifier is fully certain that it got the result right, and it did!

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.         			| Keep right   									| 
| .0     				| |
| .0					| |
| .0	      			| |
| .0				    | |


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ―

### For the future
### Design and Test a Model Architecture
#### Preprocessing
TODO I would like to return to this in the future if I find time.
I decided to generate additional data because ... 
To add more data to the the data set, I used the following techniques because ... 
Here is an example of an original image and an augmented image ...
The difference between the original data set and the augmented data set is the following ... 



― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ― ✂ ―

Original Udacity text below:

## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We have included an Ipython notebook that contains further instructions 
and starter code. Be sure to download the [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting three files: 
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file 

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/481/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

### Requirements for Submission
Follow the instructions in the `Traffic_Sign_Classifier.ipynb` notebook and write the project report using the writeup template as a guide, `writeup_template.md`. Submit the project code and writeup document.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

