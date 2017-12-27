# The Udacity Self-Driving Car Engineer Nanodegree projects

## Index

#### [Term 1 - Project 1: Lane-Line Detection](Term1-proj1-LaneLines)

#### [Term 1 - Project 2: Traffic-Sign Classication](Term1-proj2-Traffic-Sign-Classifier)

#### [Term 1 - Project 3: Behavioral cloning](Term1-proj3-Behavioral-Cloning)

#### [Term 1 - Project 4: Advanced Lane-Line Detection and Tracking](Term1-proj4-Advanced-Lane-Lines)

#### [Term 1 - Project 5: Vehicle Detection and Tracking](Term1-proj5-Vehicle-Detection)


### Tips and tricks
* TensorFlow: Grow GPU memory as required by the program (otherwise cudnn errors!):
[Link to guthub issue](https://github.com/tensorflow/tensorflow/issues/6698)
```python
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config=config, ...)
```
