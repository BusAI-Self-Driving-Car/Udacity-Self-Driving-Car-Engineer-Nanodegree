import numpy as np
import cv2
import csv

def load_data(lines):
    images = []
    measurements = []

    iter_lines = iter(lines)
    next(iter_lines)
    for line in iter_lines:
        img_filename = line[0]	# center-camera
        img_filename = img_filename.split('/')[-1]

        #source_path = '../../data/simulator/sim-drive-data-udacity/IMG'
        source_path = '../../data/simulator/track1-run1-pretty-centered/IMG/'
        source_path += img_filename
        image = cv2.cvtColor(cv2.imread(source_path), cv2.COLOR_BGR2RGB) #cv2.imread(source_path)

        images.append(image)

        # Augmenting with vertically flipped image to balance the dataset skew towards leftward
        # steering angles. With the original training data, the car keeps going in anticlockwise
        # circles! 
        flipped_image=image.copy()
        flipped_image=cv2.flip(image,1)
        images.append(flipped_image)

        measurement = float(line[3])
        measurements.append(measurement)
        measurements.append(-measurement) # steering angle corr. to flipped image

    X_train = np.array(images)
    y_train = np.array(measurements)
    return X_train, y_train
    
X_train, y_train = load_data(lines)
print("image shape = {}".format(X_train[0].shape))
print("X_train shape = {}".format(X_train.shape))
print("y_train shape = {}".format(y_train.shape))
