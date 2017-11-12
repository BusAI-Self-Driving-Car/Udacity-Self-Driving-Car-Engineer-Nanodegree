import numpy as np
import cv2
import csv

lines = []
with open('../../data/simulator/track1-run1-pretty-centered/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


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
	image = cv2.imread(source_path)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

print("image shape = {}".format(X_train[0].shape))
print("X_train shape = {}".format(X_train.shape))
print("y_train shape = {}".format(y_train.shape))

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle='True', nb_epoch=9)

model.save('model.h5')
