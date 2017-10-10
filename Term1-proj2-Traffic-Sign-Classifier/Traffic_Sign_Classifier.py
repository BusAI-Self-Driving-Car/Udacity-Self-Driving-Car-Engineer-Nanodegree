#! /usr/bin/python

# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = "./examples/traffic-signs-data/train.p"
validation_file = "./examples/traffic-signs-data/valid.p"
testing_file = "./examples/traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print("X_train.shape: {}".format(X_train.shape))