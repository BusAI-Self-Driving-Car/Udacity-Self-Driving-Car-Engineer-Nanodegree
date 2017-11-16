# Read data from the driving_log.csv file
import csv
import numpy as np

import cv2
import sklearn
from sklearn.utils import shuffle


def read_csv_driving_log():
    lines = []

    for data_dir in dict_config_params['data_dirs']:
        
        with open(data_dir + 'driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                
                steer_angle = float(line[3])
                drop_sample = False
                if abs(steer_angle) == 0.0: # dict_config_params['angle_threshold']
                    sample_drop_prob = dict_config_params['sample_drop_prob']
                    drop_sample = np.random.choice([True, False], p=[sample_drop_prob, 1-sample_drop_prob])

                if not drop_sample:
                    line.append(data_dir)
                    lines.append(line)
            
    print("No. of lines read from CSV-file: {}".format(len(lines)))
    #for line in lines:
    #     print("line: {}".format(line))
    return lines

# Image preprocessing
def preprocess_image(image):
    if dict_config_params['convert_rbg2gray']:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        return image

def get_data(batch_samples):
    images = []
    angles = []
    for batch_sample in batch_samples:
        path = batch_sample[-1] + 'IMG/'
        name_center = path + batch_sample[0].split('/')[-1]
        name_left = path + batch_sample[1].split('/')[-1]
        name_right = path + batch_sample[2].split('/')[-1]

        # Augmenting with vertically flipped image to balance the dataset skew towards leftward
        # steering angles. With the original training data, the car keeps going in anticlockwise
        # circles!               
        # print("name_center: {}".format(name_center))                
        angle_noise = dict_config_params['angle_noise']
        camera = np.random.choice(['center', 'left', 'right'])

        gate = True
        if dict_config_params['use_one_of_three_imgs']:
            gate = False
        
        steer_angle = float(batch_sample[3])
        if dict_config_params['use_only_center_img'] or (gate==True or camera=='center'):
            center_image = cv2.cvtColor(cv2.imread(name_center), cv2.COLOR_BGR2RGB)
            center_image = preprocess_image(center_image)
            flipped_image=center_image.copy()
            flipped_image=cv2.flip(center_image,1)
            images.extend((center_image, flipped_image))

            center_angle = steer_angle
            center_angle += np.random.normal(0, angle_noise)
            angles.extend((center_angle, -center_angle))

        if ~dict_config_params['use_only_center_img'] and (gate==True or camera == 'left'):
            left_image = cv2.cvtColor(cv2.imread(name_left), cv2.COLOR_BGR2RGB)
            left_image = preprocess_image(left_image)
            flipped_image=left_image.copy()
            flipped_image=cv2.flip(left_image,1)
            images.extend((left_image, flipped_image))

            left_angle = steer_angle + dict_config_params['angle_correction']
            left_angle += np.random.normal(0, angle_noise)
            angles.extend((left_angle, -left_angle))

        if ~dict_config_params['use_only_center_img'] and (gate==True or camera == 'right'):
            right_image = cv2.cvtColor(cv2.imread(name_right), cv2.COLOR_BGR2RGB)
            right_image = preprocess_image(right_image)
            flipped_image=right_image.copy()
            flipped_image=cv2.flip(right_image,1)
            images.extend((right_image, flipped_image))

            right_angle = steer_angle - dict_config_params['angle_correction']
            right_angle += np.random.normal(0, angle_noise)
            angles.extend((right_angle, -right_angle))                                   

            
    if dict_config_params['convert_rbg2gray']:
        X = np.expand_dims(np.array([img for img in images]), 4)
    else:
        X = np.array(images)
        
    y = np.array(angles)
    return X, y

def generator(samples, batch_size=32):
    
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates        
        # shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            X, y = get_data(batch_samples)                        
            yield shuffle(X, y)