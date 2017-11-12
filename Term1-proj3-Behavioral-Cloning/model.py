from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def configure_tensorflow_session():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7   # 0.3
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    
def get_model_nvidia_arch():
    init = 'glorot_uniform'
    
    model = Sequential()
    model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x/255. - .5))
    
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), init=init))
    model.add(Activation('elu'))
    model.add(Dropout(0.2))
    #model.add(MaxPooling2D((2, 2)))
    
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2), init=init))
    model.add(Activation('elu'))
    model.add(Dropout(0.2))
    #model.add(MaxPooling2D((2, 2)))
    
    model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2), init=init))
    model.add(Activation('elu'))
    model.add(Dropout(0.2))
    #model.add(MaxPooling2D((2, 2)))
    
    model.add(Convolution2D(64, 3, 3, border_mode='valid', init=init))
    model.add(Activation('elu'))
    model.add(Dropout(0.2))
    #model.add(MaxPooling2D((2, 2)))
    
    model.add(Convolution2D(64, 3, 3, border_mode='valid', init=init))
    model.add(Activation('elu'))
    model.add(Dropout(0.2))
    #model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    
    model.add(Dense(100))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(50))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(10))
    model.add(Activation('elu'))

    model.add(Dense(1))
    return model

def get_model_lenet():
    model = Sequential()
    model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x/255. - .5))
    
    model.add(Convolution2D(6, 5, 5))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D((2, 2)))

    model.add(Convolution2D(16, 5, 5))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(120))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(84))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))    

    model.add(Dense(1))    
    return model

# Epoch 12/12 loss: 0.0070 - val_loss: 0.0206
# Result: Car crosses the bridge and goes beyond. Brakes all the time, but manages
# to come to road-center from the sides, without cross the sidelines. At one point it
# just stopped driving though!
def get_model_single_layer_cropped():
    model = Sequential()
    model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x/255. - .5))
    model.add(Convolution2D(32, 3, 3))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1))
    return model

# Epoch 12/12 loss: 0.0016 - val_loss: 0.0196
# Result: Car stays on road until bridge. Brakes all the time, but manages to 
# come to road-center from the sides, without cross the sidelines.
def get_model_single_layer():
    model = Sequential()
    model.add(Lambda(lambda x: x/255. - .5, input_shape=(160,320,3)))
    model.add(Convolution2D(32, 3, 3))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1))
    return model

















