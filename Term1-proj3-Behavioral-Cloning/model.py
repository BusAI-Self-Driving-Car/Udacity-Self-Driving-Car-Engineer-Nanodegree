from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from config import dict_config_params
from import_data import read_csv_driving_log, get_data, generator
from sklearn.model_selection import train_test_split


def configure_tensorflow_session():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3   # 0.3
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    
def get_model_nvidia_arch(dict_config_params):
    
    nn_conv_drop_prob = dict_config_params['nn_conv_drop_prob']  #0.2
    
    if dict_config_params['convert_rbg2gray']:
        input_shape=(160,320,1)
    else:
        input_shape=(160,320,3)
    
    model = Sequential()
    model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=input_shape))
    model.add(Lambda(lambda x: x/255. - .5))
    
    # model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), init=init))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(nn_conv_drop_prob))
    
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(nn_conv_drop_prob))
    
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(nn_conv_drop_prob))
    
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Dropout(nn_conv_drop_prob))
    
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Dropout(nn_conv_drop_prob))

    model.add(Flatten())
    
    nn_dense_drop_prob = dict_config_params['nn_dense_drop_prob']  #0.5
    
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(nn_dense_drop_prob))
    
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(nn_dense_drop_prob))
    
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(nn_dense_drop_prob))

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

if __name__ == '__main__':

    lines = read_csv_driving_log()
        
    # Generator functions
    train_samples, validation_samples = train_test_split(shuffle(lines), test_size=0.2)
    print("num_train_samples = {}".format(len(train_samples)))
    print("num_validation_samples = {}".format(len(validation_samples)))
    print()

    train_generator = generator(train_samples, batch_size=dict_config_params['batch_size'])
    validation_generator = generator(validation_samples, batch_size=dict_config_params['batch_size'])

    if False:
        X_plot, y_plot = get_data(train_samples)
        #print("y_plot = {}".format(y_plot))
        plot_histogram(y_plot)
        
        color_map = None
        if dict_config_params['convert_rbg2gray']:
            color_map = 'gray'
        visualize_images(X_plot, y_plot, num_random_imgs=20, color_map=color_map)

    ## NN model
    configure_tensorflow_session()

    # model = get_model_single_layer()
    # model = get_model_single_layer_cropped()
    # model = get_model_lenet()
    model = get_model_nvidia_arch(dict_config_params)

    model.summary()

    model.compile(loss='mse', optimizer='adam')

    # X_train, y_train = shuffle(np.array(images), np.array(measurements)) 
    # history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle='True', nb_epoch=12)

    mult_factor = dict_config_params['mult_factor_samples_per_epoch']
    print("mult_factor = {}".format(mult_factor))
    print()
    history_object = model.fit_generator(train_generator, 
                                         samples_per_epoch= mult_factor*len(train_samples), 
                                         validation_data=validation_generator, 
                                         nb_val_samples=mult_factor*len(validation_samples), 
                                         nb_epoch=12)

    if False:
        ### plot the training and validation loss for each epoch
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()













