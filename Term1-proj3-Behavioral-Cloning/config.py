"""
Various configuration parameters for this project
"""

dict_config_params = {}

## Data I/O
dict_config_params['data_dirs'] = []
dict_config_params['data_dirs'].append('../../data/simulator/track1-run1-pretty-centered/')
dict_config_params['data_dirs'].append('../../data/simulator/track2-run1-pretty-centered/')

dict_config_params['data_dirs'].append('../../data/simulator/track1-side-to-center/')
dict_config_params['data_dirs'].append('../../data/simulator/track2-side-to-center/')

dict_config_params['data_dirs'].append('../../data/simulator/track2-side-to-center-poles/')
# dict_config_params['data_dirs'].append('../../data/simulator/track2-side-to-center-poles-2/')
# dict_config_params['data_dirs'].append('../../data/simulator/track2-sharp-turns/')

dict_config_params['batch_size'] = 8  # 32

dict_config_params['angle_threshold'] = 0.05
dict_config_params['angle_correction'] = 0.2
dict_config_params['angle_noise'] = 0   # 0.15

## Data preprocessing
dict_config_params['convert_rbg2gray'] = False  # ATTN: img-preproc should be part of NN model!

## Data augmentation
dict_config_params['sample_drop_prob'] = 0.9
dict_config_params['mult_factor_samples_per_epoch'] = 1
dict_config_params['flip_images'] = True
if dict_config_params['flip_images']:
    dict_config_params['mult_factor_samples_per_epoch'] *= 2
    
dict_config_params['use_only_center_img'] = False
dict_config_params['use_one_of_three_imgs'] = False
if dict_config_params['use_only_center_img'] or dict_config_params['use_one_of_three_imgs']:
    dict_config_params['mult_factor_samples_per_epoch'] *= 1
else:
    dict_config_params['mult_factor_samples_per_epoch'] *= 3


## NN model params
dict_config_params['nn_conv_drop_prob'] = 0.25
dict_config_params['nn_dense_drop_prob'] = 0.25

## Debug
dict_config_params['dbg_data_generators'] = False