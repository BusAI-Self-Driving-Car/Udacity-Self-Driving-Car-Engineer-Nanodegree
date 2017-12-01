# coding: utf-8

dict_config_params = {
    
    # Training data
    'truncate_training_data_after':-1,
    
    # Classifier
    'use_cached_classifier':True,
    'overwrite_cached_classifier':True,
    
    # Flags for which features to use
    'use_hog_feat':True,
    'use_spatial_feat':True, 
    'use_hist_feat':True,     
    
    # Common params for color and HOG features
    #'color_space':'RGB',
    
    # Color feature params
    'color_cspace':'RGB',
    'color_spatial_size':(32, 32),
    'color_hist_bins':32,
    'color_hist_range':(0, 256),
    
    # HOG feature params
    'use_gray_img':True,
    'hog_channel':'ALL', #'ALL',
    'hog_cspace':'RGB',
    'hog_n_orientations': 9,
    'hog_pixels_per_cell': 8,
    'hog_cells_per_block': 2
}