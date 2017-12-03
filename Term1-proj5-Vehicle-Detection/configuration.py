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
    
    # Color feature params
    'color_cspace':'YCrCb',
    'color_spatial_size':(32, 32),
    'color_hist_bins':32,
    'color_hist_range':(0, 256),
    
    # HOG feature params
    'use_gray_img':False,
    'hog_channel':'ALL', #'ALL',
    'hog_cspace':'YCrCb',
    'hog_n_orientations': 9,
    'hog_pixels_per_cell': 8,
    'hog_cells_per_block': 2,
    'hog_subsampling_max': 3,
    'hog_subsampling_step': 1,
    
    # Heatmap
    'heat_threshold':2,
    'buffer_len_hotwindows':7,
}