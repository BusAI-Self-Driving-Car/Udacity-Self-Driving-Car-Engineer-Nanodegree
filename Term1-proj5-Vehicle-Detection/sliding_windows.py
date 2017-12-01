
# coding: utf-8

# In[1]:


from IPython.core.display import display, HTML

### Widen notebook to fit browser window
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[ ]:


import numpy as np
import cv2

from configuration import dict_config_params
from feature_extracters import transform_colorspace, extract_features, extract_hog_features, extract_color_features, get_hog_features, bin_spatial, color_hist


# In[ ]:


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, y_start_stop, svc, X_scaler, scale=1):
    
    car_windows = []
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_cropped = img[y_start_stop[0]:y_start_stop[1], :, :]
    #img_cropped = transform_colorspace(img_cropped, cspace='YCrCb')
    
    if scale != 1:
        imshape = img_cropped.shape
        img_cropped = cv2.resize(img_cropped, 
                                 (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = img_cropped[:,:,0]
    ch2 = img_cropped[:,:,1]
    ch3 = img_cropped[:,:,2]

    orient = dict_config_params['hog_n_orientations']
    pix_per_cell = dict_config_params['hog_pixels_per_cell']
    cells_per_block = dict_config_params['hog_cells_per_block']
    
    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cells_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cells_per_block + 1 
    nfeat_per_block = orient * cells_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cells_per_block + 1
    
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1,_ = get_hog_features(ch1, orient, pix_per_cell, cells_per_block, feature_vec=False)
    hog2,_ = get_hog_features(ch2, orient, pix_per_cell, cells_per_block, feature_vec=False)
    hog3,_ = get_hog_features(ch3, orient, pix_per_cell, cells_per_block, feature_vec=False)
    
    if dict_config_params['use_gray_img'] is True:
        hogg, hog_image = extract_hog_features(img_cropped, hog_feat=True, visualize=False)    
    
    count_window = 0
    count_car_window = 0
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))            
            
            if dict_config_params['use_gray_img'] is True:
                hog_features = hogg[ypos:ypos+nblocks_per_window, 
                                            xpos:xpos+nblocks_per_window].ravel()                         

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(img_cropped[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            color_features = extract_color_features(subimg, spatial_feat=True, hist_feat=True)
            
            # Combine HOG and color features
            img_features = np.hstack((hog_features, color_features))

            # Scale features and make a prediction            
            test_features = X_scaler.transform(img_features.reshape(1, -1))
            
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            count_window += 1            
            if test_prediction == 1:
                count_car_window += 1
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                
                car_windows.append(((xbox_left, ytop_draw + y_start_stop[0]),
                              (xbox_left + win_draw, ytop_draw + win_draw + y_start_stop[0])))
                cv2.rectangle(draw_img,
                              car_windows[-1][0], car_windows[-1][1],
                              (0, 255, 0), 6) 
    
    
    #print("\nfind_cars(): img_features.shape = {}".format(img_features.shape))
    #print("count_window: {}, count_car_window: {}".format(count_window, count_car_window))
    return draw_img, car_windows


# In[ ]:


def search_windows(img, windows, classifier, X_scaler):

    hog_feat = dict_config_params['use_hog_feat']
    spatial_feat = dict_config_params['use_spatial_feat']
    hist_feat = dict_config_params['use_hist_feat']
    
    # Create an empty list to receive positive detection windows
    on_windows = []
    
    count_window = 0
    count_car_window = 0
    # Iterate over all windows in the list
    for window in windows:
        
        # Extract the test window from original image
        window_img = cv2.resize(img[window[0][1]:window[1][1], 
                                  window[0][0]:window[1][0]], 
                              (64, 64))      
        
        # Extract features for that window
        img_features = extract_features(window_img, verbose=False, 
                                hog_feat=hog_feat, spatial_feat=spatial_feat, hist_feat=hist_feat)                
        
        # Scale extracted features to be fed to classifier
        test_features = X_scaler.transform(np.array(img_features).reshape(1, -1))
        
        # Predict using your classifier
        prediction = classifier.predict(test_features)
        
        # If positive (prediction == 1) then save the window
        count_window += 1
        if prediction == 1:
            count_car_window += 1
            on_windows.append(window)
            
    #print("\nsearch_windows(): img_features.shape: {}".format(img_features.shape))
    #print("count_window: {}, count_car_window: {}".format(count_window, count_car_window))
    
    # Return windows for positive detections
    return on_windows


# In[ ]:


def slide_window(img, 
                 x_start_stop=[None, None], y_start_stop=[None, None], 
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """ A function that takes an image,
    start and stop positions in both x and y, 
    window size (x and y dimensions),  
    and overlap fraction (for both x and y)"""
    
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    
    # Initialize a list to append window positions to
    window_list = []
    
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# In[ ]:


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):

    imcopy = np.copy(img)
    
    for bbox in bboxes:
        # Draw a rectangle
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)

    return imcopy


# In[ ]:


def add_heat(heatmap, bbox_list):

    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox        
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    return heatmap
    
def apply_heat_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap

def draw_labeled_bboxes(img, labels):
    
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):

        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)

    return img


# In[ ]:


from scipy.ndimage.measurements import label

def get_heat_based_bboxes(img, hot_windows):
    
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    
    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)

    # Apply threshold to help remove false positives
    heat = apply_heat_threshold(heat, 4)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    
    return draw_img, heatmap

