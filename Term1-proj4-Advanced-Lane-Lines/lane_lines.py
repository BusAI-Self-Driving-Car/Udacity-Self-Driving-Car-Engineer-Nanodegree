
# coding: utf-8

# In[8]:


import numpy as np
import cv2
import matplotlib.pyplot as plt

# Define a class to receive the characteristics of each line detection
class Line:
    
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        
        self.pixel_pos_x = np.array([])
        self.pixel_pos_y = np.array([])        
        
        # x values of the last n fits of the line
        self.recent_xfitted = []
        
        # average x values of the fitted line over the last n iterations
        self.bestx = None     
        
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        
        # radius of curvature of the line in some units
        self.radius_of_curvature = None 
        
        # distance in meters of vehicle center from the line
        self.line_base_pos = None 
        
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        
        # x values for detected line pixels
        self.fitx = None  
        
        # y values for detected line pixels
        self.fity = None
        
    def get_line_polyfit(self):
        # Fit a second order polynomial
        fit = np.polyfit(self.pixel_pos_y, self.pixel_pos_x, 2)
        return fit
    
    def get_x_y_for_plotting(self, img_shape_y):
        
        ploty = np.linspace(0, img_shape_y-1, img_shape_y )
        
        fit = self.get_line_polyfit()
        fitx = fit[0] * ploty**2 + fit[1] * ploty + fit[2]

        return ploty, fitx
    
    def get_lane_line_curvature(self, dict_config_params, img_shape_y):
    
        ploty = []
        ploty, fitx = self.get_x_y_for_plotting(img_shape_y)
        y_eval = np.max(ploty)

        y_meter_per_pixel = dict_config_params['y_meter_per_pixel']
        x_meter_per_pixel = dict_config_params['x_meter_per_pixel']

        # Fit polynomials to x,y in world space
        fit = np.polyfit(ploty * y_meter_per_pixel, fitx * x_meter_per_pixel, 2)

        # Calculate the new radii of curvature
        curvature_rad = (((1 + (2 * fit[0] * y_eval * y_meter_per_pixel + fit[1])**2)**1.5) / 
                            np.absolute(2*fit[0]))

        return curvature_rad


# In[ ]:


class window:
    def __init__(self, x_low, x_high, y_low, y_high):
        self.x_low = np.int32(x_low)
        self.x_high = np.int32(x_high)
        self.y_low = np.int32(y_low)
        self.y_high = np.int32(y_high)


# In[ ]:


def get_lane_indices(nonzero_indices, window_left, window_right):
    
    nonzeroy = np.array(nonzero_indices[0])
    nonzerox = np.array(nonzero_indices[1])
    
    left_lane_inds = ((nonzeroy >= window_left.y_low) & (nonzeroy < window_left.y_high) & 
                      (nonzerox >= window_left.x_low) & (nonzerox < window_left.x_high)).nonzero()[0]
    
    right_lane_inds = ((nonzeroy >= window_right.y_low) & (nonzeroy < window_right.y_high) & 
                       (nonzerox >= window_right.x_low) & (nonzerox < window_right.x_high)).nonzero()[0]
    
    return left_lane_inds, right_lane_inds
    
# Extract left and right line pixel positions
def get_lane_pixel_positions(nonzero_indices, lane_indices):
    nonzeroy = np.array(nonzero_indices[0])
    nonzerox = np.array(nonzero_indices[1])
    
    x = nonzerox[lane_indices]
    y = nonzeroy[lane_indices] 
    
    return x, y


# In[ ]:


def detect_lane_lines(binary_warped, line_left, line_right, plot_image=False):
    
    #print("binary_warped.shape = {}".format(binary_warped.shape))
    #print()
    
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype(np.uint8)
        
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[np.int(binary_warped.shape[0]/2):,:], axis=0)
    #print("histogram.shape = {}".format(histogram.shape))
    #print()        
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    n_windows = 9
    
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/n_windows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Set the width of the windows +/- margin
    margin = 100
    
    # Set minimum number of pixels found to recenter window
    minpix = 50
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for n_window in range(n_windows):
        
        # Identify window boundaries in x and y (and right and left)
        y_low = binary_warped.shape[0] - (n_window + 1) * window_height
        y_high = binary_warped.shape[0] - n_window * window_height
        
        xleft_low = leftx_current - margin
        xleft_high = leftx_current + margin
        
        xright_low = rightx_current - margin
        xright_high = rightx_current + margin
        
        w_left = window(xleft_low, xleft_high, y_low, y_high) 
        w_right = window(xright_low, xright_high, y_low, y_high) 
        
        # Draw the windows on the visualization image
        # print("left_window = ({}, {}), ({}, {})".format(xleft_low, y_low, xleft_high, y_high))
        if False: #plot_image:
            cv2.rectangle(out_img,
                          (xleft_low, y_low), (xleft_high, y_high), 
                          (0,255,0), 5) 
            cv2.rectangle(out_img,
                          (xright_low, y_low), (xright_high, y_high), 
                          (0,255,0), 5)     
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds, good_right_inds = get_lane_indices(nonzero, w_left, w_right)
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        # Else the next windows retain the x-coords. of the previous (below) one
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices    
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    line_left.pixel_pos_x, line_left.pixel_pos_y = get_lane_pixel_positions(nonzero, left_lane_inds)
    line_right.pixel_pos_x, line_right.pixel_pos_y = get_lane_pixel_positions(nonzero, right_lane_inds)
    
    ploty, left_fitx = line_left.get_x_y_for_plotting(binary_warped.shape[0])
    ploty, right_fitx = line_right.get_x_y_for_plotting(binary_warped.shape[0])
    
    # Color the left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    if plot_image:        
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
    
    return out_img


# In[ ]:


def track_lane_lines(binary_warped, line_left, line_right):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype(np.uint8)
            
    margin = 100  # dict_config_params['x_margin']
    
    left_fit = line_left.get_line_polyfit()
    right_fit = line_right.get_line_polyfit()
    
    x_left = left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2]
    x_right = right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2]
        
    w_left = window(x_left - margin, x_left + margin, nonzeroy.min(), nonzeroy.max()) 
    w_right = window(x_right - margin, x_right + margin, nonzeroy.min(), nonzeroy.max()) 
    
    left_lane_inds, right_lane_inds = get_lane_indices(nonzero, w_left, w_right)
    
    line_left.pixel_pos_x, line_left.pixel_pos_y = get_lane_pixel_positions(nonzero, left_lane_inds)
    line_right.pixel_pos_x, line_right.pixel_pos_y = get_lane_pixel_positions(nonzero, right_lane_inds)

    ploty, left_fitx = line_left.get_x_y_for_plotting(binary_warped.shape[0])
    ploty, right_fitx = line_right.get_x_y_for_plotting(binary_warped.shape[0])
    
    # Color lane-pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    ## Draw search windows for the left and right lane lines
    window_img = np.zeros_like(out_img)
    
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_window_left_line = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_window_right_line = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
    left_window_pts = np.hstack((left_window_left_line, left_window_right_line))
    
    right_window_left_line = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_window_right_line = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
    right_window_pts = np.hstack((right_window_left_line, right_window_right_line))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_window_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_window_pts]), (0,255, 0))
    out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return out_img


# In[ ]:


def project_lane_lines_to_road(frame_undistorted, top_down_binary,
                               line_left, line_right, perspective_M_inv):
    
    if False:  # For debugging: return binary with lane line search windows
        img_size = (top_down_binary.shape[1], top_down_binary.shape[0])
        warped = cv2.warpPerspective(top_down_binary, perspective_M_inv, img_size, flags=cv2.INTER_LINEAR)    
        return warped
    
    # Create an image to draw the lines on
    color_warp = np.zeros_like(frame_undistorted).astype(np.uint8)
        
    ploty, left_fitx = line_left.get_x_y_for_plotting(frame_undistorted.shape[0])
    ploty, right_fitx = line_right.get_x_y_for_plotting(frame_undistorted.shape[0])
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, perspective_M_inv, 
                                  (color_warp.shape[1], color_warp.shape[0])) 
    
    # Combine the result with the original image
    result = cv2.addWeighted(frame_undistorted, 1, newwarp, 0.3, 0)
    
    return result


# In[ ]:


def write_curvature_text_to_image(img, dict_config_params,
                                  line_left, line_right):
    frame_size = img.shape
    left_curverad = line_left.get_lane_line_curvature(dict_config_params, frame_size[0])
    right_curverad = line_right.get_lane_line_curvature(dict_config_params, frame_size[0])
    
    mean_curvature = np.mean([left_curverad, right_curverad])
    #print("curvature left, right = {} m, {} m".format(left_curverad, right_curverad))
    
    font = cv2.FONT_ITALIC
    cv2.putText(img, 'Radius of curvature: {:.02f}m'.format(mean_curvature), (50, 50), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, 'Dist. from lane-center: {:.02f}m'.format(0.), (50, 100), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

