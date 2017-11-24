
# coding: utf-8

# ### ToDo
# 
# * ...

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

from camera_calibration import calibrate_camera, undistort_image
from image_binarization import abs_sobel_thresh_x_or_y, mag_thresh, dir_threshold, hls_select
from perspective_transformation import get_src_dst_vertices, get_perspective_transform, warp_image_to_top_down_view

def process_frame(image, mtx, dist, perspective_M):
    
    # Gradient thresholding
    ## Thresholding on x- or y-gradients
    grad_x_binary = abs_sobel_thresh_x_or_y(image, orient='x', thresh_min=20, thresh_max=100)
    grad_y_binary = abs_sobel_thresh_x_or_y(image, orient='y', thresh_min=20, thresh_max=100)

    ## Thresholding on magnitude of gradient
    mag_binary = mag_thresh(image, sobel_kernel=3, mag_thresh=(30, 100))

    ## Thresholding on direction of gradient
    dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))

    ## Combine different gradient thresholding strategies
    combined = np.zeros_like(mag_binary)
    combined[((grad_x_binary==1) & (grad_y_binary==1)) | ((mag_binary==1) & (dir_binary==1))] = 1

    ## Color thresholding
    hls_binary = hls_select(image, thresh=(90, 255))

    # Combine the threshold and gradient thresholding
    combined_binary = np.zeros_like(grad_x_binary)
    combined_binary[(grad_x_binary == 1) | (hls_binary == 1)] = 1    

    img_size = (image.shape[1], image.shape[0])
    top_down_binary = warp_image_to_top_down_view(combined_binary.astype(np.uint8), 
                                   img_size, 
                                   mtx, dist, perspective_M).astype(bool)
    return top_down_binary


def initialize():
    # Calibrate camera
    mtx, dist = calibrate_camera(use_calib_cache=True)
    
    # Perspective transform
    test_image = cv2.cvtColor(cv2.imread('./test_images/straight_lines1.jpg'), cv2.COLOR_BGR2RGB)

    img_size = (test_image.shape[1], test_image.shape[0])
    src, dst = get_src_dst_vertices(img_size)
    
    img_undistorted = undistort_image(test_image, mtx, dist, plot_images=False)
    _, perspective_M, perspective_M_inv = get_perspective_transform(img_undistorted, img_size, 
                                                                    mtx, dist, src, dst)
    
    return mtx, dist, perspective_M, perspective_M_inv
    
    
def test_process_frame():
    mtx, dist, perspective_M, perspective_M_inv = initialize()

    # Read in an image
    image = cv2.cvtColor(cv2.imread('./test_images/straight_lines2.jpg'), cv2.COLOR_BGR2RGB)
    img_undistorted = undistort_image(image, mtx, dist, plot_images=False)
    top_down_binary = process_frame(img_undistorted, mtx, dist, perspective_M)

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(top_down_binary, cmap='gray')
    ax2.set_title('Undistorted and Warped Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    

#test_process_frame()


# In[2]:


class window:
    def __init__(self, x_low, x_high, y_low, y_high):
        self.x_low = np.int32(x_low)
        self.x_high = np.int32(x_high)
        self.y_low = np.int32(y_low)
        self.y_high = np.int32(y_high)

def get_lane_indices(nonzero_indices, window_left, window_right):
    
    nonzeroy = np.array(nonzero_indices[0])
    nonzerox = np.array(nonzero_indices[1])
    
    left_lane_inds = ((nonzeroy >= window_left.y_low) & (nonzeroy < window_left.y_high) & 
                      (nonzerox >= window_left.x_low) & (nonzerox < window_left.x_high)).nonzero()[0]
    
    right_lane_inds = ((nonzeroy >= window_right.y_low) & (nonzeroy < window_right.y_high) & 
                       (nonzerox >= window_right.x_low) & (nonzerox < window_right.x_high)).nonzero()[0]
    
    return left_lane_inds, right_lane_inds
    
# Extract left and right line pixel positions
def get_lane_pixel_positions(nonzero_indices, left_lane_inds, right_lane_inds):
    nonzeroy = np.array(nonzero_indices[0])
    nonzerox = np.array(nonzero_indices[1])
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    return leftx, lefty, rightx, righty

def get_line_fits(binary_warped, leftx, lefty, rightx, righty):
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    return left_fit, right_fit

def get_x_y_for_plotting(binary_warped, left_fit, right_fit):
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return ploty, left_fitx, right_fitx


# In[3]:


def track_lane_lines(binary_warped, left_fit, right_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype(np.uint8)
            
    margin = 100  # dict_config_params['x_margin']
    x_left = left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2]
    x_right = right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2]
        
    w_left = window(x_left - margin, x_left + margin, nonzeroy.min(), nonzeroy.max()) 
    w_right = window(x_right - margin, x_right + margin, nonzeroy.min(), nonzeroy.max()) 
    
    left_lane_inds, right_lane_inds = get_lane_indices(nonzero, w_left, w_right)
    leftx, lefty, rightx, righty = get_lane_pixel_positions(nonzero, left_lane_inds, right_lane_inds)
    
    left_fit, right_fit = get_line_fits(binary_warped, leftx, lefty, rightx, righty)
    ploty, left_fitx, right_fitx = get_x_y_for_plotting(binary_warped, left_fit, right_fit)
    
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
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return result, left_fitx, right_fitx, ploty
    

def detect_lane_lines(binary_warped, plot_image=False):
    
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

    leftx, lefty, rightx, righty = get_lane_pixel_positions(nonzero, left_lane_inds, right_lane_inds)
    
    left_fit, right_fit = get_line_fits(binary_warped, leftx, lefty, rightx, righty)
    ploty, left_fitx, right_fitx = get_x_y_for_plotting(binary_warped, left_fit, right_fit)
    
    # Color the left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    if plot_image:        
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
    
    return out_img, left_fit, right_fit, left_fitx, right_fitx, ploty

def test_detect_lane_lines():
    mtx, dist, perspective_M, perspective_M_inv = initialize()

    # Read in an image
    image = cv2.cvtColor(cv2.imread('./test_images/straight_lines2.jpg'), cv2.COLOR_BGR2RGB)
    img_undistorted = undistort_image(image, mtx, dist, plot_images=False)
    top_down_binary = process_frame(img_undistorted, mtx, dist, perspective_M)
    
    print("top_down_binary.shape = {}".format(top_down_binary.shape))
    out_image, left_fit, right_fit, left_fitx, right_fitx, ploty = detect_lane_lines(top_down_binary, 
                                                                                     plot_image=True)

    print("left_fit: {}".format(left_fit))
    print("right_fit: {}".format(right_fit))
    
# test_detect_lane_lines()


# In[4]:


# Reset global
num_frames_processed = 0


# In[5]:


### Video
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from lane_lines import Line

def get_lane_line_curvatures(left_fitx, right_fitx, ploty):
    
    y_eval = np.max(ploty)

    y_meter_per_pixel = dict_config_params['y_meter_per_pixel']
    x_meter_per_pixel = dict_config_params['x_meter_per_pixel']

    # Fit polynomials to x,y in world space
    left_fit = np.polyfit(ploty * y_meter_per_pixel, left_fitx * x_meter_per_pixel, 2)
    right_fit = np.polyfit(ploty * y_meter_per_pixel, right_fitx * x_meter_per_pixel, 2)
    
    # Calculate the new radii of curvature
    left_curverad = (((1 + (2 * left_fit[0] * y_eval * y_meter_per_pixel + left_fit[1])**2)**1.5) / 
                        np.absolute(2*left_fit[0]))

    right_curverad = (((1 + (2 * right_fit[0] * y_eval * y_meter_per_pixel + right_fit[1])**2)**1.5) / 
                        np.absolute(2*right_fit[0]))
    
    return left_curverad, right_curverad

def project_lane_lines_to_road(frame_undistorted, top_down_binary,
                               left_fitx, right_fitx, ploty, perspective_M_inv):
    
    if True:
        img_size = (top_down_binary.shape[1], top_down_binary.shape[0])
        warped = cv2.warpPerspective(top_down_binary, perspective_M_inv, img_size, flags=cv2.INTER_LINEAR)    
        return warped
    
    # Create an image to draw the lines on
    color_warp = np.zeros_like(frame_undistorted).astype(np.uint8)
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, perspective_M_inv, (color_warp.shape[1], color_warp.shape[0])) 
    
    # Combine the result with the original image
    result = cv2.addWeighted(frame_undistorted, 1, newwarp, 0.3, 0)
    
    return result

def process_video_frame(frame):
    # NOTE: output should be a color image (3 channel) for processing video below    
    global num_frames_processed    
    global mtx, dist, perspective_M, perspective_M_inv
    global left_fit, right_fit, left_fitx, right_fitx
    #global line_left, line_right
    
    frame_undistorted = undistort_image(frame, mtx, dist, plot_images=False)
    top_down_binary = process_frame(frame_undistorted, mtx, dist, perspective_M)
    
    out_img = None
    ploty = None
    if num_frames_processed==0:        
        out_img, left_fit, right_fit, left_fitx, right_fitx, ploty = detect_lane_lines(top_down_binary, 
                                                                                       plot_image=False)         
    else:        
        out_img, left_fitx, right_fitx, ploty = track_lane_lines(top_down_binary, left_fit, right_fit)
    
    num_frames_processed += 1
    # print("num_frames_processed: {}".format(num_frames_processed))
        
    img_lines_on_road = project_lane_lines_to_road(frame_undistorted, out_img,
                                                   left_fitx, right_fitx, ploty, perspective_M_inv)
    
    left_curverad, right_curverad = get_lane_line_curvatures(left_fitx, right_fitx, ploty)
    # print("left_curverad, right_curverad = {} m, {} m".format(left_curverad, right_curverad))

    return img_lines_on_road
    
    
# Globals
left_fit, right_fit, left_fitx, right_fitx = None, None, None, None
mtx, dist, perspective_M, perspective_M_inv = None, None, None, None
line_left = None
line_right = None


if __name__ == '__main__':
    ## Config parameters
    dict_config_params = {}

    # Data I/O
    dict_config_params['x_margin'] = 100

    # Lane radius-of-curvature calculations
    dict_config_params['y_meter_per_pixel'] = 30/720 # meters per pixel in y dimension
    dict_config_params['x_meter_per_pixel'] = 3.7/700 # meters per pixel in x dimension
    
    mtx, dist, perspective_M, perspective_M_inv = initialize()
    line_left = Line()
    line_right = Line()
    
    ## secs. 38--43 are difficult
    clip1 = VideoFileClip("project_video.mp4").subclip(0,2)
    # clip1 = VideoFileClip("project_video.mp4")
    clip = clip1.fl_image(process_video_frame)
    clip.write_videofile("out_project_video.mp4", audio=False)

    # Reset global
    num_frames_processed = 0

