#! /usr/bin/python

from os import path, listdir
from os.path import join, basename
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import sys
sys.dont_write_bytecode = True

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


class Line:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = np.float32(x1)
        self.y1 = np.float32(y1)
        self.x2 = np.float32(x2)
        self.y2 = np.float32(y2)
        self.slope = self.get_slope()
        self.offset = self.get_offset()

    def get_slope(self):
        # Compute slope. Prevent division by zero.
        return (self.y2-self.y1)/(self.x2-self.x1 + np.finfo(float).eps)

    def get_offset(self):
        # Compute y-offset of the line
        return self.y1-self.slope*self.x1


def draw_lines(img, lines, color=(255, 0, 0), thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  

    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    lane_lines = []

    lane_line_segments_left = []
    lane_line_segments_right = []
    for line in lines:
        if line.slope < 0:
            lane_line_segments_left.append(line)
        elif line.slope > 0:
            lane_line_segments_right.append(line)
        else:
            pass  # ignore lines with infinite slope

    # left lane-line
    left_slope = np.median([l.slope for l in lane_line_segments_left])
    # print ("left_slope: {}".format(left_slope))
    left_offset = np.median([l.offset for l in lane_line_segments_left])
    x1, y1 = 0, left_offset
    x2, y2 = -left_offset / left_slope, 0
    lane_lines.append(Line(x1, y1, x2, y2))

    # Right lane-line
    right_slope = np.median([l.slope for l in lane_line_segments_right])
    # print("right_slope: {}".format(right_slope))
    right_offset = np.median([l.offset for l in lane_line_segments_right])
    x1, y1 = 0, right_offset
    x2, y2 = (img.shape[0] - right_offset) / right_slope, img.shape[0]
    lane_lines.append(Line(x1, y1, x2, y2))

    # Draw lane-lines
    for l in lane_lines:
            cv2.line(img, (l.x1, l.y1), (l.x2, l.y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    out_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            out_lines.append(Line(x1, y1, x2, y2))

    draw_lines(line_img, out_lines)
    return line_img


# Python 3 has support for cool math symbols.
def weighted_img(initial_img, img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.
def extract_lane_lines(test_image, plot_images=False):

    if plot_images:
        # print("original image :")
        plt.subplot(3, 2, 1)
        plt.imshow(test_image)
        #plt.show()

    # Convert input image to grayscale
    img_gray = grayscale(test_image)
    if plot_images:
        # print("img_gray :")
        plt.subplot(3, 2, 2)
        plt.imshow(img_gray, cmap='gray')
        #plt.show()

    # Apply Gaussian blur
    kernel_size = 5
    img_blur_gray = gaussian_blur(img_gray, kernel_size)

    # Detect edges using Canny-algorithm
    low_threshold = 50
    high_threshold = 150
    img_edges_gray = canny(img_blur_gray, low_threshold, high_threshold)

    # Retrieve edges only within the region of interest
    imshape = test_image.shape
    vertices = np.array([[(0, imshape[0]), (450, 310),
                          (490, 310), (imshape[1], imshape[0])]], dtype=np.int32)
    img_masked_edges = region_of_interest(img_edges_gray, vertices)
    if plot_images:
        # print("img_masked_edges :")
        plt.subplot(3, 2, 3)
        plt.imshow(img_masked_edges, cmap='gray')
        #plt.show()

    # Define the Hough transform parameters
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 10  # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 15  # minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments
    img_hough_lines = \
        hough_lines(img_masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    img_masked_hough_lines = region_of_interest(img_hough_lines, vertices)
    if plot_images:
        # print("img_hough_lines :")
        plt.subplot(3, 2, 4)
        plt.imshow(img_masked_hough_lines)
        #plt.show()

    # print("img.shape :{}".format(img.shape))
    # print("img_hough_lines.shape :{}".format(img_masked_hough_lines.shape))
    img_blended = weighted_img(test_image, img_masked_hough_lines, α=0.8, β=1., λ=0.)
    if plot_images:
        # print("img_blended: original image + extrapolated lane-lines")
        plt.subplot(3, 2, 5)
        plt.imshow(img_blended)

    # print("\n")
    return img_blended

def process_video_frame(frame):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    frame = cv2.resize(frame, (960, 540))
    result = extract_lane_lines(frame, plot_images=False)
    return result


def main():
    TESTDATA_ROOT = "."

    # -----------------------
    # Test pipeline on images
    # -----------------------
    dir_test_images = "%s/test_images/" % TESTDATA_ROOT
    test_images = [path.join(dir_test_images, name) for name in listdir(dir_test_images)]
    print("test_images: \n{}".format(test_images))

    print("\n")
    plot_images = True
    for test_img in test_images:
        out_path = join('test_images_output', basename(test_img))
        print("out_path: {}".format(out_path))

        # Read in image
        img = cv2.cvtColor(cv2.imread(test_img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img_blended = extract_lane_lines(img, plot_images=plot_images)
        cv2.imwrite(out_path, cv2.cvtColor(img_blended, cv2.COLOR_RGB2BGR))
        plt.show()

    # ------------------------
    # Test pipeline on videos
    # ------------------------
    # video_name = "solidWhiteRight"
    video_name = "solidYellowLeft"
    # video_name = "challenge"

    white_output = '%s/test_videos_output/%s.mp4' % (TESTDATA_ROOT, video_name)
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    # clip1 = VideoFileClip("%s/test_videos/solidWhiteRight.mp4", TESTDATA_ROOT).subclip(0,5)
    clip1 = VideoFileClip("%s/test_videos/%s.mp4" % (TESTDATA_ROOT, video_name))

    white_clip = clip1.fl_image(process_video_frame)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)


if __name__ == '__main__':
    main()