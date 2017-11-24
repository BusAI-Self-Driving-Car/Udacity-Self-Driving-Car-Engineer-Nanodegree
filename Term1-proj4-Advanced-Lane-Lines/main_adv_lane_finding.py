from camera_calibration.py import calibrate_camera
from image_binarization.py import 


if __name__ == '__main__':
    # Calibrate camera
    mtx, dist = calibrate_camera(use_calib_cache=True)
    
    # Read in an image
    image = cv2.cvtColor(cv2.imread('./test_images/straight_lines1.jpg'), cv2.COLOR_BGR2RGB)

    ## Color thresholding
    hls_binary = hls_select(image, thresh=(90, 255))

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(hls_binary, cmap='gray')
    ax2.set_title('Thresholded S', fontsize=50)