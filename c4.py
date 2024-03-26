import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

from moviepy.editor import VideoFileClip
from IPython.display import HTML

images = glob.glob('./Calibration_Images/calibration*.jpg')

objpoints = []
imgpoints = []

imggg = mpimg.imread('test2.jpg')
#plt.imshow(test_image)
#plt.show()

nx = 9
ny = 6
offset = 100

#Prepare object points
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

for fname in images:
    img = mpimg.imread(fname)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # If found, draw corners
    if ret == True:
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        imgpoints.append(corners)
        objpoints.append(objp)

def video(imggg):
    imgg = imggg
    def corners_unwarp(imgg, xx, yy, ob, im):
        retp, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(ob, im, imgg.shape[1:], None, None)
        undist = cv2.undistort(imgg, mtx, dist, None, mtx)
        
        return undist 
    undist = corners_unwarp(imggg, 9, 6, objpoints, imgpoints)
    img = undist
    #### Sobel
    def pipeline(img):
        def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
            
            # Apply the following steps to img
            # 1) Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # 2) Take the derivative in x or y given orient = 'x' or 'y'
            if orient == 'x':
                sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            else:
                sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
            
            # 3) Take the absolute value of the derivative or gradient
            abs_sobel = np.absolute(sobel)
            # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
            scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
            # 5) Create a mask of 1's where the scaled gradient magnitude 
                    # is > thresh_min and < thresh_max
            sbinary = np.zeros_like(scaled_sobel)
            sbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
            # 6) Return this mask as your binary_output image
            #plt.imshow(sbinary, cmap='gray')
            binary_output = sbinary
            return binary_output

        ##### Magnitude Threshold 
        def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
            
            # Apply the following steps to img
            # 1) Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # 2) Take the gradient in x and y separately
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
            # 3) Calculate the magnitude 
            abs_sobel_x = np.sqrt(np.square(sobel_x))
            abs_sobel_y = np.sqrt(np.square(sobel_y))
            gradmag = np.sqrt(sobel_x**2 + sobel_y**2)
            # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
            scaled_sobel_x = np.uint8(255*abs_sobel_x/np.max(abs_sobel_x))
            scaled_sobel_y = np.uint8(255*abs_sobel_y/np.max(abs_sobel_y))
            scaled_sobel = np.uint8(255*gradmag/np.max(gradmag))
            # 5) Create a binary mask where mag thresholds are met
            sbinary = np.zeros_like(gradmag)
            sbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
            # 6) Return this mask as your binary_output image
            binary_output = sbinary # Remove this line
            return binary_output

        ####### Direction Threshold 
        def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
            
            # Apply the following steps to img
            # 1) Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # 2) Take the gradient in x and y separately
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
            # 3) Take the absolute value of the x and y gradients
            abs_sobel_x = np.absolute(sobel_x)
            abs_sobel_y = np.absolute(sobel_y)
            # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
            derivative = np.arctan2(abs_sobel_y,abs_sobel_x)
            # 5) Create a binary mask where direction thresholds are met
            sbinary = np.zeros_like(derivative)
            sbinary[(derivative >= thresh[0]) & (derivative <= thresh[1])] = 1
            # 6) Return this mask as your binary_output image
            binary_output = sbinary            
            return binary_output

        # Apply each of the thresholding functions
        gradx = abs_sobel_thresh(img, orient='x', thresh_min=100, thresh_max=255)
        grady = abs_sobel_thresh(img, orient='y', thresh_min=100, thresh_max=255)
        mag_binary = mag_thresh(img, sobel_kernel= 3, mag_thresh=(90, 180))
        dir_binary = dir_threshold(img, sobel_kernel= 3, thresh=(0, np.pi/2))

        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

        def hls_select(img, thresh=(0, 255)):
            # 1) Convert to HLS color space
            hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            H = hls[:,:,0]
            L = hls[:,:,1]
            S = hls[:,:,2]
            # 2) Apply a threshold to the S channel
            binary = np.zeros_like(S)
            binary[(S>thresh[0]) & (S<=thresh[1])] = 1
            # 3) Return a binary image of threshold result
            binary_output = binary
            #binary_output = np.copy(img) # placeholder line
            return binary_output
        
        # Optional TODO - tune the threshold to try to match the above image!    
        hls_binary = hls_select(img, thresh=(200, 255))

        color_binary = np.dstack(( np.zeros_like(combined), combined, hls_binary)) *255
        return color_binary


    out = pipeline(imggg)
    img_float32 = np.float32(out)
    res = cv2.cvtColor(img_float32, cv2.COLOR_RGB2GRAY)

    ret, region_select = cv2.threshold(res, 10, 130, cv2.THRESH_BINARY)
    ysize = region_select.shape[0]
    xsize = region_select.shape[1]

     
    left_bottom = [175, 660]
    right_bottom = [1200, 660]
    #apex = [690, 420]
    left_up = [500,480]
    right_up = [790,480]


    # Fit lines (y=Ax+B) to identify the  3 sided region of interest
    # np.polyfit() returns the coefficients [A, B] of the fit
    fit_left = np.polyfit((left_bottom[0], left_up[0]), (left_bottom[1], left_up[1]), 1)
    fit_right = np.polyfit((right_bottom[0], right_up[0]), (right_bottom[1], right_up[1]), 1)
    fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)
    fit_up = np.polyfit((left_up[0], right_up[0]), (left_up[1], right_up[1]), 1)

    # Find the region inside the lines
    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    #print(XX)
    #print(YY)
    region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                        (YY > (XX*fit_right[0] + fit_right[1])) & \
                        (YY < (XX*fit_bottom[0] + fit_bottom[1])) & \
                        (YY > (XX*fit_up[0] + fit_up[1]))


    # Color pixels red which are inside the region of interest
    #region_select[region_thresholds] = [255, 0, 0]
    region_select[~region_thresholds] = 0
    binary_warped = region_select
    #plt.imshow(binary_warped)
    #plt.show()

    def find_lane_pixels(binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        #histogram = np.sum(binary_warped[450:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        #plt.imshow(out_img)
        #plt.show()
        #out_img = binary_warped
        #print(out_img.shape)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        #midpoint = np.int(750)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        #print(histogram.shape[0])

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 6
        # Set the width of the windows +/- margin
        margin = 200
        # Set minimum number of pixels found to recenter window
        minpix = 70

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        #print(nonzerox[50])
        #print(nonzerox[55])
        #print(len(nonzeroy))
        #print(len(nonzerox))

        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(np.int(nwindows/2)):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            ### TO-DO: Find the four below boundaries of the window ###
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            #cv2.rectangle(out_img,(win_xleft_low,win_y_low), (win_xleft_high,win_y_high),(0,255,0), 2) 
            #cv2.rectangle(out_img,(win_xright_low,win_y_low), (win_xright_high,win_y_high),(0,255,0), 2) 

            ### TO-DO: Identify the nonzero pixels in x and y within the window ###
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            ### TO-DO: If you found > minpix pixels, recenter next window ###
            ### (`right` or `leftx_current`) on their mean position ###
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds])) # Remove this when you add your function

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img, histogram

    def fit_polynomial(binary_warped):
        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img, histogram = find_lane_pixels(binary_warped)

        ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
        try:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
        except:
            left_fit = [1,1,0]
            right_fit = [1,1,0]

        # Generate x and y values for plotting
        ploty = np.linspace(binary_warped.shape[0]//2, binary_warped.shape[0], binary_warped.shape[0]//2 )
        #ploty = np.linspace(binary_warped.shape[0]//2, 600, 300)
        #ploty = np.linspace(500, binary_warped.shape[0], 220)
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [255, 0, 0]

        # Plots the left and right polynomials on the lane lines
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')

        draw_points_left = (np.asarray([left_fitx, ploty]).T).astype(np.int32)
        draw_points_right = (np.asarray([right_fitx, ploty]).T).astype(np.int32)

        cv2.polylines(imgg, [draw_points_left], False, (255,0,0), 5)
        cv2.polylines(imgg, [draw_points_right], False, (255,0,0), 5)

       
        return imgg, left_fit, right_fit, histogram
    imgg, left_fit, right_fit, histogram = fit_polynomial(binary_warped)
    return imgg

# out = video(imggg)
# plt.imshow(out)
# plt.show()

yellow_output = 'challenge_video_ans.mp4'
clip2 = VideoFileClip("Challenge_Videos/challenge_video.mp4")
white_clip = clip2.fl_image(video) #NOTE: this function expects color images!!
white_clip.write_videofile(yellow_output, audio=False)




