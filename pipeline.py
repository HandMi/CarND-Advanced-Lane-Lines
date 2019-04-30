import numpy as np
import cv2
import matplotlib.image as mpimg
import parameters as params
import lane


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)   

def argmedian(hist):
    total_sum = np.sum(hist)
    cumulative_sum = np.cumsum(hist)
    argmed = np.argmin(np.abs(cumulative_sum-total_sum//2))
    return argmed

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = argmedian(histogram[:midpoint])    
    rightx_base = argmedian(histogram[midpoint:])+midpoint
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//params.nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Step through the windows one by one
    for window in range(params.nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - params.window_margin 
        win_xleft_high = leftx_current + params.window_margin 
        win_xright_low = rightx_current - params.window_margin 
        win_xright_high = rightx_current + params.window_margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > params.window_minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > params.window_minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
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
    return leftx, lefty, rightx, righty, out_img




class lane_detection:
    
    def __init__(self, img, objpoints, imgpoints):
       self.reset_lanes()
       ret, self.undist_mtx, self.dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)

    def reset_lanes(self):
       self.lane_left = lane.lane()
       self.lane_right = lane.lane()

    def calibrate_warp(self, vertices, destination):
       self.warp_matrix = cv2.getPerspectiveTransform(vertices, destination)

    def undistort(self,img):
       undist = cv2.undistort(img, self.undist_mtx, self.dist_coeffs, None, self.undist_mtx)
       return undist

    def apply_sobel(self, img, sobel_kernel=3, sob_thresh=(0, 255), mag_thresh=(0, 255), dir_thresh=(0, np.pi/2)):
        # Take both Sobel x and y gradients
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        
        # Compute absolute values
        abs_sobel_x = np.absolute(sobel_x)
        abs_sobel_y = np.absolute(sobel_y)
        
        scaled_sobel_x = np.uint8(255*abs_sobel_x/np.max(abs_sobel_x))
        scaled_sobel_y = np.uint8(255*abs_sobel_y/np.max(abs_sobel_y))
        
        ## Treshold in x-direction
        binary_x = np.zeros_like(scaled_sobel_x)
        binary_x[(scaled_sobel_x >= sob_thresh[0]) & (scaled_sobel_x <= sob_thresh[1])] = 1
     
        ## Treshold in y-direction
        binary_y = np.zeros_like(scaled_sobel_y)
        binary_y[(scaled_sobel_y >= sob_thresh[0]) & (scaled_sobel_y <= sob_thresh[1])] = 1
     
        ## Magnitude threshold
        # Calculate the gradient magnitude
        abs_grad = np.sqrt(sobel_x**2 + sobel_y**2)
        # Rescale to 8 bit
        scaled_grad= np.uint8(255*abs_grad/np.max(abs_grad))
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_mag = np.zeros_like(scaled_grad)
        binary_mag[(scaled_grad >= mag_thresh[0]) & (scaled_grad <= mag_thresh[1])] = 1
        
        ## Direction threshold
        grad_dir = np.arctan2(abs_sobel_y, abs_sobel_x)
        
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_dir = np.zeros_like(grad_dir)
        binary_dir[(grad_dir >= dir_thresh[0]) & (grad_dir <= dir_thresh[1])] = 1
        
        
        # Combine all results
        bin_combined = np.zeros_like(binary_x)
        bin_combined[((binary_x == 1) & (binary_y == 1)) | ((binary_mag == 1) & (binary_dir == 1))] = 1
        return bin_combined

    def sobel_filter(self, img):
        ret = self.apply_sobel(img, params.SOBEL_KERNEL_SIZE, params.SOBEL_THRESHOLD, params.MAGNITUDE_THRESHOLD, params.DIRECTIONAL_THRESHOLD)
        return ret*255

    def grayscale(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        ret = cv2.addWeighted(hls[:,:,2],0.5,img[:,:,0],0.5,0)
        return hls[:,:,2]

    def hsv_color_filter(self, img):
       hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
       image_yellow = cv2.inRange(hsv, params.dark_yellow, params.light_yellow)
       image_white = cv2.inRange(hsv, params.dark_white, params.light_white)
       ret_image = image_white+image_yellow
       return ret_image


    def fit_polynomial_windows(self,binary_warped):
        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
        self.lane_left.update_fit(leftx, lefty)
        self.lane_right.update_fit(rightx, righty)

    def search_around_polynomial(self,binary_warped,lane):
        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        ### Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###
        fit = lane.best_fit
        lane_inds = ((nonzerox > (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + 
                        fit[2] - params.window_margin)) & (nonzerox < (fit[0]*(nonzeroy**2) + 
                        fit[1]*nonzeroy + fit[2] + params.window_margin )))
        
        # Again, extract left and right line pixel positions
        lanex = nonzerox[lane_inds]
        laney = nonzeroy[lane_inds]
        
        # Fit new polynomials
        if(len(lanex)>0):
            lane.update_fit(lanex,laney)
        else:
            lane.error_frames += 1

    def visualize_polynomials(self, binary_warped):    
        result =  np.zeros((binary_warped.shape[0],binary_warped.shape[1],3), np.uint8)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

        ### Calc both polynomials using ploty, left_fit and right_fit ###
        if(self.lane_left.degradation>0):
            left_fitx = self.lane_left.best_fit[0]*ploty**2 + self.lane_left.best_fit[1]*ploty + self.lane_left.best_fit[2]
            left_fit_pts=np.transpose(np.vstack([left_fitx, ploty]))
            cv2.polylines(result,np.int_([left_fit_pts]),False,(255,0,0),thickness=8)
        if(self.lane_right.degradation>0):    
            right_fitx = self.lane_right.best_fit[0]*ploty**2 + self.lane_right.best_fit[1]*ploty + self.lane_right.best_fit[2]
            right_fit_pts=np.transpose(np.vstack([right_fitx, ploty]))
            cv2.polylines(result,np.int_([right_fit_pts]),False,(255,0,0),thickness=8)
        if(self.lane_left.degradation>0) and (self.lane_right.degradation>0):
            pts = np.hstack((np.array([left_fit_pts]), np.array([np.flipud(right_fit_pts)])))
            cv2.fillPoly(result, np.int_([pts]), (0,100, 0))
        # Plot the polynomial lines onto the image
        return result

    def display_data(self, lane):
        img = np.zeros((200,500,3), np.uint8)

        if (lane.degradation == 0):
            color_box = (188,0,0)
        elif (lane.degradation == 1):
            color_box = (217,217,0)
        else:
            color_box = (60,179,113)
        img[2:-2,2:-2]=color_box
        txt1 = "Curvature Radius: " + str(round(lane.radius_of_curvature,1)) + " m"
        cv2.putText(img,txt1,(10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 1)
        txt2 = "Position: " + str(round(lane.line_base_pos,2)) + " m"
        cv2.putText(img,txt2,(10, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 1)
        txt3 = "Error Frames: " + str(lane.error_frames)
        cv2.putText(img,txt3,(10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 1)
        return img
    

    def initial_pipeline(self, img):
        warped_undist=cv2.warpPerspective(img, self.warp_matrix, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        gray = self.grayscale(img)
        blur = gaussian_blur(gray,3)
        sobel_filtered = self.sobel_filter(blur)
        hsv_color_mask = self.hsv_color_filter(img)
        filtered = cv2.addWeighted(sobel_filtered,0.5,hsv_color_mask,0.5,0)
        warped = cv2.warpPerspective(filtered, self.warp_matrix, filtered.shape[1::-1], flags=cv2.INTER_LINEAR)
        self.fit_polynomial_windows(warped)
        return warped

    def image_pipeline(self, img):
        undist = self.undistort(img)
        warped_undist=cv2.warpPerspective(undist, self.warp_matrix, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        gray = self.grayscale(undist)
        blur = gaussian_blur(gray,3)
        sobel_filtered = self.sobel_filter(blur)
        hsv_color_mask = self.hsv_color_filter(undist)
        filtered = cv2.addWeighted(sobel_filtered,0.5,hsv_color_mask,0.5,0)
        warped = cv2.warpPerspective(filtered, self.warp_matrix, filtered.shape[1::-1], flags=cv2.INTER_LINEAR)
        if (self.lane_left.line_base_pos>0.0):
            self.lane_left.degradation = 0
        if (self.lane_right.line_base_pos<0.0):
            self.lane_right.degradation = 0
        if (self.lane_left.degradation < 2 or self.lane_right.degradation < 2 ):
            self.fit_polynomial_windows(warped)
        if (self.lane_left.degradation > 1):
            self.search_around_polynomial(warped,self.lane_left)
        if (self.lane_right.degradation > 1):
            self.search_around_polynomial(warped,self.lane_right)
        poly_img = self.visualize_polynomials(warped)
        img_reverse = cv2.warpPerspective(poly_img, self.warp_matrix, poly_img.shape[1::-1], flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)

        ## filtered_colored = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB);
        ##result = cv2.addWeighted(filtered_colored,1 , img_reverse, 0.5,0)
        result = cv2.addWeighted(undist,1 , img_reverse, 0.5,0)
        left_text = self.display_data(self.lane_left)
        right_text = self.display_data(self.lane_right)
        box_offset = 20
        overlay_left = cv2.addWeighted(result[box_offset:box_offset+left_text.shape[0],box_offset:box_offset+left_text.shape[1]],0.5 , left_text, 0.7,0)
        overlay_right = cv2.addWeighted(result[box_offset:box_offset+right_text.shape[0],result.shape[1]-box_offset-right_text.shape[1]:result.shape[1]-box_offset],0.5 , right_text, 0.7,0)
        result[box_offset:box_offset+left_text.shape[0],box_offset:box_offset+left_text.shape[1]] = overlay_left
        result[box_offset:box_offset+right_text.shape[0],result.shape[1]-box_offset-right_text.shape[1]:result.shape[1]-box_offset] = overlay_right
        ##self.compute_curvature()
        return result