## Writeup

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistorted_output.png "Undistorted"
[image2]: ./examples/undistorted_test.png "Undistorted Test"
[image3]: ./examples/Sobel_and_color.png "Sobel and Color Images"
[image4]: ./examples/white.gif "White Filter"
[image5]: ./examples/yellow.gif "Yellow Filter"
[image6]: ./examples/filtered.png "Filtered Output"
[image7]: ./examples/warp1.png "Warp Input"
[image8]: ./examples/warp2.png "Warp Output"
[image9]: ./examples/warp3.png "Warp Output 2"
[image10]: ./examples/hist.png "Histogram"
[image11]: ./examples/windows.png "Sliding Window Search"
[image12]: ./examples/windows2.png "Polynomial Search"
[image13]: ./examples/pipeline.png "Finished pipeline"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---

### Camera Calibration

To correctly display the size and shapes of 3-dimensional objects we first have to take into account the distortion inherent to the camera's setting. We start by calculating the distortion coefficients by looking at images of pre-defined checkboard patterns.

To this end we define an equi-distant, planar 6x9 chessboard grid `objpoints`) which we associate with the actual corners of the chessboard grid in the images (`imgpoints`) obtained from OpenCV's `cv2.findChessboardCorners` function.
These arrays are then used as inputs for `cv2.calibrateCamera` which returns the distortion matrix. We then apply this correction to the test image using the `cv2.undistort()` function and obtain this result: 

![Distorted and Undistorted Chessboard][image1]

### Pipeline (single images)

Before we jump to the videos it makes sense to first create a concept pipeline running on single images.

#### 1. Distortion Corrected Image

First we apply the previously found calibration to our test images:
![Undistorted Test Image][image2]

#### 2. Create a binary image

Next we apply a series of image transforms to the undistorted images. 

### 2.1. Sobel transform

As proposed in the course material, we use a combination of [Sobel derivatives](https://en.wikipedia.org/wiki/Sobel_operator) in both x- and y-direction as well as thresholds based on magnitude and Sobel gradient direction. Pixels are marked if either both the x- and y-Sobel gradients exceed a certain threshold or if the Sobel gradient's direction is inside the cone where we would expect potential lane markings.

### 2.2. Color spaces

The Sobel filter applied to the usual grayscale image obtained from an RGB image gives good enough results to clearly identify most lane markings. However in certain situations with diffuse backgrounds (such as tree shados, dirt, snow) the Sobel transform will fail to detect lane lines that the human eye could identify by color alone. Therefore it makes sense to process the image through another color/saturation filter before applying Sobel. We will use a combination of the RGB-R channel, which performs well at identify white lane markings, as well as the HLS-S channel, which detects yellow lane markings very well because of the high saturation. Thus our main use cases are covered by this grayscale transformation:

```python
def grayscale(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    ret = cv2.addWeighted(hls[:,:,2],0.5,img[:,:,0],0.5,0)
    return hls[:,:,2]
```
The finished grayscale images will look like this:
![Grayscale Image][image3]

As we can see, the lane markings are clearly discernible. However, we will stack another color filtered image on top using the HSV-based filter from the first project:

### 2.3. color-specific filter

we define the following segments of the HSV color space and add this filter on top of the Sobel filter:

```python
dark_yellow = (18, 60, 90)
light_yellow = (35,255,255)
dark_white = (0, 0, 200)
light_white = (255,25,255)
```
![White][image4]
![Yellow][image5]

Our final filter will only detect pixels, which belong to an edge detected by the Sobel transformation and which are colored either white or yellow. This takes care of most unwanted lines we would pick up from shadows or lane boundaries which are not lane markings.

```python
    # Sobel filter the blurred grayscale image
    sobel_filtered = self.sobel_filter(blur)

    # HSV color mask on the original image
    hsv_color_mask = self.hsv_color_filter(undist)

    # weighted average of the two filters
    binary = np.zeros_like(sobel_filtered)
    binary[(hsv_color_mask >= 1) & (sobel_filtered >=1)] = 1
```
![Filtered Output][image6]

#### 3. Perspective Transform

Next we will use an image of a straight line to calibrate the perspective transform to a bird's eye view perspective. 
![Warp Input][image7]
The calibration of the lane detector is done in the ```calibrate_warp``` function:
```python
def calibrate_warp(self, vertices, destination):
    self.warp_matrix = cv2.getPerspectiveTransform(vertices, destination)
```
The vertices were manually read from the image since they heavily depend on the way the camera was mounted. The destination pixels were chosen to leave a 300 pixel margin on each side. Note that the source vertices were offset by 50 pixels from the bottom of the image because of the static shape of the car's hood.
```python
vertices = np.float32([[[287,670],[583,458],[701,458],[1031,670]]])
destination = np.float32([[[300,700],[300,20],[980,20],[980,700]]])
```
To verify that the perspective transform correctly warps parallel lane markings to parallel planar lines we should have a look at the result:

![Warp Output][image8]

and also an independent image:

![Warp Output2][image9]

The calibration seems to work reasonably well.

#### 4. Selecting Lane Pixels

As described in the lecture, a sliding window search is applied to the warped binary images. This means, we initially have to determine their approximate position by looking for peaks in the histogram in x-direction. Instead of using the maxima of the histogram as suggested in the lecture, we will instead use the median:

```python
def argmedian(hist):
    total_sum = np.sum(hist)
    cumulative_sum = np.cumsum(hist)
    argmed = np.argmin(np.abs(cumulative_sum-total_sum//2))
    return argmed
```
This does not make a big difference in our use cases:

![Histogram][image10]

For the actual sliding window search (```find_lane_pixels(binary_warped)``` in pipeline.py) we proceed as in the lecture. The first window will be centered at the median of the number of pixels. The next window will always be recentered depending on the average x positions of the detected pixels in the last window.

![Sliding Window Search][image11]

Once a fit is established, concurrent searches will not be done via sliding windows but instead we will look for pixels in the vicinity of the last fit. This is implemented in ```search_around_polynomial(self,binary_warped,lane)```

![Search around Polynomials][image12]

#### 5. Lane Class

In order to better encapsulate the lane fitting, it makes sense to add a separate class for the lane markings:
```python
class lane():
    def __init__(self):
        # was the line detected in the last iteration?
        self.degradation = 0 
        #difference in fit coefficients between last and new fits
        self.error_frames = 0
        # last n fits of the line
        self.lane_history = deque(maxlen=params.ERROR_FRAMES)
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #radius of curvature of the line in some units
        self.radius_of_curvature = 0.0
        #distance in meters of vehicle center from the line
        self.line_base_pos = 0.0
```
In ```degradation``` we would like to store the current state of the lane line (```0``` no fit, ```1``` unstable fit, ```2``` stable fit). This heavily depends on the number of ```error_frames```, i.e. the number of consecutive frames where we could not add a new fit to the ```lane_history``` because sanity checks were not passed or no lane pixels were found in the latest image. In the following we want to allow for 20 ```ERROR_FRAMES``` until the fit is reset. This is specified in ```parameters.py```.
When a new fit is added to ```lane_history``` the ```best_fit``` is update by simply averaging the coefficients. On every update also the radius of curvature and the distance of the lane to the ego position is calculated:
```python
def compute_curvature_and_position(fit):
    # convert fit parameters to meter
    A = fit[0]*mpx/(mpy**2)
    B = fit[1]*(mpx/mpy)
    C = fit[2]*mpx
    y_evals = np.linspace(0,y_base,11)
    # evaluate at the curvature along the curve
    curvature =  (2*A)/ (1 + (2*A*y_evals + B)**2)**1.5
    # take the average of the curvature along the curve and compute the reciprocal to get an approximation for the curvature
    radius_of_curvature = len(y_evals)/sum(curvature)
    # intersection of the fit with the image frame is the base position
    line_base_pos = A*y_base**2+B*y_base+C-params.CAMERA_X
    return radius_of_curvature, line_base_pos
```
where ```mpx, mpy``` are the meter to pixel conversion rates in x- and y-direction (based on an approximate lane width of 3.7 m and the length of dashed markings of 10 feet). The fit's parameters are first converted to meter, then the curvature is evaluated along the lane. The radius of curvature is then calculated as ```1/average_of_curvatures```. This gives a more robust estimate of the curvature than just evaluation at one of the end point.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

In ```test.py ``` the image pipeline is set up for a single image:

![Pipeline Image][image13]

Here the degradation is marked as 1, unstable, since there is only a single frame in the lane history.

---

### Pipeline (video)

#### 1. Final Video Output

This is the final output of the pipeline applied to the project_video.mp4:

[![Watch the project video](https://img.youtube.com/vi/MNDkkiJ_JOM/maxresdefault.jpg)](https://www.youtube.com/watch?v=MNDkkiJ_JOM)

and the challenge_video.mp4:

[![Watch the project video](https://img.youtube.com/vi/Ry53GTyE3xY/maxresdefault.jpg)](https://www.youtube.com/watch?v=Ry53GTyE3xY)
---

### Discussion

In the videos it can be seen that the pipeline works fairly well on highways with wide curves. However, a look at the harder challenge video reveals a number of unaddressed weaknesses:

[![Watch the project video](https://img.youtube.com/vi/9DLdS88xzF8/maxresdefault.jpg)](https://youtu.be/9DLdS88xzF8)

* The pipeline does not dynamically adopt to lighting changes
* Curves that are too narrow, even to the point that they exit the side of the frame, cannot be traced
* Hills and dips are not accounted for
* In its current state the pipeline splits the image in half in the sliding window search. If one of the two lanes crosses the middle pixels will be attributed to the other lanes leading to unpredictable fits.
* The video recording shows a static reflection of the dashboard which should somehow be filtered out.
* All in all there are not enough checks that should lead to a reset of the lane.
