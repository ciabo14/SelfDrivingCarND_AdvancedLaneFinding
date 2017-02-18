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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

In order to detect correctly the lanes, because cameras uses lenses and are not a pinhole camera, the camera need to be calibrated in order to obtain a transformation of the images like they were captured using an ideal model.

####1. Computation of camera matrix

Given a point in the 3D image P(X,Y,Z), this point is transformed by the camera into a 2D point p(x,y) via the camera Matrix.

![equation](http://latex.codecogs.com/gif.latex?P%5Csimp)

However, lenses introduce  `radial ` and  `tangetial ` distortion, a correction of these distortion are required in order to have an image as taken from the pinhole camera.

**Camera matrix** and **distortion coefficients** can be discovered taking images of a known pattern in different position together with the pattern in a non *distorted* form, and found the transformation that map the distorted images in the known pattern. 

The process of discovering this elements is the called **camera calibration**.

The Camera in entirely managed in the `CameraManager.py` by the *CameraManager* class. 
If the camera was never calibrated, the manager calibrate the camera with the support of the chess pattern images located in the folder "./camera_cal" and then save the camera matrix and the distortion coefficients into a file `calibration.p` in the same file. Instead, if the calibration was already executed, the manager load the file and the relative items.

During the calibration we first found the internal corners of the chess pattern (known as number **(9,6)**) for each of the image provided.
```python
	gray_image = cv2.cvtColor(original_img,cv2.COLOR_BGR2GRAY)ret, corners = cv2.findChessboardCorners(gray_image, self.grid_shape, None)
```
![alt tag](https://github.com/ciabo14/SelfDrivingCarND_AdvancedLaneFinding/blob/master/images/FoundCorners_1.png)
![alt tag](https://github.com/ciabo14/SelfDrivingCarND_AdvancedLaneFinding/blob/master/images/FoundCorners_2.png)

If the corners are found these are appened to the set of points for all the images 
```python
if ret:
	imgpoints.append(corners)
```
and finally the camera matrix and the distortion coefficients are computed using the cv2 function
```python
ret, self.cam_mtx, self.cal_dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_image.shape[::-1], None, None)
```
*self.cam_mtx*  and *self.cal_dist* are then used to compute the undistorted image for each image analized

```python
cv2.undistort(img, self.cam_mtx, self.cal_dist, None, self.cam_mtx)
```

![alt tag](https://github.com/ciabo14/SelfDrivingCarND_AdvancedLaneFinding/blob/master/images/CameraUndistortion.png)

###Pipeline (single images)

Once the camera is calibrated (ad this is executed only once whe I started working on the project), each image (single or from a frame), can be elaborated in order to detect lane.
As the graph below shows, the images are processed as follow:

1. Undistort the image using the camera matrix and the undistortion parameters as shown above
2. Apply image analisys *filters* in order to keep almost only pixels from lane lines:
  1. Apply sobel operator along x and y direction; 
    1. Combine sobel x and sobel y masks with magnitude thresholds
    2. Combine sobel x and sobel y masks with direction thresholds
    3. Combine the both filters above in a Bitwise OR manner
  2. Apply HLS image thresholds over Saturation and Hue channels
3. Combine filter both from Sobel application and color channels thresholding in a Bitwise OR manner.
4. Apply perspective transformation to select lane section only
5. Elaborate the image obtained from the perspective transformation application as follow:
  1. Compute lane lines pixels using slinding windows and the histogram, or using the last recognised lane lines
  2. Evaluate the found lines from a plausability point of view
  3. Compute lines polynomial fit
  4. Compute lanes curvature and position
6. Transform the binary mask with the found lane lines back in the original perspective 
![alt tag](https://github.com/ciabo14/SelfDrivingCarND_AdvancedLaneFinding/blob/master/images/Pipeline_Diagram.png)  

####1. Correction of the image distortion.

Test images, or frames from the video, were undistorted using the CameraManager function *undistort_image()*. This function only take an image as input, and return the undistorted image 

```python
def undistort_image(self,img):
	return cv2.undistort(img, self.cam_mtx, self.cal_dist, None, self.cam_mtx)
```
Below an example of how an image appears after the distortion correction.

![alt tag](https://github.com/ciabo14/SelfDrivingCarND_AdvancedLaneFinding/blob/master/images/TestImageUndistortion.png)  

####2. Image filter application

Lines are elements in the image recognized by drivers because of their shape, color and position/direction. Moreover, lines are detected in different light conditions. Good Light, presence of shadows ecc. 

In order to recognize lines as a human driver does, the same recognition process is eligible for machines. 

For this reasons two different filtering to the images were applied in order to discover lane lines:

1. Color filtering
2. Shape and position filtering using Sobel operator

The first intuitive way to use color filtering, is to filter white and yellow colors in the image and discard all the other colors. However, using the RGB color space, we can develop a filter correlated to the light in the image (enviroment). 
However, moving to a different space we can capture lines indipendetly form the the light (day light, artificial lights or even shadows). 
This is the case of the HLS color space. 
Filtering the image in the Hue and Saturation channels, we are able to remove the majority of the pixels keeping lane lines even in shadows conditions.

I applyed a threshold to the Hue and Saturation channels defined as below:

```python
mask_HLS = {"low_thresholds":np.array([ 0,  0,  100]), "high_thresholds":np.array([ 100, 255, 255])}
...

def color_select(self, mask_dict, img_representation = "RGB"):
	mask = np.zeros_like(self.gray)
	if(img_representation == "RGB"):
		mask[((self.undistorted_image[:,:,0] >= mask_dict["low_thresholds"][0]) & (self.undistorted_image[:,:,0] <= mask_dict["high_thresholds"][0])) & ((self.undistorted_image[:,:,1] >= mask_dict["low_thresholds"][1]) & (self.undistorted_image[:,:,1] <= mask_dict["high_thresholds"][1]))& 
	
	if(img_representation == "HLS"):
		mask[((self.HLS_image[:,:,0] >= mask_dict["low_thresholds"][0]) & (self.HLS_image[:,:,0] <= mask_dict["high_thresholds"][0]))&((self.HLS_image[:,:,1] >= mask_dict["low_thresholds"][1]) & (self.HLS_image[:,:,1] <= mask_dict["high_thresholds"][1]))&((self.HLS_image[:,:,2] >= mask_dict["low_thresholds"][2]) & (self.HLS_image[:,:,2] <= mask_dict["high_thresholds"][2]))] = 1	

	return mask
```

Below you can find the application of the HLS color filtering to one of the test images.

![alt tag](https://github.com/ciabo14/SelfDrivingCarND_AdvancedLaneFinding/blob/master/images/HLSFIltering.png)  



Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

