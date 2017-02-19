**Advanced Lane Finding Project**

The provided project is made of the following files:
[//]: # (File References)
[***main.py***]: is the files containing the main, that runs the entire algorithm in the test images rather then in the videos
[***CameraManager.py***]: is the class responsible for the camera management. It is used to calibrate the camera and to apply everuy kind of transformation to the images that involve transformations due to the camera
[***ImageManager.py***]: is the core manager of the algorithm. It manage all the work into images, from creation of the CameraManager, up to the call to the line detection function.
[***RoadImage.py***]: represent a road image class. It includes all the representation of the image used during the process, and is responsible to apply color/sobel filters to the original undistorted image 
[***Line.py***]: the line class is responsible for all the line detection steps. It not only detect lines from the bird eyes view of the road, but manage also the history of seen lines and manage the curvature/position calculation

---
###Writeup / README

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
  1. Apply HLS image thresholds over Saturation and Hue channels
  2. Apply sobel operator along x and y direction; 
    1. Combine sobel x and sobel y masks with magnitude thresholds
    2. Combine sobel x and sobel y masks with direction thresholds
    3. Combine the both filters above in a Bitwise OR manner
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
####2. Image filtering
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
**Sobel operator** is a very powerful operator to detect edges. Depending on the kernel size, it can detect sharper or stronger edges in the desired direction. Edges are computed convolving the kernel all along the image and computing a gradient value for each pixel of the image. Thresholding this gradient let us to choose which pixels are for us edges (an then lane lines).
Sobel application along x and y, can be combined in different ways. For this project:
1. I first combined the two direction application looking at the magnitude mask
2. Then I used the x and y application of the operator to compute a sobel direction mask
3. I combined the magnitude and the direction mask in a single sobel mask.  
The ImageManager is responsible to aply this filtering with the function *combine_sobel_filters()*. It first calls the RoadImage *apply_sobel_operator()* that runs the procedure of filters computation
```python
self.mag_thresh = (30,255)
self.sobel_kernel = 5
self.dir_thresh = (0.6, 1.2)

...

def apply_sobel_operator(self):

	sobel_x = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
	sobel_y = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
	self.abs_sobel_x = self.abs_sobel_thresh(sobel_x,"x")
	self.abs_sobel_y = self.abs_sobel_thresh(sobel_y,"y")
	self.mag_sobel = self.mag_thresh_mask(sobel_x,sobel_y)
	self.dir_sobel = self.dir_threshold(sobel_x,sobel_y)
```
and then combine the sobel filters application as:
```python
def combine_sobel_filter(self,image):
	sobel_combined = np.zeros_like(image.gray)
	#sobel_combined[((image.abs_sobel_x == 1) & (image.abs_sobel_y == 1)) | ((image.mag_sobel == 1) & (image.dir_sobel == 1))] = 1
	sobel_combined[((image.mag_sobel == 1) & (image.dir_sobel == 1))] = 1
	return sobel_combined
```
This bring in results like in the image below:
![alt tag](https://github.com/ciabo14/SelfDrivingCarND_AdvancedLaneFinding/blob/master/images/SobelFiltering.png)  
Finally, color and sobel masks are combined  in a Bitwise OR manner, leading at the following edge image:
![alt tag](https://github.com/ciabo14/SelfDrivingCarND_AdvancedLaneFinding/blob/master/images/HLS_SobelMasksApplication
.png)
Below The code that describe all the filtering process executed by the ImageManager class:
```python
def filter_image_for_line_detection(self):
	self.img.apply_sobel_operator()
	self.img.set_sobel_combined(self.combine_sobel_filter(self.img))
	self.img.set_color_line_mask(self.combine_color_filters(self.img))
	self.img.set_lane_lines_mask(cv2.bitwise_or(self.img.sobel_combined,self.img.color_line_mask))
```
####3. Perspective transformation
Perspective transformation to bird eyes perspective is very useful to limit the section of the image where to focus the interest and, more important, to work on an image without prospective distortion (parallel lines appears parallel in the bird eyes image and not convergent in the vanishing Point).
For this purpose the *cv2.warpPerspective* was computed it the edge image, selecting as source and destination corners of the rectangle the following corners:
Below The code that describe all the filtering process executed by the ImageManager class:
```python
height_section = np.uint(img_size[1]/2)

top_left_coordinate = height_section - .107*np.uint(img_size[1]/2)
top_right_coordinate = height_section + .113*np.uint(img_size[1]/2)
bottom_left_coordinate = height_section - .7*np.uint(img_size[1]/2)
bottom_right_coordinate = height_section + .75*np.uint(img_size[1]/2)

top_margin = np.uint(img_size[0]/1.55)
bottom_margin = np.uint(img_size[0])

src_corners = np.float32([[bottom_left_coordinate,bottom_margin], #bottomLeft
	[bottom_right_coordinate,bottom_margin],	#bottomRight
    [top_right_coordinate,top_margin], #topRight
    [top_left_coordinate,top_margin]]) #topLeft
    
"""
| Source        | Destination   | 
|:-------------:|:-------------:| 
| 192, 720      | 200, 720      | 
| 1120, 720     | 1080, 720     |
| 712, 464      | 1080, 0       |
| 571, 464      | 200, 0        |
 """    
```
I also tryed with different fixed corners. However this bring very similar results.
```python
src_corners = np.array([[585, 460], [203, 720], [1127, 720], [695, 460]]).astype(np.float32)
			dst_corners = np.array([[320, 0], [320, 720], [960, 720], [960, 0]]).astype(np.float32)
```
![alt tag](https://github.com/ciabo14/SelfDrivingCarND_AdvancedLaneFinding/blob/master/images/PerspectiveTransformation.png)
![alt tag](https://github.com/ciabo14/SelfDrivingCarND_AdvancedLaneFinding/blob/master/images/PerspectiveTransformation-Filtered.png)
####4. Lane lines detection

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

