import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
class RoadImage:


	def __init__(self, original_image, undistorted_image):
		#self.sobel_thresh = (25,200) #UDACITY
		self.sobel_thresh = (50,200)  #MY THS
		self.mag_thresh = (30,255)
		self.sobel_kernel = 5
		self.dir_thresh = (0.6, 1.2)
		self.s_chan_thresh = (90, 255)
		
		self.rgb = cv2.GaussianBlur(original_image,(5,5),0)#original_image
		self.undistorted_image = cv2.GaussianBlur(undistorted_image,(5,5),0)#undistorted_image
		self.gray = cv2.cvtColor(self.undistorted_image,cv2.COLOR_RGB2GRAY)
		self.HLS_image = cv2.cvtColor(self.undistorted_image,cv2.COLOR_RGB2HLS)

	def apply_sobel_operator(self):

		sobel_x = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
		sobel_y = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
		self.abs_sobel_x = self.abs_sobel_thresh(sobel_x,"x")
		self.abs_sobel_y = self.abs_sobel_thresh(sobel_y,"y")
		self.mag_sobel = self.mag_thresh_mask(sobel_x,sobel_y)
		self.dir_sobel = self.dir_threshold(sobel_x,sobel_y)

	def set_bird_eye_image(self,image):
		self.bird_eye_image = image

	def set_bird_eye_image_rgb(self,image):
		self.bird_eye_image_rgb = image

	def set_sobel_combined(self,image):
		self.sobel_combined = image

	def set_color_line_mask(self,image):
		self.color_line_mask = image

	def set_lane_lines_mask(self,image):
		self.lane_lines_mask = image

	def abs_sobel_thresh(self, sobel_mask, orient='x'):

		abs_sobel = np.absolute(sobel_mask)

		scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))	

		binary_output = np.zeros_like(scaled_sobel)
		binary_output[(scaled_sobel >= self.sobel_thresh[0]) & (scaled_sobel <= self.sobel_thresh[1])] = 1
		
		return binary_output

	def mag_thresh_mask(self,sobel_mask_x,sobel_mask_y):

		# Calculate the gradient magnitude
		gradmag = np.sqrt(sobel_mask_x**2 + sobel_mask_y**2)
		# Rescale to 8 bit
		scale_factor = np.max(gradmag)/255 
		gradmag = (gradmag/scale_factor).astype(np.uint8) 
		# Create a binary image of ones where threshold is met, zeros otherwise
		mag_binary = np.zeros_like(gradmag)
		mag_binary[(gradmag >= self.mag_thresh[0]) & (gradmag <= self.mag_thresh[1])] = 1

		return mag_binary

	def dir_threshold(self,sobel_mask_x,sobel_mask_y):

		# Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
		gradient_dir = np.arctan2(np.absolute(sobel_mask_y),np.absolute(sobel_mask_x))
		# Create a binary mask where direction thresholds are met
		dir_binary = np.zeros_like(gradient_dir)
		dir_binary[(gradient_dir >= self.dir_thresh[0]) & (gradient_dir <= self.dir_thresh[1])] = 1
		# Return this mask as your binary_output image
		return dir_binary


   	# Apply a mask defined by mask_dict, to the color space defined by img_representation. The thresholds in the mask are defined for all the channels in the color space 
	def color_select(self, mask_dict, img_representation = "RGB"):

		mask = np.zeros_like(self.gray)
		if(img_representation == "RGB"):
			#mask = cv2.inRange(self.undistorted_image, mask_dict["low_thresholds"], mask_dict["high_thresholds"])
			mask[((self.undistorted_image[:,:,0] >= mask_dict["low_thresholds"][0]) & (self.undistorted_image[:,:,0] <= mask_dict["high_thresholds"][0]))&
			((self.undistorted_image[:,:,1] >= mask_dict["low_thresholds"][1]) & (self.undistorted_image[:,:,1] <= mask_dict["high_thresholds"][1]))&
			((self.undistorted_image[:,:,2] >= mask_dict["low_thresholds"][2]) & (self.undistorted_image[:,:,2] <= mask_dict["high_thresholds"][2]))] = 1
		if(img_representation == "HLS"):
			#mask = cv2.inRange(self.HLS_image, mask_dict["low_thresholds"], mask_dict["high_thresholds"])
			mask[((self.HLS_image[:,:,0] >= mask_dict["low_thresholds"][0]) & (self.HLS_image[:,:,0] <= mask_dict["high_thresholds"][0]))&
			((self.HLS_image[:,:,1] >= mask_dict["low_thresholds"][1]) & (self.HLS_image[:,:,1] <= mask_dict["high_thresholds"][1]))&
			((self.HLS_image[:,:,2] >= mask_dict["low_thresholds"][2]) & (self.HLS_image[:,:,2] <= mask_dict["high_thresholds"][2]))] = 1		

		return mask
   	
   	# Apply a mask in the color space defined by img_representation, to the channel channel only. The thresholds in the function definition are specified for the selected 
   	# channel only in the form (low_th, high_th) 
	def color_channel_select(self,img_representation = "RGB", channel = 0, thresholds = (-1,-1)):

		if(thresholds == (-1,-1)):
			thresholds = self.s_chan_thresh
		##### img_representation is a selection switch among RGB, HLS image version (HSV and gray are not considered in this particular case)  
		if(channel<0 or channel >2):
			raise Exception("The channel required is not an image channel") 
		
		binary_output = np.zeros_like(self.gray)
		if(img_representation == "RGB"):
			image_channel = self.undistorted_image[:,:,channel]
		if(img_representation == "HLS"):
			image_channel = self.HLS_image[:,:,channel]

		binary_output[(image_channel>thresholds[0]) & (image_channel<= thresholds[1])] = 1
		
		return binary_output
	