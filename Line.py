import numpy as np
import cv2	
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import deque

class Line:

	def __init__(self):
		
		# used to show some images for debug purposes
		self.DEBUG = False
		# Set number of windows into which split the image
		self.nwindows = 9
		# Set the width of the windows +/- margin
		self.margin = 80
		# Set minimum number of pixels found to recenter window
		self.minpix = 50
		# Define the number of frames from which use the found pixels for left and right lines detection
		self.last_frame_used = 5
		# Define the number of minumum pixels to sign the line as detected 
		self.min_pix_line_identification = 6000

		self.use_lines_history = True
		self.max_curvature_deviation = 2000

		#Define the parameters used to smooth the polinomial fit parameters and the computed radius using last history
		self.fit_alpha = 0.2
		self.rof_alpha = 0.05
		
		self.first_frame = True
		self.left_line_missing = 0
		self.right_line_missing = 0

		#distance in meters of vehicle center from the line
		
		#last polynomial coefficients found
		self.left_fit = None
		self.right_fit = None
		
		#last frame detected pixels
		self.last_frames_left_x = []
		self.last_frames_left_y = []
		
		self.last_frames_right_x = []
		self.last_frames_right_y = []
		
		self.last_left_x = []
		self.last_left_y = []
		
		self.last_right_x = []
		self.last_right_y = []

		self.last_frame_left_curvature = []
		self.last_frame_right_curvature = []

		self.mean_curvature = None

		self.offset = None

		self.detection = "First Frame"

	def detect_lines(self, binary_warped):

		left_pixel_positions_x,left_pixel_positions_y,right_pixel_positions_x,right_pixel_positions_y = self.pixels_detection(binary_warped)

		self.manage_detected_pixels(left_pixel_positions_x, left_pixel_positions_y, right_pixel_positions_x, right_pixel_positions_y)
		self.fit_found_lanes(binary_warped)
		self.manage_curvature()

		self.find_offset()	
			
		if self.DEBUG:
			self.detection = "{} -- FF: {} -- missed lane: {} - {} -- num mix: {} - {}".format(self.detection,self.first_frame, self.left_line_missing, 
				self.right_line_missing,len(self.left_x), len(self.right_x))
		
	def pixels_detection(self,binary_warped):

		out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
		left_lane_inds = []
		right_lane_inds = []

		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		
		# In case first frame or in last 5 frames I did not found a left or right line
		if(self.first_frame or self.left_line_missing > 5 or self.right_line_missing > 5):
			
			if self.DEBUG:
				self.detection = "SW detection -- FF: {}".format(self.first_frame)
				
			self.left_line_missing = 0 
			self.right_line_missing = 0
			self.first_frame = False

			window_height = np.int(binary_warped.shape[0]/self.nwindows)
			leftx_current,rightx_current = self.scan_from_hist(binary_warped)
		
			for window in range(self.nwindows):
				
				# Identify window boundaries in x and y (and right and left)
				win_y_low = binary_warped.shape[0] - (window+1)*window_height
				win_y_high = binary_warped.shape[0] - window*window_height
				win_xleft_low = leftx_current - self.margin
				win_xleft_high = leftx_current + self.margin
				win_xright_low = rightx_current - self.margin
				win_xright_high = rightx_current + self.margin
				
				if(self.DEBUG):
					cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
					cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
				# Identify the nonzero pixels in x and y within the window
				good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
				good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
				# Append these indices to the lists
				left_lane_inds.append(good_left_inds)
				right_lane_inds.append(good_right_inds)
				
				# If you found > minpix pixels, recenter next window on their mean position
				if len(good_left_inds) > self.minpix:
				    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
				if len(good_right_inds) > self.minpix:        
				    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

			# Concatenate the arrays of indices
			left_lane_inds = np.concatenate(left_lane_inds)
			right_lane_inds = np.concatenate(right_lane_inds)

		# Select pixels in the lane starting from the last lines detected
		else:

			left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] - self.margin)) & 
				(nonzerox < (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] + self.margin))) 
			right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] - self.margin)) & 
				(nonzerox < (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] + self.margin)))  
			if self.DEBUG:
				self.detection = "Old lines detection FF: {}".format(self.first_frame)

		#Extract left and right line pixel positions
		left_pixel_positions_x = nonzerox[left_lane_inds]
		left_pixel_positions_y = nonzeroy[left_lane_inds] 
		right_pixel_positions_x = nonzerox[right_lane_inds]
		right_pixel_positions_y = nonzeroy[right_lane_inds] 

		return left_pixel_positions_x,left_pixel_positions_y,right_pixel_positions_x,right_pixel_positions_y

	def scan_from_hist(self, binary_warped):

		histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
		
		# Define the points along x axis from which start with the lane lines search
		midpoint = np.int(histogram.shape[0]/2)
		leftx_base = np.argmax(histogram[:midpoint])
		rightx_base = np.argmax(histogram[midpoint:]) + midpoint

		return leftx_base, rightx_base 

	### This is the function used to populate the queue last frame queue with the detected pixels for right and left lines
	def manage_detected_pixels(self, left_x, left_y, right_x, right_y):
	
		self.left_x = left_x
		self.left_y = left_y
		self.right_x = right_x
		self.right_y = right_y

		if self.use_lines_history:
			num_frames = len(self.last_frames_left_x)
			
			if(num_frames == 0):
				self.last_frames_left_x.append(left_x)
				self.last_frames_left_y.append(left_y)
				self.last_frames_right_x.append(right_x)
				self.last_frames_right_y.append(right_y)				
			else:
				if num_frames >= self.last_frame_used:
					del(self.last_frames_left_x[0])
					del(self.last_frames_left_y[0])
					del(self.last_frames_right_x[0])
					del(self.last_frames_right_y[0])
					
				if len(left_x) > self.min_pix_line_identification:
					self.last_frames_left_x.append(left_x)
					self.last_frames_left_y.append(left_y)
					self.left_line_missing = 0
				else:
					self.last_frames_left_x.append(self.last_left_x)
					self.last_frames_left_y.append(self.last_left_y)
					self.left_line_missing += 1
				if len(right_x) > self.min_pix_line_identification:
					self.last_frames_right_x.append(right_x)
					self.last_frames_right_y.append(right_y)
					self.right_line_missing = 0
				else:
					self.last_frames_right_x.append(self.last_right_x)
					self.last_frames_right_y.append(self.last_right_y)
					self.right_line_missing += 1

		else:
			if self.last_left_x != None:
				if len(left_x) < self.min_pix_line_identification:
					self.left_x = self.last_left_x
					self.left_y = self.last_left_y
				if len(right_x) < self.min_pix_line_identification:
					self.right_x = self.last_right_x
					self.right_y = self.last_right_y
		
		self.set_last_xy(left_x, left_y, right_x, right_y)

	def set_last_xy(self, left_x, left_y, right_x, right_y):
		self.last_left_x = left_x
		self.last_left_y = left_y
		self.last_right_x = right_x
		self.last_right_y = right_y

	def fit_found_lanes(self, binary_warped):
		
		# Fit a second order polynomial to each
		if self.use_lines_history:
			current_left_x = [item for sublist in self.last_frames_left_x for item in sublist]
			current_left_y = [item for sublist in self.last_frames_left_y for item in sublist]
			current_right_x = [item for sublist in self.last_frames_right_x for item in sublist]
			current_right_y = [item for sublist in self.last_frames_right_y for item in sublist]
		else:
			current_left_x = self.left_x
			current_left_y = self.left_y
			current_right_x = self.right_x
			current_right_y = self.right_y

		self.ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
		
		l_fit = np.polyfit(current_left_y, current_left_x, 2)
		r_fit = np.polyfit(current_right_y, current_right_x, 2)
		
		if(self.use_lines_history):
			self.left_fit = l_fit
			self.right_fit = r_fit
		else:
			self.compute_smoothed_poly(l_fit, r_fit)

		self.left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
		self.right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]

		if(self.DEBUG):
			pass
			#plt.plot(self.left_fitx, self.ploty, color='green', linewidth=3)
			#plt.plot(self.right_fitx, self.ploty, color='green', linewidth=3)
			#plt.gca().invert_yaxis() # to visualize as we do the images
			#plt.show()
		if(self.DEBUG):
			pass
			#self.plot_image_lines_windows(out_img, nonzeroy, nonzerox, left_lane_inds, right_lane_inds, left_fitx, right_fitx, ploty)
			#self.plot_image_lines(binary_warped,left_pixel_positions_x, left_pixel_positions_y, right_pixel_positions_x, right_pixel_positions_y,left_fitx, right_fitx, ploty)


	def manage_curvature(self):


		l_curvature = self.estimate_Rof("l")
		r_curvature = self.estimate_Rof("r")
		
		if self.use_lines_history:
			
			left_mean = np.mean(self.last_frame_left_curvature)
			right_mean = np.mean(self.last_frame_right_curvature)

			num_frames = len(self.last_frame_left_curvature)
			tmp_curvature = 0

			if num_frames == 0:
				self.last_frame_left_curvature.append(l_curvature)
				self.last_frame_right_curvature.append(r_curvature)
				self.mean_curvature = np.mean([l_curvature,r_curvature])
			else:
				if num_frames >= self.last_frame_used:
					del(self.last_frame_left_curvature[0])
					del(self.last_frame_right_curvature[0])
					
				if left_mean + self.max_curvature_deviation > l_curvature > left_mean - self.max_curvature_deviation:
					self.last_frame_left_curvature.append(l_curvature)
				else:
					self.last_frame_left_curvature.append(left_mean)
					l_curvature = left_mean

				if right_mean + self.max_curvature_deviation > r_curvature > right_mean - self.max_curvature_deviation:
					self.last_frame_right_curvature.append(r_curvature)
				else:
					self.last_frame_right_curvature.append(right_mean)
					r_curvature = right_mean

			self.mean_curvature = np.mean([l_curvature,r_curvature])
		
		else:
			self.compute_smoothed_curvature(l_curvature, r_curvature)

	def estimate_Rof(self, line):
	    yscale = 30 / 720 # Real world metres per y pixel
	    xscale = 3.7 / 700 # Real world metres per x pixel
	    
	    # Fit new polynomial
	    if line == "l":
	    	fit_cr = np.polyfit(self.ploty * yscale, self.left_fitx * xscale, 2)
	    if line == "r":
	    	fit_cr = np.polyfit(self.ploty * yscale, self.right_fitx * xscale, 2)

	    # Calculate curve radius
	    curverad = ((1 + (2 * fit_cr[0] * np.max(self.ploty) * yscale + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])

	    return curverad

	def compute_smoothed_poly(self,l_fit, r_fit):
	
		# Update polynomials using weighted average with last frame
		if self.left_fit is None:
			# If first frame, initialise buffer
			self.left_fit = l_fit
			self.right_fit = r_fit
		else:
			# Otherwise, update buffer
			l_poly = (1 - self.fit_alpha) * self.left_fit + self.fit_alpha * l_fit
			r_poly = (1 - self.fit_alpha) * self.right_fit + self.fit_alpha * r_fit
			self.left_fit = l_fit
			self.right_fit = r_fit
	
	def compute_smoothed_curvature(self,l_curvature, r_curvature):
		
		# Get mean of curvatures
		mean_curvature = np.mean([l_curvature, r_curvature])
		# Update curvature using weighted average with last frame
		if self.mean_curvature is None:
			self.mean_curvature = mean_curvature
		else:
			self.mean_curvature = (1 - self.rof_alpha) * self.mean_curvature + self.rof_alpha * mean_curvature

	# Find the offset of the car and the base of the lane lines
	def find_offset(self):
		lane_width = 3.7  # metres
		h = 720  # height of image (index of image bottom)
		w = 1280 # width of image
		
		# Find the bottom pixel of the lane lines
		l_px = self.left_fit[0] * h ** 2 + self.left_fit[1] * h + self.left_fit[2]
		r_px = self.right_fit[0] * h ** 2 + self.right_fit[1] * h + self.right_fit[2]
		
		# Find the number of pixels per real metre
		scale = lane_width / np.abs(l_px - r_px)
		
		# Find the midpoint
		midpoint = np.mean([l_px, r_px])
		
		# Find the offset from the centre of the frame, and then multiply by scale
		self.offset = (w/2 - midpoint) * scale

	def plot_image_lines_windows(self, out_img, left_pixel_positions_x, left_pixel_positions_y, right_pixel_positions_x, right_pixel_positions_y, right_fitx, ploty):

		out_img[left_pixel_positions_y, left_pixel_positions_x] = [255, 0, 0]
		out_img[right_pixel_positions_y, right_pixel_positions_x] = [0, 0, 255]
		plt.imshow(out_img)
		plt.plot(left_fitx, ploty, color='yellow')
		plt.plot(right_fitx, ploty, color='yellow')
		plt.xlim(0, 1280)
		plt.ylim(720, 0)
		plt.show()

	def plot_image_lines(self,binary_warped,left_pixel_positions_x, left_pixel_positions_y, right_pixel_positions_x, right_pixel_positions_y,left_fitx, right_fitx, ploty):
		# Create an image to draw on and an image to show the selection window
		out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
		window_img = np.zeros_like(out_img)
		# Color in left and right line pixels
		#out_img[left_pixel_positions_y, left_pixel_positions_x] = [255, 0, 0]
		#out_img[right_pixel_positions_y, right_pixel_positions_x] = [0, 0, 255]

		# Generate a polygon to illustrate the search window area
		# And recast the x and y points into usable format for cv2.fillPoly()
		left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-self.margin, ploty]))])
		left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+self.margin, ploty])))])
		left_line_pts = np.hstack((left_line_window1, left_line_window2))
		right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-self.margin, ploty]))])
		right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+self.margin, ploty])))])
		right_line_pts = np.hstack((right_line_window1, right_line_window2))

		# Draw the lane onto the warped blank image
		cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
		cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
		result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

		plt.imshow(result)
		plt.plot(left_fitx, ploty, color='red')
		plt.plot(right_fitx, ploty, color='yellow')
		plt.xlim(0, 1280)
		plt.ylim(720, 0)
		plt.show()

		self.lane_lines_image = result
