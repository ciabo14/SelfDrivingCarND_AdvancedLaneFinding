from RoadImage import RoadImage
from Line import Line
import numpy as np
import cv2	

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import time

class ImageManager:

	def __init__(self, cm):
		self.cm = cm
		self.line = Line()
		self.last_time_stamp = int(round(time.time()*1000))

	def get_row_lane_lines():
		pass
	
	def analyze_time(self):

		t = int(round(time.time()*1000))
		print(t - self.last_time_stamp)
		self.last_time_stamp = t

	def find_lane_lines(self, image):
		#print("================")
		#self.analyze_time()

		self.img = RoadImage(image,self.cm.undistort_image(image))
		
		self.filter_image_for_line_detection()

		
		self.img.set_bird_eye_image(self.cm.perspective_transformation(self.img.lane_lines_mask))
		self.img.set_bird_eye_image_rgb(self.cm.perspective_transformation(self.img.undistorted_image))

	
		#self.analyze_time()

		self.line.detect_lines(self.img.bird_eye_image)

		#self.analyze_time()


		#print(self.line.detection)
		#detected_lane_lines = self.line.lane_lines_image
		result = self.drow_detected_lines()
		return result
		f, axes  = plt.subplots(2, 2, figsize=(15, 8))
		f.tight_layout()

		axes[0,0].imshow(self.img.rgb,cmap='gray')
		axes[0,0].set_title('Original RGB Image', fontsize=20)
		axes[0,1].imshow(self.img.lane_lines_mask,cmap='gray')
		axes[0,1].set_title('sobel combined Image', fontsize=20)


		axes[1,0].imshow(self.img.brid_eye_image_rgb,cmap='gray')
		axes[1,0].set_title('color filtered Image', fontsize=20)
		axes[1,1].imshow(self.img.brid_eye_image)
		axes[1,1].set_title('lane lines Image', fontsize=20)
		plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
		plt.show()

	def filter_image_for_line_detection(self):
		self.img.apply_sobel_operator()
		self.img.set_sobel_combined(self.combine_sobel_filter(self.img))
		self.img.set_color_line_mask(self.combine_color_filters(self.img))
		self.img.set_lane_lines_mask(cv2.bitwise_or(self.img.sobel_combined,self.img.color_line_mask))

	def combine_sobel_filter(self,image):
		sobel_combined = np.zeros_like(image.gray)
		#sobel_combined[((image.abs_sobel_x == 1) & (image.abs_sobel_y == 1)) | ((image.mag_sobel == 1) & (image.dir_sobel == 1))] = 1
		sobel_combined[((image.mag_sobel == 1) & (image.dir_sobel == 1))] = 1
		return sobel_combined

	def combine_color_filters(self,image):
		##### The yellow and white masks represents masks to be applied to HSV image 
		
		white_mask_RGB = {"low_thresholds":np.array([ 200, 150, 200]), "high_thresholds":np.array([ 255, 255, 255])}
		yellow_mask_RGB = {"low_thresholds":np.array([ 150, 150, 0]), "high_thresholds":np.array([ 255, 255, 125])}
		mask_HLS = {"low_thresholds":np.array([ 0,  0, 100]), "high_thresholds":np.array([ 100, 255, 255])}
		
		white_lines_RGB = image.color_select(white_mask_RGB,img_representation = "RGB")
		yellow_lines_RGB = image.color_select(yellow_mask_RGB,img_representation = "RGB")

		mask_RGB = np.zeros_like(white_lines_RGB) 
		#mask_RGB = cv2.bitwise_or(white_lines_RGB,yellow_lines_RGB)

		white_lines_HLS = image.color_select(mask_HLS,img_representation="HLS")

		#white_lines_HLS = image.color_channel_select(img_representation = "HLS", channel = 2, thresholds = (90,255))

		#return white_lines
		#yellow_lines = image.hls_select(yellow_mask)
		#return cv2.bitwise_or(mask_RGB,white_lines_HLS)
		return white_lines_HLS

	def drow_detected_lines(self):

		warp_zero = np.zeros_like(self.img.bird_eye_image).astype(np.uint8)
		color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

		pts_left = np.array([np.transpose(np.vstack([self.line.left_fitx-20, self.line.ploty]))])
		pts_left_2 = np.array([np.flipud(np.transpose(np.vstack([self.line.left_fitx+20, self.line.ploty])))])
		pts_right = np.array([np.flipud(np.transpose(np.vstack([self.line.right_fitx-20, self.line.ploty])))])

		pts_r = np.array([np.transpose(np.vstack([self.line.right_fitx-20, self.line.ploty]))])
		pts_r_2 = np.array([np.flipud(np.transpose(np.vstack([self.line.right_fitx+20, self.line.ploty])))])
		
		pts = np.hstack((pts_left, pts_right))
		pts_left = np.hstack((pts_left, pts_left_2))
		pts_right = np.hstack((pts_r, pts_r_2))

		# Draw the lane onto the warped blank image
		cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
		cv2.fillPoly(color_warp, np.int_([pts_left]), (255,0, 0))
		cv2.fillPoly(color_warp, np.int_([pts_right]), (0,0, 255))		

		newwarp = self.cm.perspective_transformation_to_original(color_warp) 
		# Combine the result with the original image
		result = cv2.addWeighted(self.img.undistorted_image, 1, newwarp, 0.3, 0)

		# Write lane curvature on image
		curvature = int(self.line.mean_curvature)
		if curvature > 10000:
			curvature = "straight"

		cv2.putText(result, 'Lane Radius: {}m'.format(curvature), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, 0)
		# Write lane offset on image
		cv2.putText(result, 'Lane Offset: {}m'.format(round(self.line.offset, 4)), (10, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, 0)
		
		#plt.imshow(result)
		#plt.show()
		
		return result