import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from CameraManager import CameraManager
from ImageManager import ImageManager
import os
from moviepy.editor import VideoFileClip
nx=9
ny=6

test_images_path = "./test_images"
output_images_path = "./output_images"

def test_camera_calibration(image_path, cm):

	original_image = cv2.imread(image_path)
	ret, corners = cv2.findChessboardCorners(original_image, (9,6), None)

	if ret:
		cv2.drawChessboardCorners(original_image, (9,6), corners, ret)

	undistorted_image = cm.undistort_image(original_image)


	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
	f.tight_layout()
	ax1.imshow(original_image)
	ax1.set_title('Original Image', fontsize=50)
	ax2.imshow(undistorted_image)
	ax2.set_title('Undistorted Image', fontsize=50)
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	plt.show()


def test_camera_undirstorsion(cm):	
	
	original_image = cv2.imread("./test_images/test2.jpg")
	undistorted_image = cm.undistort_image(original_image)
	warped = cm.perspective_transformation(undistorted_image)

	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
	f.tight_layout()
	ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
	ax1.set_title('Original Image', fontsize=50)
	ax2.imshow(cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB))
	ax2.set_title('Undistorted Image', fontsize=50)
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	plt.show()

def process_test_images():
	
	files = os.listdir(test_images_path)

	for f in files:
		rgb_image = cv2.cvtColor(cv2.imread("./test_images/"+f),cv2.COLOR_BGR2RGB)
		im = ImageManager(cm)
		results = im.find_lane_lines(rgb_image)
		cv2.imwrite("{}/{}".format(output_images_path,f),cv2.cvtColor(results,cv2.COLOR_RGB2BGR))

		### Used to show the application of the threshold to the HLS image
		"""
		f, axes = plt.subplots(2, 2, figsize=(24, 9))
		f.tight_layout()
		axes[0,0].imshow(im.img.HLS_image[:,:,0])
		axes[0,0].set_title('H channel', fontsize=20)
		axes[0,1].imshow(im.img.HLS_image[:,:,1])
		axes[0,1].set_title('L channel', fontsize=20)
		axes[1,0].imshow(im.img.HLS_image[:,:,2])
		axes[1,0].set_title('S channel', fontsize=20)
		axes[1,1].imshow(im.img.color_line_mask)
		axes[1,1].set_title('HLS mask', fontsize=20)
		plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
		plt.show()
		"""

		### Used to show the application of the Sobel masking
		"""
		f, axes = plt.subplots(2, 2, figsize=(24, 9))
		f.tight_layout()
		axes[0,0].imshow(im.img.undistorted_image)
		axes[0,0].set_title('Original Image', fontsize=20)
		axes[0,1].imshow(im.img.mag_sobel,cmap="gray")
		axes[0,1].set_title('Magnitude Sobel mask', fontsize=20)
		axes[1,0].imshow(im.img.dir_sobel,cmap="gray")
		axes[1,0].set_title('Direction Sobel mask', fontsize=20)
		axes[1,1].imshow(im.img.sobel_combined,cmap="gray")
		axes[1,1].set_title('Combinarion of mag and dir masks', fontsize=20)
		plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
		plt.show()
		"""

		### Used to show the color + sobel masks applications
		"""
		f, (ax0, ax1)  = plt.subplots(1, 2, figsize=(15, 8))
		f.tight_layout()

		ax0.imshow(im.img.undistorted_image)
		ax0.set_title('Original RGB Image', fontsize=20)
		ax1.imshow(im.img.lane_lines_mask,cmap='gray')
		ax1.set_title('Color/Sobel mask application', fontsize=20)
		plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
		plt.show()
		"""


		### Used to show the perspective transformation

		"""
		cv2.line(cv2.cvtColor(im.img.lane_lines_mask,cv2.COLOR_GRAY2RGB),(92, 720),(1120, 720),(0,255,0),3)
		cv2.line(cv2.cvtColor(im.img.lane_lines_mask,cv2.COLOR_GRAY2RGB),(1120, 720),(712, 464),(0,255,0),3)
		cv2.line(cv2.cvtColor(im.img.lane_lines_mask,cv2.COLOR_GRAY2RGB),(712, 464),(571, 464),(0,255,0),3)
		cv2.line(cv2.cvtColor(im.img.lane_lines_mask,cv2.COLOR_GRAY2RGB),(571, 464),(92, 720),(0,255,0),3)
		cv2.line(cv2.cvtColor(im.img.bird_eye_image,cv2.COLOR_GRAY2RGB),(200, 720),(1080, 720),(0,255,0),3)
		cv2.line(cv2.cvtColor(im.img.bird_eye_image,cv2.COLOR_GRAY2RGB),(1080, 720),(1080, 0),(0,255,0),3)
		cv2.line(cv2.cvtColor(im.img.bird_eye_image,cv2.COLOR_GRAY2RGB),(1080, 0),(200, 0),(0,255,0),3)
		cv2.line(cv2.cvtColor(im.img.bird_eye_image,cv2.COLOR_GRAY2RGB),(200, 0),(200, 720),(0,255,0),3)
		
		f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
		f.tight_layout()
		ax1.imshow(im.img.lane_lines_mask,cmap="gray")
		ax1.set_title('Original Image', fontsize=20)
		ax2.imshow(im.img.bird_eye_image,cmap="gray")
		ax2.set_title('Undistorted and Warped Image', fontsize=20)
		plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
		plt.show()
		"""	
			
		

def test_video():
	print("Running on test video1...")
	# Define our Lanes object
	#im = ImageManager(cm)
	#####################################
	# Run our pipeline on the test video 
	#####################################

	clip = VideoFileClip("./project_video.mp4")
	output_video = "./project_video_processed.mp4"
	output_clip = clip.fl_image(process_image)
	output_clip.write_videofile(output_video, audio=False)

	#clip = VideoFileClip("./challenge_video.mp4")
	#output_video = "./challenge_video_processed.mp4"
	#output_clip = clip.fl_image(process_image)
	#output_clip.write_videofile(output_video, audio=False)

	#clip = VideoFileClip("./harder_challenge_video.mp4")
	#output_video = "./harder_challenge_video_processed.mp4"
	#output_clip = clip.fl_image(process_image)
	#output_clip.write_videofile(output_video, audio=False)

	

def process_image(img):
	result = im.find_lane_lines(img)
	return result


cm = CameraManager(".\camera_cal",(nx,ny))
im = ImageManager(cm)

if __name__ == "__main__":

	nx=9
	ny=6
	cm = CameraManager(".\camera_cal",(nx,ny))
	if(not cm.calibration_done):
		print("Camera Calibration")
		cm.calibrate_camera()
		print("Camera Calibrated")
	#### USED to test calibration ONLY 
	#test_camera_calibration(".\camera_cal\calibration3.jpg",cm)
	#undst_img = cm.undistort_image(cv2.imread("./test_images/test2.jpg"))
	#test_camera_undirstorsion(cm)

	#process_test_images()
	test_video()

