import os.path
import pickle
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



calibration_file = ".\camera_cal\calibration.p"
perspective_transformation_file = ".\camera_cal\persp_trans.p"

class CameraManager:

	def __init__(self, images_path, grid_shape):
		self.cam_mtx = []
		self.cal_dist = []
		self.rvecs = []
		self.tvecs = []
		self.calibration_done = False
		self.DEBUG = True

		if os.path.isfile(calibration_file):
			calibration = pickle.load(open(calibration_file, "rb"))
			self.cam_mtx = calibration["mtx"]
			self.cal_dist = calibration["dist"]
			self.rvecs = calibration["rvecs"]
			self.tvecs = calibration["tvecs"]
			self.calibration_done = True

		self.path = images_path
		self.grid_shape = grid_shape

	def calibrate_camera(self):

		objpoints = []
		imgpoints = []

		images = glob.glob(self.path+"\calibration*.jpg")
		
		objp = np.zeros((self.grid_shape[0]*self.grid_shape[1],3),np.float32)
		objp[:,:2] = np.mgrid[0:self.grid_shape[0],0:self.grid_shape[1]].T.reshape(-1,2)

		for image in images:
			original_img = cv2.imread(image)
			gray_image = cv2.cvtColor(original_img,cv2.COLOR_BGR2GRAY)
			
			ret, corners = cv2.findChessboardCorners(gray_image, self.grid_shape, None)
			if ret:
				imgpoints.append(corners)
				objpoints.append(objp)
			
				img = cv2.drawChessboardCorners(original_img, self.grid_shape, corners, ret)

				if self.DEBUG:
					plt.imshow(img)
					plt.show()

		ret, self.cam_mtx, self.cal_dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_image.shape[::-1], None, None)
		calibration = {"mtx":self.cam_mtx, "dist": self.cal_dist, "rvecs": self.rvecs,"tvecs": self.tvecs}
		pickle.dump(calibration, open(pickle_file_path, "wb" ))
		self.calibration_done = True
	
	def undistort_image(self,img):
		return cv2.undistort(img, self.cam_mtx, self.cal_dist, None, self.cam_mtx)

	def perspective_transformation(self,undst_img):
		
		if not os.path.isfile(perspective_transformation_file):
			persp_trans = pickle.load(open(perspective_transformation_file, "rb"))
			self.M = persp_trans['M']
			self.Minv = persp_trans["Minv"]
		else:


			offset = 200 
			
			img_size = undst_img.shape
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

			#cv2.drawChessboardCorners(undst_img, (2,2), src_corners, True)
			
			
			dst_corners = np.float32([[0+offset,img_size[0]],#bottomLeft
				[img_size[1]-offset,img_size[0]],#bottomRight
			    [img_size[1]-offset,0],#topRight
			    [0+offset,0]])#topLeft

			
			#src_corners = np.array([[585, 460], [203, 720], [1127, 720], [695, 460]]).astype(np.float32)
			#dst_corners = np.array([[320, 0], [320, 720], [960, 720], [960, 0]]).astype(np.float32)
			
			"""
			src_corners = np.float32(
				[[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
				[((img_size[0] / 6) - 10), img_size[1]],
				[(img_size[0] * 5 / 6) + 60, img_size[1]],
				[(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
			dst_corners = np.float32(
				[[(img_size[0] / 4), 0],
				[(img_size[0] / 4), img_size[1]],
				[(img_size[0] * 3 / 4), img_size[1]],
				[(img_size[0] * 3 / 4), 0]])
			"""
			self.M = cv2.getPerspectiveTransform(src_corners, dst_corners)
			self.Minv = cv2.getPerspectiveTransform(dst_corners, src_corners)
			p_trans = {"M":self.M, "Minv": self.Minv}
			pickle.dump(p_trans, open(perspective_transformation_file, "wb" ))

		warped = cv2.warpPerspective(undst_img, self.M, (undst_img.shape[1],undst_img.shape[0]))

		### Used to show the perspective transformation
		"""
		cv2.line(undst_img,(int(bottom_left_coordinate),int(bottom_margin)),(int(bottom_right_coordinate),int(bottom_margin)),(255,0,0),3)
		cv2.line(undst_img,(int(bottom_right_coordinate),int(bottom_margin)),(int(top_right_coordinate),int(top_margin)),(255,0,0),3)
		cv2.line(undst_img,(int(top_right_coordinate),int(top_margin)),(int(top_left_coordinate),int(top_margin)),(255,0,0),3)
		cv2.line(undst_img,(int(top_left_coordinate),int(top_margin)),(int(bottom_left_coordinate),int(bottom_margin)),(255,0,0),3)
		cv2.line(warped,(int(0+offset),int(img_size[0])),(int(img_size[1]-offset),int(img_size[0])),(255,0,0),3)
		cv2.line(warped,(int(img_size[1]-offset),int(img_size[0])),(int(img_size[1]-offset),int(0)),(255,0,0),3)
		cv2.line(warped,(int(img_size[1]-offset),int(0)),(int(0+offset),int(0)),(255,0,0),3)
		cv2.line(warped,(int(0+offset),int(0)),(int(0+offset),int(img_size[0])),(255,0,0),3)
					
		f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
		f.tight_layout()
		ax1.imshow(undst_img)
		ax1.set_title('Original Image', fontsize=20)
		ax2.imshow(warped)
		ax2.set_title('Undistorted and Warped Image', fontsize=20)
		plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
		plt.show()
		"""		

		return warped

	def perspective_transformation_to_original(self,undst_img):

		return cv2.warpPerspective(undst_img, self.Minv, (undst_img.shape[1], undst_img.shape[0])) 

