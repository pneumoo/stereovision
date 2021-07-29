import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Calibration image number
calimnum = 10

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# For left and right cameras
objpLeft = np.zeros((6*7,3), np.float32)
objpLeft[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
objpRight = np.zeros((6*7,3), np.float32)
objpRight[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpointsLeft = [] # 3d point in real world space for the left camera
imgpointsLeft = [] # 2d points in image plane for the left camera
objpointsRight = [] # 3d point in real world space for the right camera
imgpointsRight = [] # 2d points in image plane for the right camera

# Begin video capture while loop
# Loop persists until we have filled our L&R image point matrices and are ready for calibration step
cam = cv.VideoCapture(0)
cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
while len(objpointsLeft) < calimnum and len(objpointsRight) < calimnum:
	# Get frame and split into left/right if it's a good frame
	retval, frame = cam.read()
	cv.imshow('camera',frame)
	height,width,colors = frame.shape
	halfwidth = width // 2
	frameLeft = frame[:, :halfwidth, :]
	frameRight = frame[:, halfwidth:, :]
	
	# Convert L/R to grayscale
	grayFrameLeft = cv.cvtColor(frameLeft, cv.COLOR_BGR2GRAY)
	grayFrameRight = cv.cvtColor(frameRight, cv.COLOR_BGR2GRAY)	
	
	# Loop for checkerboard pattern on L&R frames
	retLeft, cornersLeft = cv.findChessboardCorners(grayFrameLeft, (7,6), None)
	retRight, cornersRight = cv.findChessboardCorners(grayFrameRight, (7,6), None)
	
	# If found, add object points, image points (after refining them)
	if retLeft == True & retRight == True:
		objpointsLeft.append(objpLeft)
		cornersLeft2 = cv.cornerSubPix(grayFrameLeft,cornersLeft, (11,11), (-1,-1), criteria)
		imgpointsLeft.append(cornersLeft)

		objpointsRight.append(objpRight)
		cornersRight2 = cv.cornerSubPix(grayFrameRight,cornersRight, (11,11), (-1,-1), criteria)
		imgpointsRight.append(cornersRight)			
		
		# Draw and display the corners for L&R (comment this section out for SPEED)
		cv.drawChessboardCorners(grayFrameLeft, (7,6), cornersLeft2, retLeft)
		cv.imshow('LEFT', grayFrameLeft)
		cv.drawChessboardCorners(grayFrameRight, (7,6), cornersRight2, retRight)
		cv.imshow('RIGHT', grayFrameRight)
		cv.waitKey(500)

	if cv.waitKey(1) == 27:
		break		
cv.destroyAllWindows()
		
		
		
# Calibration, now that we have the object and image points
retL, mtxL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpointsLeft, imgpointsLeft, grayFrameLeft.shape[::-1], None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpointsRight, imgpointsRight, grayFrameRight.shape[::-1], None, None)		

# Get optimal camera matricies for L&R
hL,  wL = grayFrameLeft.shape[:2]
hR,  wR = grayFrameRight.shape[:2]
newcameramtxLeft, roiLeft = cv.getOptimalNewCameraMatrix(mtxL, distL, (wL,hL), 1, (wL,hL))
newcameramtxRight, roiRight = cv.getOptimalNewCameraMatrix(mtxR, distR, (wR,hR), 1, (wR,hR))

# np.savez_compressed('calibration.npx, imageSize=imageSize,
        # leftMapX=leftMapX, leftMapY=leftMapY, leftROI=leftROI,
        # rightMapX=rightMapX, rightMapY=rightMapY, rightROI=rightROI)


# test this shit with a new video
while True:
	# Get frame and split into left/right if it's a good frame
	retval, frame = cam.read()
	if retval == True:
		height,width,colors = frame.shape
		halfwidth = width // 2
		frameLeft = frame[:, :halfwidth, :]
		frameRight = frame[:, halfwidth:, :]
		
		# Convert L/R to grayscale
		grayFrameLeft = cv.cvtColor(frameLeft, cv.COLOR_BGR2GRAY)
		grayFrameRight = cv.cvtColor(frameRight, cv.COLOR_BGR2GRAY)	
		
		# Undistort the grayscale images
		undistortLeft = cv.undistort(grayFrameLeft, mtxL, distL, None, newcameramtxLeft)
		undistortRight = cv.undistort(grayFrameRight, mtxR, distR, None, newcameramtxRight)
		
		stereo = cv.StereoSGBM_create(
			minDisparity=0,
			numDisparities=16,
			blockSize=1,
			P1=0,
			P2=0,
			disp12MaxDiff=0,
			uniquenessRatio=0,
			speckleWindowSize=0,
			speckleRange=0,
			preFilterCap=0,
			mode=cv2.STEREO_SGBM_MODE_SGBM
		)
		disparity = stereo.compute(grayFrameLeft, grayFrameRight)
		disparityNorm = cv2.normalize(disparity, disparity, alpha=0, beta=127,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
		
		cv2.imshow('LEFT FRAME', frameLeft)
		cv2.imshow('RIGHT FRAME', frameRight)
		cv2.imshow('GRAY LEFT FRAME', grayFrameLeft)
		cv2.imshow('GRAY RIGHT FRAME', grayFrameRight)
		cv2.imshow('3D DISPARITY', disparity)
		cv2.imshow('3D NORMALIZED DISPARITY', disparityNorm)

	

cv.destroyAllWindows()