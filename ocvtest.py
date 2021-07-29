import numpy as np
import cv2
import glob


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
chessBoardGrid = np.zeros((6*9,3), np.float32)
chessBoardGrid[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
leftObjectPoints = [] # 3d point in real world space
leftImagePoints = [] # 2d points in image plane.
rightObjectPoints = [] # 3d point in real world space
rightImagePoints  = [] # 2d points in image plane.

leftimages = glob.glob('leftcamera*.jpg')
rightimages = glob.glob('rightcamera*.jpg')

for fname in leftimages:
    img = cv2.imread(fname)
    leftGrayFrame = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(leftGrayFrame, (9,6),None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        leftObjectPoints.append(chessBoardGrid)
        corners2 = cv2.cornerSubPix(leftGrayFrame,corners,(11,11),(-1,-1),criteria)
        leftImagePoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners2 ,ret)
        # cv2.imshow('img',img)
        # cv2.waitKey(10)
leftImageSize = leftGrayFrame.shape[::-1]			


for fname in rightimages:
    img = cv2.imread(fname)
    rightGrayFrame = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(img, (9,6),None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        rightObjectPoints.append(chessBoardGrid)
        corners2 = cv2.cornerSubPix(rightGrayFrame,corners,(11,11),(-1,-1),criteria)
        rightImagePoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners2,ret)
        # cv2.imshow('img',img)
        # cv2.waitKey(10)
rightImageSize = rightGrayFrame.shape[::-1]



cv2.destroyAllWindows()


#Check that left and right image sizes match...
if leftImageSize == rightImageSize:
	imageSize = leftImageSize
else:
	print("ERROR: Left and Right image sizes do not match!!!")

	

#Calibrating individual cameras
print("Calibrating camera and distortion matricies for each camera...")		
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(leftObjectPoints, leftImagePoints, imageSize, None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(rightObjectPoints, rightImagePoints, imageSize, None, None)
print(retL/len(leftObjectPoints))
print(retR/len(leftObjectPoints))
np.savez_compressed("LeftCameraMats.npz", mtxL=mtxL, distL=distL, rvecsL=rvecsL, tvecsL=tvecsL)	
np.savez_compressed("RightCameraMats.npz", mtxR=mtxR, distR=distR, rvecsR=rvecsR, tvecsR=tvecsR)	


# Calculate undistortion matricies for left and right cameras
print("Calibrating cameras together...")
(retval, _, _, _, _, rotationMatrix, translationVector, _, _) = cv2.stereoCalibrate(
        leftObjectPoints, leftImagePoints, rightImagePoints,
        mtxL, distL, mtxR, distR, imageSize, cv2.CALIB_FIX_INTRINSIC)

# Calculates needed for undistortion of rectification map
print("Rectifying cameras...")
(leftRectification, rightRectification, leftProjection, rightProjection,
        dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
                mtxL, distL, mtxR, distR, imageSize, rotationMatrix, translationVector,
                None, None, None, None, None,
                cv2.CALIB_ZERO_DISPARITY)

# Creates maps for the final calibrated, rectified, stereo system
print("Calculating left and right X/Y mappings for undistortion and rectification...")				
leftMapX, leftMapY = cv2.initUndistortRectifyMap(
        mtxL, distL, leftRectification,
        leftProjection, imageSize, cv2.CV_32FC1)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(
        mtxR, distR, rightRectification,
        rightProjection, imageSize, cv2.CV_32FC1)

#Save the final stereo pair calibration
# print("Saving final stereo pair calibration...")		
# np.savez_compressed(stereofilename, imageSize=imageSize,
        # leftMapX=leftMapX, leftMapY=leftMapY, leftROI=leftROI,
        # rightMapX=rightMapX, rightMapY=rightMapY, rightROI=rightROI)

		
		
'''		==== TESTING CAMERA MATRICIES W/ REPROJECTION =====
mean_error = 0
for i in range(len(leftObjectPoints)):
    imgpoints2, _ = cv2.projectPoints(leftObjectPoints[i], rvecsL[i], tvecsL[i], mtxL, distL)
    error = cv2.norm(leftImagePoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "Left mean error: {}".format(mean_error/len(leftObjectPoints)) )
mean_error = 0
for i in range(len(rightObjectPoints)):
    imgpoints2, _ = cv2.projectPoints(rightObjectPoints[i], rvecsR[i], tvecsR[i], mtxR, distR)
    error = cv2.norm(rightImagePoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "Right optcammat mean error: {}".format(mean_error/len(rightObjectPoints)) )

newcameramtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL,distL,imageSize,1,imageSize)
newcameramtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR,distR,imageSize,1,imageSize)

mean_error = 0
for i in range(len(leftObjectPoints)):
    imgpoints2, _ = cv2.projectPoints(leftObjectPoints[i], rvecsL[i], tvecsL[i], newcameramtxL, distL)
    error = cv2.norm(leftImagePoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "Left mean error: {}".format(mean_error/len(leftObjectPoints)) )
mean_error = 0
for i in range(len(rightObjectPoints)):
    imgpoints2, _ = cv2.projectPoints(rightObjectPoints[i], rvecsR[i], tvecsR[i], newcameramtxR, distR)
    error = cv2.norm(rightImagePoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "Right optcammat mean error: {}".format(mean_error/len(rightObjectPoints)) )
'''

#==========================RUN THE CAMREAS FOR DISPARITY========================

		
print("Starting rectified cameras w/ disparity...")

#Load up the stereo calibration paramters, (optional... this could be commented out)
# calibration = np.load(stereofilename, allow_pickle=False)
# imageSize = tuple(calibration["imageSize"])
# leftMapX = calibration["leftMapX"]
# leftMapY = calibration["leftMapY"]
# leftROI = tuple(calibration["leftROI"])
# rightMapX = calibration["rightMapX"]
# rightMapY = calibration["rightMapY"]
# rightROI = tuple(calibration["rightROI"])		

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)	
stereo = cv2.StereoSGBM_create(
	minDisparity=0,
	numDisparities=32,
	blockSize=7,
	P1=0,
	P2=0,
	disp12MaxDiff=0,
	uniquenessRatio=0,
	speckleWindowSize=45,
	speckleRange=16,
	preFilterCap=0,
	mode=cv2.STEREO_SGBM_MODE_SGBM)
stereo1 = cv2.StereoSGBM_create(
	minDisparity=0,
	numDisparities=32,
	blockSize=7,
	P1=0,
	P2=0,
	disp12MaxDiff=0,
	uniquenessRatio=0,
	speckleWindowSize=45,
	speckleRange=16,
	preFilterCap=0,
	mode=cv2.STEREO_SGBM_MODE_SGBM)	
while True:
	# Get frame and split into left/right if it's a good frame
	retval, frame = cam.read()
	if retval == True:
		height,width,colors = frame.shape
		halfwidth = width // 2
		leftFrame = frame[:, :halfwidth, :]
		rightFrame = frame[:, halfwidth:, :]
		
		

		
		leftUndistort = cv2.undistort(leftFrame, mtxL, distL)
		rightUndistort = cv2.undistort(rightFrame, mtxR, distR)
		# stereoLeft = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR)
		# stereoRight = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR)		
		
		# Convert L/R to grayscale
		# frameLeftGray = cv2.cvtColor(leftFrame, cv2.COLOR_BGR2GRAY)
		# frameRightGray = cv2.cvtColor(rightFrame, cv2.COLOR_BGR2GRAY)
		leftUndistortGray = cv2.cvtColor(leftUndistort, cv2.COLOR_BGR2GRAY)
		rightUndistortGray = cv2.cvtColor(rightUndistort, cv2.COLOR_BGR2GRAY)		
		# stereoLeftGray = cv2.cvtColor(stereoLeft, cv2.COLOR_BGR2GRAY)
		# stereoRightGray = cv2.cvtColor(stereoRight, cv2.COLOR_BGR2GRAY)			

		

		
		uncalibratedDisparity = stereo.compute(leftUndistortGray, rightUndistortGray)
		#stereoDisparity = stereo1.compute(stereoLeftGray, stereoRightGray)
		# disparityNorm = cv2.normalize(disparity, disparity, alpha=0, beta=127,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
		
		# cv2.imshow('Uncalibrated Left Frame', leftFrame)
		# cv2.imshow('Uncalibrated Right Frame', rightFrame)
		cv2.imshow('LEFT UNDISTORT', leftUndistort)
		cv2.imshow('RIGHT UNDISTORT', rightUndistort)
		# cv2.imshow('LEFT UNDISTORT W/ OPTIMAL CAMERA MAT', leftOptCamMat)
		# cv2.imshow('RIGHT UNDISTORT W/ OPTIMAL CAMERA MAT', rightOptCamMat)
		# cv2.imshow('Stereo-calibrated Left', stereoLeft)
		# cv2.imshow('Stereo-calibrated Right', stereoRight)

		cv2.imshow('Uncalibrated Disparity', uncalibratedDisparity/2048)
		# cv2.imshow('Stereo Disparity', stereoDisparity/2048)				
				
	if cv2.waitKey(1) == 27:
		break		
cv2.destroyAllWindows()			









'''
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import glob


camWidth = 1280 #in pixels
camHeight = 480 #in pixels
sqSize = 1 #in inches
picture = 0
grid = '9x6'

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while picture != 20:
	# Get frame and split into left/right if it's a good frame
	retval, frame = cam.read()
	if retval == True:
		height,width,colors = frame.shape
		halfwidth = width // 2
		frameLeft = frame[:, :halfwidth, :]
		frameRight = frame[:, halfwidth:, :]
		
		cv2.imshow('show', frameLeft)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			cv2.imwrite('leftcamera_{0}_{1}_{2}in_{3}_img{4}.jpg'.format(width, height, sqSize, grid, picture), frameLeft)
			print('Captured leftcamera_{0}_{1}_{2}in_{3}_img{4}.jpg'.format(width, height, sqSize, grid, picture))
			picture += 1

picture = 0
while picture != 20:
	# Get frame and split into left/right if it's a good frame
	retval, frame = cam.read()
	if retval == True:
		height,width,colors = frame.shape
		halfwidth = width // 2
		frameLeft = frame[:, :halfwidth, :]
		frameRight = frame[:, halfwidth:, :]
		
		cv2.imshow('show', frameRight)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			cv2.imwrite('rightcamera_{0}_{1}_{2}in_{3}_img{4}.jpg'.format(width, height, sqSize, grid, picture), frameRight)
			print('Captured rightcamera_{0}_{1}_{2}in_{3}_img{4}.jpg'.format(width, height, sqSize, grid, picture))
			picture += 1			
'''





''' === FOR A COMBINED STEREO USB CAMERA ===

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while True:
	retval, frame = cam.read()
	if retval == True:
		# Split the images into to left and right
		height,width,colors = frame.shape
		halfwidth = width // 2
		frameLeft = frame[:, :halfwidth, :]
		frameRight = frame[:, halfwidth:, :]
		grayFrameLeft = cv2.cvtColor(frameLeft, cv2.COLOR_BGR2GRAY)
		grayFrameRight = cv2.cvtColor(frameRight, cv2.COLOR_BGR2GRAY)	
		
		#stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
		# SGBM Parameters ----------------- 
		stereo = cv2.StereoSGBM_create(
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
		
	if cv2.waitKey(1) == 27:
		break

cam.release()
cv2.destroyAllWindows()

'''





""" == FOR TWO SEPARATE USB CAMERAS ===


video_capture_0 = cv2.VideoCapture(0)
video_capture_1 = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret0, frame0 = video_capture_0.read()
    ret1, frame1 = video_capture_1.read()

    if (ret0):
        # Display the resulting frame
        cv2.imshow('LEFT', frame0)

    if (ret1):
        # Display the resulting frame
        cv2.imshow('RIGHT', frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture_0.release()
video_capture_1.release()
cv2.destroyAllWindows()
"""