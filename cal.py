import numpy as np
import cv2
import glob

PATH = 'C:\\Users\\Brian\Documents\\Python Scripts\\stereovision\\stereocalpics_9x6grid_320x240\\'
calfile = 'calibrationmats.npz'
rectfile = 'rectificationmats.npz'


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
chessBoardGrid = np.zeros((6*9,3), np.float32)
chessBoardGrid[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
leftObjectPoints = [] # 3d point in real world space
leftImagePoints = [] # 2d points in image plane.
rightObjectPoints = [] # 3d point in real world space
rightImagePoints  = [] # 2d points in image plane.

print("Loading calibration images from " + str(PATH))
leftimages = sorted(glob.glob(str(PATH) + 'leftcalpic*.jpg'))
rightimages = sorted(glob.glob(str(PATH) + 'rightcalpic*.jpg'))

print("Finding chessboard corners for left images...")
for fname in leftimages:
	img = cv2.imread(fname)
	leftGrayFrame = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# Find the chess board corners
	ret, corners = cv2.findChessboardCorners(leftGrayFrame, (9,6),None)
	# If found, add object points, image points (after refining them)
	if ret == True:
		leftObjectPoints.append(chessBoardGrid)
		corners2 = cv2.cornerSubPix(leftGrayFrame,corners,(11,11),(-1,-1),criteria)
		leftImagePoints.append(corners2)
		# Draw and display the corners
		cv2.drawChessboardCorners(img, (9,6), corners2 ,ret)
		# cv2.imshow('img',img)
		# cv2.waitKey(10)
leftImageSize = leftGrayFrame.shape[::-1]

print("Finding chessboard corners for right images...")
for fname in rightimages:
	img = cv2.imread(fname)
	rightGrayFrame = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# Find the chess board corners
	ret, corners = cv2.findChessboardCorners(img, (9,6),None)
	# If found, add object points, image points (after refining them)
	if ret == True:
		rightObjectPoints.append(chessBoardGrid)
		corners2 = cv2.cornerSubPix(rightGrayFrame,corners,(11,11),(-1,-1),criteria)
		rightImagePoints.append(corners2)
		# Draw and display the corners
		cv2.drawChessboardCorners(img, (9,6), corners2,ret)
		# cv2.imshow('img',img)
		# cv2.waitKey(10)
rightImageSize = rightGrayFrame.shape[::-1]
cv2.destroyAllWindows()




# Check that left and right image sizes match...
if leftImageSize == rightImageSize:
	imageSize = leftImageSize
else:
	print("ERROR: Left and Right image sizes do not match!!!")

if leftObjectPoints == rightObjectPoints:
	objectPoints = leftObjectPoints
else:
	print("ERROR: Left and Right object matricies sizes do not match!!!")




# Individually calibrate each camera
# This finds the intrinsic properties for each camera before we find the stereo
# properties relating the cameras to each other...
print("Calibrating left camera...")
leftCalibrationError, leftCameraMatrix, leftDistortionCoefficients, leftRotationMatrix, leftTranslationVector = cv2.calibrateCamera(
	objectPoints, leftImagePoints, imageSize, None, None)

print("Calibrating right camera...")
rightCalibrationError, rightCameraMatrix, rightDistortionCoefficients, rightRotationMatrix, rightTranslationVector = cv2.calibrateCamera(
	objectPoints, rightImagePoints, imageSize, None, None)


# Now that we have the individual camera properties, we can relate the cameras to each other
# cv2.stereoCalibrate() is used to determine the rotational matrix and translational vector
# that describes how the two cameras are oriented relative to each other in space.
# It's important to note, here, that this function requires the calibration photos to have been taken
# for the left and right cameras simultaneously and of the same scene. It also requires that the image/object points
# for each camara matrix be in the same order. In other words, the two cameras need to be looking at the same thing
# in order for them to orient themselves to one another with this function.
print("Finding Rotation matrix and Translation vector between left and right cameras...")
_, _, _, _, _, rotationMatrix, translationVector, _, _ = cv2.stereoCalibrate(
	objectPoints, leftImagePoints, rightImagePoints,
	leftCameraMatrix, leftDistortionCoefficients,
	rightCameraMatrix, rightDistortionCoefficients,
	imageSize, None, None, None, None, 
	cv2.CALIB_FIX_INTRINSIC, criteria)

OPTIMIZE_ALPHA = 1
# Rectify the cameras relative to one another. 
print("Stereo rectification, with alpha = " + str(OPTIMIZE_ALPHA))
leftRectification, rightRectification, leftProjection, rightProjection, disparityToDepthMap, leftROI, rightROI = cv2.stereoRectify(
	leftCameraMatrix, leftDistortionCoefficients,
	rightCameraMatrix, rightDistortionCoefficients,
	imageSize, rotationMatrix, translationVector,
	None, None, None, None, None,
	OPTIMIZE_ALPHA)

# Generates the undistortion mappings between left and right cameras
print("Generating left and right mappings from cv2.initUndistortRectifyMap()")
leftMapX, leftMapY = cv2.initUndistortRectifyMap(
	leftCameraMatrix, leftDistortionCoefficients, leftRectification,
	leftProjection, imageSize, cv2.CV_32FC1)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(
	rightCameraMatrix, rightDistortionCoefficients, rightRectification,
	rightProjection, imageSize, cv2.CV_32FC1)

print("Saving calibration points as " + str(calfile))
np.savez_compressed(calfile, 
	leftCalibrationError=leftCalibrationError, 
	leftCameraMatrix=leftCameraMatrix, 
	leftDistortionCoefficients=leftDistortionCoefficients,
	leftRotationMatrix = leftRotationMatrix,
	leftTranslationVector = leftTranslationVector,
	rightCalibrationError = rightCalibrationError,
	rightCameraMatrix=rightCameraMatrix,
	rightDistortionCoefficients=rightDistortionCoefficients,
	rightRotationMatrix = rightRotationMatrix,
	rightTranslationVector = rightTranslationVector)

print("Saving rectification maps as " + str(rectfile))
np.savez_compressed(rectfile, 
	imageSize=imageSize,
	leftMapX=leftMapX, 
	leftMapY=leftMapY, 
	leftROI=leftROI,
	rightMapX=rightMapX, 
	rightMapY=rightMapY, 
	rightROI=rightROI,
	disparityToDepthMap = disparityToDepthMap)
