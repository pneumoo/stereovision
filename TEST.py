import numpy as np
import cv2
import glob


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
chessBoardGrid = np.zeros((6*9,3), np.float32)
chessBoardGrid[:,:2] = 25.4 * np.mgrid[0:9,0:6].T.reshape(-1,2)
leftObjectPoints = [] # 3d point in real world space
leftImagePoints = [] # 2d points in image plane.
rightObjectPoints = [] # 3d point in real world space
rightImagePoints  = [] # 2d points in image plane.

leftimages = sorted(glob.glob('leftcamera*.jpg'))
rightimages = sorted(glob.glob('rightcamera*.jpg'))

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

	# Check that left and right object point matricies match...
if leftObjectPoints == rightObjectPoints:
	objectPoints = leftObjectPoints
else:
	print("ERROR: Left and Right object matricies sizes do not match!!!")

	
	
	
OPTIMIZE_ALPHA = 1
TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.00001)
REMAP_INTERPOLATION = cv2.INTER_LINEAR
DEPTH_VISUALIZATION_SCALE = 2048

# print("Calibrating left camera...")
_, leftCameraMatrix, leftDistortionCoefficients, _, _ = cv2.calibrateCamera(
        objectPoints, leftImagePoints, imageSize, None, None)
# print("Calibrating right camera...")
_, rightCameraMatrix, rightDistortionCoefficients, _, _ = cv2.calibrateCamera(
        objectPoints, rightImagePoints, imageSize, None, None)


rotationMatrix = np.array([[1.,0.,0.],
                        [0., 1., 0.],
                        [0., 0., 1.]])# Initial R guess is no rotation from left to right cameras

translationVector = np.array([[-1.],[0.],[0.]]) # Initial T guess is difference in x-direction from left to right camera

_, _, _, _, _, rotationMatrix, translationVector, _, _ = cv2.stereoCalibrate(
        objectPoints, leftImagePoints, rightImagePoints,
        leftCameraMatrix, leftDistortionCoefficients,
        rightCameraMatrix, rightDistortionCoefficients,
        imageSize, rotationMatrix, translationVector, None, None, 
        cv2.CALIB_FIX_INTRINSIC, TERMINATION_CRITERIA)




# print("Calibrating cameras together...")


		# print("Calibrating cameras together...")
# (_, leftCameraMatrix, leftDistortionCoefficients, rightCameraMatrix, rightDistortionCoefficients, 
		# rotationMatrix, translationVector, E, F) = cv2.stereoCalibrate(
			# objectPoints, leftImagePoints, rightImagePoints,
			# imageSize, rotationMatrix, translationVector, None, None,
			# cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_USE_EXTRINSIC_GUESS, TERMINATION_CRITERIA)
			
# print("POST-stereoCalibrate individual camera matricies")
# print(leftCameraMatrix)
# print(rightCameraMatrix)
# print(leftDistortionCoefficients)
# print(rightDistortionCoefficients)	
# print(rotationMatrix)
# print(translationVector)



leftRectification, rightRectification, leftProjection, rightProjection, dispartityToDepthMap, leftROI, rightROI = cv2.stereoRectify(
                leftCameraMatrix, leftDistortionCoefficients,
                rightCameraMatrix, rightDistortionCoefficients,
                imageSize, rotationMatrix, translationVector)

leftMapX, leftMapY = cv2.initUndistortRectifyMap(
        leftCameraMatrix, leftDistortionCoefficients, leftRectification,
        leftProjection, imageSize, cv2.CV_32FC1)

rightMapX, rightMapY = cv2.initUndistortRectifyMap(
        rightCameraMatrix, rightDistortionCoefficients, rightRectification,
        rightProjection, imageSize, cv2.CV_32FC1)

print("Saving calibration...")
np.savez_compressed("stereocal.npz", 
        leftMapX=leftMapX, 
        leftMapY=leftMapY,
        leftRectification=leftRectification,
        leftProjection=leftProjection, 
        leftROI=leftROI,
        rightMapX=rightMapX, 
        rightMapY=rightMapY,
        rightRectification=rightRectification,
        rightProjection=rightProjection, 
        rightROI=rightROI,
        dispartityToDepthMap=dispartityToDepthMap)
        
		
		

# Generate images
lefttestpic = cv2.imread('lefttestpic.jpg')
righttestpic = cv2.imread('righttestpic.jpg')
fulltestpic = cv2.imread('fulltestpic.jpg')		
leftUndistort = cv2.undistort(lefttestpic, leftCameraMatrix, leftDistortionCoefficients)
rightUndistort = cv2.undistort(righttestpic, rightCameraMatrix, rightDistortionCoefficients)
leftRemap = cv2.remap(lefttestpic, leftMapX, leftMapY, REMAP_INTERPOLATION)
rightRemap = cv2.remap(righttestpic, rightMapX, rightMapY, REMAP_INTERPOLATION)

# Display images
cv2.imshow('Left Test Picture',lefttestpic)		
cv2.imshow('Right Test Picture',righttestpic)	
cv2.imshow('leftUndistort',leftUndistort)		
cv2.imshow('rightUndistort',rightUndistort)	
cv2.imshow('leftRemap',leftRemap)		
cv2.imshow('rightRemap',rightRemap)
cv2.waitKey()

cv2.destroyAllWindows() 

	
stereoMatcher = cv2.StereoSGBM_create()


cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)	
while True:
	# Get frame and split into left/right if it's a good frame
	retval, frame = cam.read()
	if retval == True:
		height,width,colors = frame.shape
		halfwidth = width // 2
		leftFrame = frame[:, :halfwidth, :]
		rightFrame = frame[:, halfwidth:, :]
		
		leftHeight, leftWidth = leftFrame.shape[:2]
		rightHeight, rightWidth = rightFrame.shape[:2]
		
		if (leftWidth, leftHeight) != imageSize:
			print("Left camera has different size than the calibration data")
			break

		if (rightWidth, rightHeight) != imageSize:
			print("Right camera has different size than the calibration data")
			break

		
		fixedLeft = cv2.remap(leftFrame, leftMapX, leftMapY, REMAP_INTERPOLATION)
		fixedRight = cv2.remap(rightFrame, rightMapX, rightMapY, REMAP_INTERPOLATION)

		grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
		grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
		depth = stereoMatcher.compute(grayLeft, grayRight)

		cv2.imshow('left', fixedLeft)
		cv2.imshow('right', fixedRight)
		cv2.imshow('depth', depth / DEPTH_VISUALIZATION_SCALE)
		cv2.waitKey()

cam.release()
cv2.destroyAllWindows()
