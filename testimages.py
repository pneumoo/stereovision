import numpy as np
import cv2
import glob


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
chessBoardGrid = np.zeros((6*9,3), np.float32)
chessBoardGrid[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
leftObjectPoints = [] # 3d point in real world space
leftImagePoints = [] # 2d points in image plane.
rightObjectPoints = [] # 3d point in real world space
rightImagePoints  = [] # 2d points in image plane.

leftimages = sorted(glob.glob('leftcalpic3*.jpg'))
rightimages = sorted(glob.glob('rightcalpic3*.jpg'))

for fname in leftimages:
    img = cv2.imread(fname)
    leftGrayFrame = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(leftGrayFrame, (9,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        leftObjectPoints.append(chessBoardGrid)
        corners2 = cv2.cornerSubPix(leftGrayFrame,corners,(11,11),(-1,-1),criteria)
        leftImagePoints.append(corners2)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners2 ,ret)
        cv2.imshow(str(fname),img)
        cv2.waitKey(0)
leftImageSize = leftGrayFrame.shape[::-1]

for fname in rightimages:
    img = cv2.imread(fname)
    rightGrayFrame = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(img, (9,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        rightObjectPoints.append(chessBoardGrid)
        corners2 = cv2.cornerSubPix(rightGrayFrame,corners,(11,11),(-1,-1),criteria)
        rightImagePoints.append(corners2)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners2,ret)
        cv2.imshow(str(fname),img)
        cv2.waitKey(0)

rightImageSize = rightGrayFrame.shape[::-1]
cv2.destroyAllWindows()

