import cv2
import numpy as np
import os

# Set to 0 to capture test pictures
# Set to 1  to capture calibration pictures
IMAGE_CAPTURE_TYPE = 1 


camWidth = 1280 #in pixels
camHeight = 480 #in pixels
sqSize = 1 #in inches
picture = 0
w = 9 #grid width
h = 6 #grid height

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

cam = cv2.VideoCapture(0)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, camWidth)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camHeight)
while True:
	# Get frame and split into left/right if it's a good frame
	retval, frame = cam.read()
	if retval == True:
		height,width,colors = frame.shape
		halfwidth = width // 2
		leftFrame = frame[:, :halfwidth, :]
		rightFrame = frame[:, halfwidth:, :]
		leftGrayFrame = cv2.cvtColor(leftFrame,cv2.COLOR_BGR2GRAY)
		rightGrayFrame = cv2.cvtColor(rightFrame,cv2.COLOR_BGR2GRAY)

		grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		cv2.imshow('grayFrame', grayFrame)

	if IMAGE_CAPTURE_TYPE == 1: # Generate calibration pictures
		retLeft, leftCorners = cv2.findChessboardCorners(leftGrayFrame,(w,h),None)
		retRight, rightCorners = cv2.findChessboardCorners(rightGrayFrame,(w,h),None)

		if retLeft == True and retRight == True:
			cv2.drawChessboardCorners(leftFrame, (w,h), leftCorners , retLeft)        
			cv2.drawChessboardCorners(rightFrame, (w,h), rightCorners , retRight)
			cv2.imshow('leftFrame Corners', leftFrame)
			cv2.imshow('rightFrame Corners', rightFrame)
			if cv2.waitKey(1) & 0xFF == ord('s'):
				cv2.imwrite('leftcalpic{}.jpg'.format(picture), leftGrayFrame)
				cv2.imwrite('rightcalpic{}.jpg'.format(picture), rightGrayFrame)
				cv2.imwrite('fullcalpic{}.jpg'.format(picture), grayFrame)
				picture += 1 
	
	if IMAGE_CAPTURE_TYPE == 0: # Generate test piutures
		if cv2.waitKey(1) & 0xFF == ord('s'):
			cv2.imwrite('lefttestpic{}.jpg'.format(picture), leftGrayFrame)
			cv2.imwrite('righttestpic{}.jpg'.format(picture), rightGrayFrame)
			cv2.imwrite('fulltestpic{}.jpg'.format(picture), grayFrame)
			picture += 1 



	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



cam.release()
cv2.destroyAllWindows()
