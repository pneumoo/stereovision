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
while True:
	# Get frame and split into left/right if it's a good frame
	retval, frame = cam.read()
	if retval == True:
		height,width,colors = frame.shape
		halfwidth = width // 2
		frameLeft = frame[:, :halfwidth, :]
		frameRight = frame[:, halfwidth:, :]
		
		cv2.imshow('left', frameLeft)
		cv2.imshow('right', frameRight)

	if cv2.waitKey(1) & 0xFF == ord('s'):
		print('took picture')
		cv2.imwrite('lefttestpic.jpg', frameLeft)
		cv2.imwrite('righttestpic.jpg',frameRight)
		cv2.imwrite('fulltestpic.jpg', frame)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



cam.release()
cv2.distroyAllWindows
		
	
		
