import numpy as np
import cv2


REMAP_INTERPOLATION = cv2.INTER_LINEAR
calfile = 'calibrationmats.npz'
rectfile = 'rectificationmats.npz'


calmats = np.load(calfile)
print("Loading rectification matricies")
rectmats = np.load(rectfile)
imageSize = rectmats['imageSize']
leftMapX = rectmats['leftMapX']
leftMapY = rectmats['leftMapY']
leftROI = rectmats['leftROI']
rightMapX = rectmats['rightMapX']
rightMapY = rectmats['rightMapY']
rightROI = rectmats['rightROI']
disparityToDepthMap = rectmats['disparityToDepthMap']



# Load a test image from file
lefttestpic = cv2.imread('testpictures\\lefttestpic4.jpg')
righttestpic = cv2.imread('testpictures\\righttestpic4.jpg')
leftRemap = cv2.remap(lefttestpic, leftMapX, leftMapY, REMAP_INTERPOLATION)
rightRemap = cv2.remap(righttestpic, rightMapX, rightMapY, REMAP_INTERPOLATION)

# graylefttestpic = cv2.cvtColor(lefttestpic, cv2.COLOR_BGR2GRAY)
# grayrighttestpic = cv2.cvtColor(righttestpic, cv2.COLOR_BGR2GRAY)
# grayleftUndistort = cv2.cvtColor(leftUndistort, cv2.COLOR_BGR2GRAY)
# grayrightUndistort = cv2.cvtColor(rightUndistort, cv2.COLOR_BGR2GRAY)
grayleftRemap = cv2.cvtColor(leftRemap, cv2.COLOR_BGR2GRAY)
grayrightRemap = cv2.cvtColor(rightRemap, cv2.COLOR_BGR2GRAY)

imgL = grayleftRemap
imgR = grayrightRemap

# SGBM Parameters -----------------
window_size = 3                     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
left_matcher = cv2.StereoSGBM_create(
    minDisparity=1,
    numDisparities=16,             # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=3,
    P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=0,
    mode=cv2.STEREO_SGBM_MODE_HH
)

# window_size = 3
# min_disp = 1
# num_disp = 96
# left_matcher = cv2.StereoSGBM_create(minDisparity = min_disp,
# 	numDisparities = num_disp,
# 	blockSize = 5,
# 	P1 = 8*3*window_size**2,
# 	P2 = 32*3*window_size**2,
# 	disp12MaxDiff = 1,
# 	uniquenessRatio = 10,
# 	speckleWindowSize = 100,
# 	speckleRange = 32
# )

right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

# FILTER Parameters
lmbda = 80000
sigma = 1.2
visual_multiplier = 1.0
 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)
print('computing disparity...')
displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
displ = np.int16(displ)
dispr = np.int16(dispr)
filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
filteredImg = np.uint8(filteredImg)
# cv2.imshow('Disparity Map', filteredImg)
# cv2.waitKey()
# cv2.destroyAllWindows()



""" stereoMatcher = cv2.StereoBM_create(numDisparities=0, blockSize=21)

# StereoBM and StereoSGBM compute 16-bit fixed point disparity maps
# where 4 bits are used for the fractional portion. The output type
# is numpy.int16.
# 
# As inputs, they only accept 8bit, single-channel images. 
# depthtestpic = stereoMatcher.compute(graylefttestpic, grayrighttestpic)
# depthUndistort = stereoMatcher.compute(grayleftUndistort, grayrightUndistort)
depthRemap = stereoMatcher.compute(grayleftRemap, grayrightRemap)


#Converts the int16 output of .compute to an 8-bit scale for imshow
depthRemap_scaleabs = cv2.convertScaleAbs(depthRemapLR.clip(0))

# falsecolormap = cv2.applyColorMap(depthRemap_scaleabs, cv2.COLORMAP_JET)

# Display images (imshow)
# If the image is 16-bit unsigned or 32-bit integer, the pixels 
# are divided by 256. That is, the value range [0,255*256] is mapped to [0,255].
# cv2.imshow('Left Test Picture',lefttestpic)		
# cv2.imshow('Right Test Picture',righttestpic)	
# cv2.imshow('leftUndistort',leftUndistort)		
# cv2.imshow('rightUndistort',rightUndistort)
# cv2.imshow('leftRemap',leftRemap)		
# cv2.imshow('rightRemap',rightRemap)
# cv2.imshow('Test Pic Depth', depthtestpic)
# cv2.imshow('Undistort Depth', depthUndistort)
# cv2.imshow('Remap Depth', depthRemap)
# cv2.imshow('Test Pic Depth', depthtestpic_scaled)
# cv2.imshow('Undistort Depth', depthUndistort_scaled)
cv2.imshow('Remap Depth', depthRemap)

# cv2.imshow("colors!", falsecolormap)
cv2.waitKey(0) """





cam = cv2.VideoCapture(0)
while True:
	# Get frame and split into left/right if it's a good frame
	retval, frame = cam.read()
	if retval == True:
		height,width,colors = frame.shape
		halfwidth = width // 2
		leftFrame = frame[:, :halfwidth, :]
		rightFrame = frame[:, halfwidth:, :]
		
		remapLeft = cv2.remap(leftFrame, leftMapX, leftMapY, REMAP_INTERPOLATION)
		remapRight = cv2.remap(rightFrame, rightMapX, rightMapY, REMAP_INTERPOLATION)

		imgL = cv2.cvtColor(remapLeft, cv2.COLOR_BGR2GRAY)
		imgR = cv2.cvtColor(remapRight, cv2.COLOR_BGR2GRAY)

		displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
		dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
		displ = np.int16(displ)
		dispr = np.int16(dispr)
		filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
		filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
		filteredImg = np.uint8(filteredImg)

		#TODO 
		# trying to use the "disparityToDepthMap" to figure out the actual depth of parts of the image
		# based on the filtered disparities. 
		reproject3D = cv2.reprojectImageTo3D(filteredImg, disparityToDepthMap, handleMissingValues=False)
		depthmap = reproject3D[:,:,2]
		# depthmap = cv2.normalize(src=depthmap, dst=depthmap, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
		test8 = np.uint8(depthmap)
		test255 = cv2.normalize(src=test8, dst=None, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)

		# print(np.amax(reproject3D[:,:,2]), np.amin(reproject3D[:,:,2]))
		# print(reproject3D)
		# reproject3D = np.uint8(reproject3D)
		# print(np.amax(reproject3D[:,:,2]), np.amin(reproject3D[:,:,2]))
		# print(reproject3D)
		test_color = cv2.applyColorMap(test255, cv2.COLORMAP_JET)

		cv2.imshow('Raw Left Camera', leftFrame)
		cv2.imshow('Raw Right Camera', rightFrame)
		cv2.imshow('Disparity Map', filteredImg)
		cv2.imshow('Reprojection', test_color)
		# cv2.imshow('Reprojection Color', reproject3D_color)
		# print(filteredImg)
		# print(reproject3D[:,:,2])
		# print(reproject3D_color)
		
				
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cam.release()
cv2.destroyAllWindows()
