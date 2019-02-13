import cv2 as cv
import numpy as np

# get images
img = cv.imread('my_imgs/test.jpg')

# read in intrinsic params
fs_read = cv.FileStorage('my_cam.yml',cv.FILE_STORAGE_READ)
mat = fs_read.getNode("intrinsic").mat()
dist = fs_read.getNode("distortion").mat()
fs_read.release()

# undistort image and display the abs difference
h, w = img.shape[:2]
mat2, roi = cv.getOptimalNewCameraMatrix(mat,dist,(w,h),1,(w,h))
img2 = cv.undistort(img, mat, dist, None, mat2)

diff = cv.absdiff(img,img2)

cv.imshow('Task 6',diff)
cv.waitKey(0)

print('Program has ended')
cv.destroyAllWindows()

