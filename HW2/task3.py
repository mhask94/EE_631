import cv2 as cv
import numpy as np

# get images
close  = cv.imread('dist_imgs/Close.jpg')
far    = cv.imread('dist_imgs/Far.jpg')
turned = cv.imread('dist_imgs/Turned.jpg')

# read in intrinsic params
fs_read = cv.FileStorage('cam.yml',cv.FILE_STORAGE_READ)
mat = fs_read.getNode("intrinsic").mat()
dist = fs_read.getNode("distortion").mat()
fs_read.release()

# store images in a list
imgs = [close,far,turned]

# undistort each image and display the abs difference
for img in imgs:
    h, w = img.shape[:2]
    mat2, roi = cv.getOptimalNewCameraMatrix(mat,dist,(w,h),1,(w,h))
    img2 = cv.undistort(img, mat, dist, None, mat2)

    diff = cv.absdiff(img,img2)

    cv.imshow('Task 3 - Distortion absdiff()',diff)
    cv.waitKey(0)

print('Program has ended')
cv.destroyAllWindows()

