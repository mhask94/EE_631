import cv2 as cv
import numpy as np
import glob

path = 'imgs/AR'
ext = '.jpg'
count = 1;

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,30,0.001)

# number of corners in chessboard (width and height)
num_w = 10
num_h = 7

obj_pt = np.zeros((num_w*num_h,3), np.float32)
obj_pt[:,:2] = np.mgrid[0:num_w,0:num_h].T.reshape(-1,2)

obj_pts = [] # 3d points in real world space
img_pts = [] # 2d points in image plane

images = glob.glob('imgs/*.jpg')

for filename in images:
    img = cv.imread(filename)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    ret, corners = cv.findChessboardCorners(gray,(num_w,num_h),None)
    if ret == True:
        obj_pts.append(obj_pt)

        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        img_pts.append(corners2)

        img = cv.drawChessboardCorners(img,(num_w,num_h),corners2,ret)
        cv.imshow('Task 1',img)
        cv.waitKey(0)

print('End of images')
print('Closing application')
cv.destroyAllWindows()

