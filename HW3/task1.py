import cv2 as cv
import numpy as np
import glob

# termination criteria for cornerSubPix()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,30,0.001)

# number of corners in chessboard (width and height)
num_w = 10
num_h = 7

# create an array of points in a grid: [(0,0,0),(1,0,0),...,(10,7,0)]
obj_pt = np.zeros((num_w*num_h,3), np.float32)
obj_pt[:,:2] = np.mgrid[0:num_w,0:num_h].T.reshape(-1,2)

for k in range(2):
    if k == 0:
        img_location = 'imgs/leftCalibrate/*.bmp'
        store_location = 'params/left_cam.yml'
        print('Calibrating left camera ... this may take a while')
    else:
        img_location = 'imgs/rightCalibrate/*.bmp'
        store_location = 'params/right_cam.yml'       
        print('Calibrating right camera ... this may take a while')

    # create empty list to store the 3d and 2d points in
    obj_pts = [] # 3d points in real world space
    img_pts = [] # 2d points in image plane
    
    # glob all of the images together
    images = glob.glob(img_location)
    
    # load each image separately
    for filename in images:
        img = cv.imread(filename)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        
        ret, corners = cv.findChessboardCorners(gray,(num_w,num_h),None)
    
        if ret == True:
            obj_pts.append(obj_pt)
    
            corners2 = cv.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
            img_pts.append(corners2)
    
    # run calibration
    ret,mat,dist,rvecs,tvecs = cv.calibrateCamera(obj_pts,img_pts,gray.shape[::-1],None,None)
    
    print('\nIntrinsic Parameter Matrix: \n',mat)
    print('\nDistortion Coefficients: \n',dist.T,'\n')
    
    #store the files
    fs = cv.FileStorage(store_location,cv.FILE_STORAGE_WRITE)
    fs.write("intrinsic",mat)
    fs.write("distortion",dist)
    fs.release()
    print('Wrote camera params to ',store_location,'\n')

cv.destroyAllWindows()

