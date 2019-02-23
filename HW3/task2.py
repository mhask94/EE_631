import cv2 as cv
import numpy as np

# termination criteria for cornerSubPix()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,30,0.001)

# number of corners in chessboard (width and height)
num_w = 10
num_h = 7

# create an array of points in a grid: [(0,0,0),(1,0,0),...,(10,7,0)]
obj_pt = np.zeros((num_w*num_h,3), np.float32)
obj_pt[:,:2] = 3.88636 * np.mgrid[0:num_w,0:num_h].T.reshape(-1,2)

# image paths
path = 'imgs/combinedCalibrate/a'
left = 'L'
right = 'R'
count = 0
ext = '.bmp'

obj_pts = []
img_pts_l = []
img_pts_r = []

print('Getting image points from cameras ...')

while count < 100:
    if count < 10:
        filename_l = path+left+'0'+str(count)+ext
        filename_r = path+right+'0'+str(count)+ext
    else:
        filename_l = path+left+str(count)+ext
        filename_r = path+right+str(count)+ext
    count += 1
    
    img_l = cv.imread(filename_l)
    img_r = cv.imread(filename_r)
    gray_l = cv.cvtColor(img_l,cv.COLOR_BGR2GRAY)
    gray_r = cv.cvtColor(img_r,cv.COLOR_BGR2GRAY)

    ret_l, corners_l = cv.findChessboardCorners(gray_l,(num_w,num_h),None)
    ret_r, corners_r = cv.findChessboardCorners(gray_r,(num_w,num_h),None)

    if ret_l == True and ret_r == True:
        obj_pts.append(obj_pt)

        corners2_l = cv.cornerSubPix(gray_l,corners_l,(5,5),(-1,-1),criteria)
        corners2_r = cv.cornerSubPix(gray_r,corners_r,(5,5),(-1,-1),criteria)
        img_pts_l.append(corners2_l)
        img_pts_r.append(corners2_r)

# read in intrinsic params
fs_read_l = cv.FileStorage('params/left_cam.yml',cv.FILE_STORAGE_READ)
mat_l = fs_read_l.getNode("intrinsic").mat()
dist_l = fs_read_l.getNode("distortion").mat()
fs_read_l.release()

fs_read_r = cv.FileStorage('params/right_cam.yml',cv.FILE_STORAGE_READ)
mat_r = fs_read_r.getNode("intrinsic").mat()
dist_r = fs_read_r.getNode("distortion").mat()
fs_read_r.release()

print('Calibrating stereo system ...')

retval, mat1, dist1, mat2, dist2, R, T, E, F = cv.stereoCalibrate(obj_pts, \
           img_pts_l,img_pts_r,mat_l,dist_l,mat_r, dist_r,gray_l.shape[::-1], \
           flags=cv.CALIB_FIX_INTRINSIC)

print('R: \n',R)
print('T: \n',T)
print('E: \n',E)
print('F: \n',F)

print('Program has ended')

