import cv2 as cv
import numpy as np

left_window = 'Task 1 - left'
right_window = 'Task 1 - right'

# create window for left camera 
cv.namedWindow(left_window)

# create window for right camera 
cv.namedWindow(right_window)

# read in imagaes
img_l = cv.imread('imgs/combinedCalibrate/aL09.bmp')
img_r = cv.imread('imgs/combinedCalibrate/aR09.bmp')
gray_l = cv.cvtColor(img_l,cv.COLOR_BGR2GRAY)
gray_r = cv.cvtColor(img_r,cv.COLOR_BGR2GRAY)
shape = (gray_l.shape[1],gray_l.shape[0])

cv.imshow(left_window,img_l)
cv.imshow(right_window,img_r)
cv.waitKey(0)

# termination criteria for cornerSubPix()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,30,0.001)

# number of corners in chessboard (width and height)
num_w = 10
num_h = 7

# create an array of points in a grid: [(0,0,0),(1,0,0),...,(10,7,0)]
obj_pt = np.zeros((num_w*num_h,3), np.float32)
obj_pt[:,:2] = np.mgrid[0:num_w,0:num_h].T.reshape(-1,2) * 3.88636
obj_pts = []
img_pts_l = []
img_pts_r = []

ret_l, corners_l = cv.findChessboardCorners(gray_l,(num_w,num_h),None)
ret_r, corners_r = cv.findChessboardCorners(gray_r,(num_w,num_h),None)

if ret_l == True and ret_r == True:
    obj_pts.append(obj_pt)
    
    corners2_l = cv.cornerSubPix(gray_l,corners_l,(5,5),(-1,-1),criteria)
    corners2_r = cv.cornerSubPix(gray_r,corners_r,(5,5),(-1,-1),criteria)
    img_pts_l.append(corners2_l)
    img_pts_r.append(corners2_r)

corner_pts_l = np.array([corners2_l[0],corners2_l[9],
                        corners2_l[60],corners2_l[69]])
corner_pts_r = np.array([corners2_r[0],corners2_r[9],
                        corners2_r[60],corners2_r[69]])

# read in intrinsic params
fs_read_l = cv.FileStorage('params/left_cam.yml',cv.FILE_STORAGE_READ)
mat_l = fs_read_l.getNode("intrinsic").mat()
dist_l = fs_read_l.getNode("distortion").mat()
fs_read_l.release()

fs_read_r = cv.FileStorage('params/right_cam.yml',cv.FILE_STORAGE_READ)
mat_r = fs_read_r.getNode("intrinsic").mat()
dist_r = fs_read_r.getNode("distortion").mat()
fs_read_r.release()

fs_read = cv.FileStorage('params/stereo.yml',cv.FILE_STORAGE_READ)
R = fs_read.getNode('R').mat()
T = fs_read.getNode('T').mat()
fs_read.release()

R1,R2,P1,P2,Q,roi1,roi2 = cv.stereoRectify(mat_l,dist_l,mat_r,dist_r,shape,R,T)

fs = cv.FileStorage('params/rectify.yml',cv.FILE_STORAGE_WRITE)
fs.write("R1",R1)
fs.write("R2",R2)
fs.write("P1",P1)
fs.write("P2",P2)
fs.write("Q",Q)
fs.release()

dst_pts_l = cv.undistortPoints(corner_pts_l, mat_l, dist_l, R=R1, P=P1)
dst_pts_r = cv.undistortPoints(corner_pts_r, mat_r, dist_r, R=R2, P=P2)
print('shape: ',dst_pts_l.shape)
print('dst_pts_l: \n',dst_pts_l)
print('mat_l: \n',mat_l)
print('dist_l: \n',dist_l)
print('R1: \n',R1)
print('P1: \n',P1)

for data in dst_pts_l:
    x = data[0][0]
    y = data[0][1]
    cv.circle(img_l,(x,y),3,(0,0,255))
for data in dst_pts_r:
    x = data[0][0]
    y = data[0][1]
    cv.circle(img_r,(x,y),3,(0,0,255))

disparity = np.array([[dst_pts_l[0][0][0]-dst_pts_r[0][0][0]],
    [dst_pts_l[1][0][0]-dst_pts_r[1][0][0]],
    [dst_pts_l[2][0][0]-dst_pts_r[2][0][0]],
    [dst_pts_l[3][0][0]-dst_pts_r[3][0][0]],
    ])

dst_pts_l = np.array(dst_pts_l).reshape((4,2))
dst_pts_l = np.hstack([dst_pts_l,disparity]).reshape((4,1,3))
dst_pts_r = np.array(dst_pts_r).reshape((4,2))
dst_pts_r = np.hstack([dst_pts_r,disparity]).reshape((4,1,3))

print('dst_pts_l: \n',dst_pts_l)
obj_dst_l = cv.perspectiveTransform(dst_pts_l, Q)
obj_dst_r = cv.perspectiveTransform(dst_pts_r, Q)
print('3D corners left image: \n',obj_dst_l)
print('3D corners right image: \n',obj_dst_r)

#P = np.array(obj_dst_l).reshape((4,3)).T
#obj_dst_r = (R @ P + T).T
#print('3D corners right image: \n',obj_dst_r.reshape((4,1,3)))

cv.imshow(left_window,img_l)
cv.imshow(right_window,img_r)
cv.waitKey(0)

cv.destroyAllWindows()
