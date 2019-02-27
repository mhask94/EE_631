import cv2 as cv
import numpy as np

left_window = 'Task 4 - left'
right_window = 'Task 4 - right'

# create window for left camera 
cv.namedWindow(left_window)

# create window for right camera 
cv.namedWindow(right_window)

# read in imagaes
img_l = cv.imread('imgs/combinedCalibrate/aL09.bmp')
img_r = cv.imread('imgs/combinedCalibrate/aR09.bmp')
gray = cv.cvtColor(img_l,cv.COLOR_BGR2GRAY)
shape = (gray.shape[1],gray.shape[0])

cv.imshow(left_window,img_l)
cv.imshow(right_window,img_r)
cv.waitKey(0)

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

# rectify images
R1,R2,P1,P2,Q,roi1,roi2 = cv.stereoRectify(mat_l,dist_l,mat_r,dist_r,shape,R,T)

map_l1, map_l2 = cv.initUndistortRectifyMap(mat_l,dist_l,R1,P1,shape,5)
map_r1, map_r2 = cv.initUndistortRectifyMap(mat_l,dist_l,R1,P1,shape,5)

rect_img_l = cv.remap(img_l,map_l1,map_l2,cv.INTER_LINEAR)
rect_img_r = cv.remap(img_r,map_r1,map_r2,cv.INTER_LINEAR)

diff_l = cv.absdiff(img_l,rect_img_l)
diff_r = cv.absdiff(img_r,rect_img_r)

def drawLines(img,color):
    x0 = -10
    x1 = 800
    for k in [1,2,3]:
        y = 100 * k
        image = cv.line(img,(x0,y),(x1,y),color,2)
    return image

rect_img_r = drawLines(rect_img_r,(0,0,255))
rect_img_l = drawLines(rect_img_l,(0,255,0))

cv.imshow(left_window,rect_img_l)
cv.imshow(right_window,rect_img_r)
cv.waitKey(0)

cv.imshow(left_window,diff_l)
cv.imshow(right_window,diff_r)
cv.waitKey(0)

cv.destroyAllWindows()
print('Program has ended')
