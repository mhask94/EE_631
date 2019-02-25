import cv2 as cv
import numpy as np

global img_pts_l, img_pts_r
img_pts_l = [] 
img_pts_r = [] 

def drawCircleLeft(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDOWN and len(img_pts_l) < 3:
        img_pts_l.append((x,y))
        cv.circle(image_l,(x,y),2,(0,0,255))
        cv.circle(image_l,(x,y),3,(0,0,255))
        cv.imshow(left_window,image_l)

def drawCircleRight(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDOWN and len(img_pts_r) < 3:
        img_pts_r.append((x,y))
        cv.circle(image_r,(x,y),2,(0,255,0))
        cv.circle(image_r,(x,y),3,(0,255,0))
        cv.imshow(right_window,image_r)

left_window = 'Task 3 - left'
right_window = 'Task 3 - right'

# create window for left camera 
cv.namedWindow(left_window)
cv.setMouseCallback(left_window, drawCircleLeft)

# create window for right camera 
cv.namedWindow(right_window)
cv.setMouseCallback(right_window, drawCircleRight)

# read in imagaes
img_l = cv.imread('imgs/combinedCalibrate/aL09.bmp')
img_r = cv.imread('imgs/combinedCalibrate/aR09.bmp')

# read in intrinsic params
fs_read_l = cv.FileStorage('params/left_cam.yml',cv.FILE_STORAGE_READ)
mat_l = fs_read_l.getNode("intrinsic").mat()
dist_l = fs_read_l.getNode("distortion").mat()
fs_read_l.release()

fs_read_r = cv.FileStorage('params/right_cam.yml',cv.FILE_STORAGE_READ)
mat_r = fs_read_r.getNode("intrinsic").mat()
dist_r = fs_read_r.getNode("distortion").mat()
fs_read_r.release()

# undistort images
img2_l = cv.undistort(img_l,mat_l,dist_l,None,mat_l)
img2_r = cv.undistort(img_r,mat_r,dist_r,None,mat_r)

image_l = img2_l.copy()
image_r = img2_r.copy()

while True:
    cv.imshow(left_window, image_l)
    cv.imshow(right_window,image_r)
    key = cv.waitKey(1)

    if key == ord('n'):
        break
        
#print('right: \n', img_pts_r)

fs_read = cv.FileStorage('params/stereo.yml',cv.FILE_STORAGE_READ)
F = fs_read.getNode('F').mat()
fs_read.release()

pts_l = np.array(img_pts_l).reshape(-1,1,2)
pts_r = np.array(img_pts_r).reshape(-1,1,2)

lines_l = cv.computeCorrespondEpilines(pts_l,1,F)
lines_r = cv.computeCorrespondEpilines(pts_r,2,F)

def drawLines(lines,img,color):
    x0 = -10
    x1 = 800
    for r in lines:
        y0 = -(r.item(0)*x0 + r.item(2)) / r.item(1)
        y1 = -(r.item(0)*x1 + r.item(2)) / r.item(1)
        image = cv.line(img,(x0,int(y0)),(x1,int(y1)),color,2)
    return image

image_r = drawLines(lines_l,image_r,(0,0,255))
image_l = drawLines(lines_r,image_l,(0,255,0))

cv.imshow(left_window,image_l)
cv.imshow(right_window,image_r)
cv.waitKey(0)

print('Program has ended')
