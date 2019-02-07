import cv2 as cv
import numpy as np

# get image
img  = cv.imread('task4_data/Object_with_Corners.jpg')

# read in intrinsic params
fs_read = cv.FileStorage('cam.yml',cv.FILE_STORAGE_READ)
mat = fs_read.getNode("intrinsic").mat()
dist = fs_read.getNode("distortion").mat()
fs_read.release()

# load data
with open("task4_data/Data_Points.txt") as f:
    #w,h = [float(x) for x in next(f).split()]
    array = []
    for line in f:
        array.append([float(x) for x in line.split()])

img_pts = []
obj_pts = []

for vec in array:
    if len(vec) == 2:
        img_pts.append(vec)
    elif len(vec) == 3:
        obj_pts.append(vec)
    else:
        print('error')

# convert to format that solvePnP() can use
img_pts = np.array(img_pts,np.float64)
obj_pts = np.array(obj_pts,np.float64)
img_pts = img_pts.reshape((img_pts.shape[0],img_pts.shape[1],1))
obj_pts = obj_pts.reshape((obj_pts.shape[0],obj_pts.shape[1],1))

# get R and t
ret, rvec, tvec = cv.solvePnP(obj_pts,img_pts,mat,dist)
R, Jacobian = cv.Rodrigues(rvec)

print('R: \n',R)
print('t: \n',tvec)

print('Program has ended')

