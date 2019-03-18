import cv2 as cv
import numpy as np
import glob

def leftImageClick(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print('x: ',x)
        print('y: ',y)
def rightImageClick(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print('x: ',x)
        print('y: ',y)

imgs_l = sorted(glob.glob('imgs/baseball/ballL*.bmp'))
imgs_r = sorted(glob.glob('imgs/baseball/ballR*.bmp'))

base_l = cv.imread(imgs_l[0])
base_r = cv.imread(imgs_r[0])
base_l = cv.cvtColor(base_l, cv.COLOR_BGR2GRAY)
base_r = cv.cvtColor(base_r, cv.COLOR_BGR2GRAY)

Xl = 345
Yl = 85
Xr = 265
Yr = 85
w = 50
h = 40

#crop_l = base_l[Yl:Yl+h,Xl:Xl+w]
#crop_r = base_r[Yr:Yr+h,Xr:Xr+w]

#cv.imshow('left', base_l)
#cv.setMouseCallback('left',leftImageClick)
#cv.imshow('right',base_r)
#cv.setMouseCallback('right',leftImageClick)
#cv.imshow('crop left',crop_l)
#cv.imshow('crop right',crop_r)
#cv.waitKey(0)

for file_l,file_r in zip(imgs_l,imgs_r):
    img_l = cv.imread(file_l)
    img_r = cv.imread(file_r)
    
    crop_base_l = base_l[Yl:Yl+h,Xl:Xl+w]
    crop_base_r = base_r[Yr:Yr+h,Xr:Xr+w]

    crop_l = img_l[Yl:Yl+h,Xl:Xl+w]
    crop_r = img_r[Yr:Yr+h,Xr:Xr+w]

    gray_l = cv.cvtColor(crop_l, cv.COLOR_BGR2GRAY)
    gray_r = cv.cvtColor(crop_r, cv.COLOR_BGR2GRAY)
    
    diff_l = cv.absdiff(crop_base_l,gray_l)
    diff_r = cv.absdiff(crop_base_r,gray_r)
    _, diff_l = cv.threshold(diff_l,30,255,cv.THRESH_BINARY)
    _, diff_r = cv.threshold(diff_r,30,255,cv.THRESH_BINARY)

    mask = np.ones((5,5),np.uint8)
    diff_l = cv.erode(diff_l,mask,iterations=1)
    diff_r = cv.erode(diff_r,mask,iterations=1)
    diff_l = cv.dilate(diff_l,mask,iterations=2)
    diff_r = cv.dilate(diff_r,mask,iterations=2)

    M_l = cv.moments(diff_l)
    M_r = cv.moments(diff_r)

    if not M_l["m00"] == 0:
        x_l = int(M_l["m10"] / M_l["m00"])
        y_l = int(M_l["m01"] / M_l["m00"])
    else:
        x_l = 0
        y_l = 0
    if not M_r["m00"] == 0:
        x_r = int(M_r["m10"] / M_r["m00"])
        y_r = int(M_r["m01"] / M_r["m00"])
    else:
        x_r = 0
        y_r = 0

    if x_l != 0 and y_l != 0:
        cv.circle(img_l,(x_l+Xl,y_l+Yl),5,(0,0,255))
        cv.circle(img_r,(x_r+Xr,y_r+Yr),5,(0,0,255))

    cv.imshow('left',img_l)
    cv.imshow('right',img_r)
    cv.imshow('diff - left', diff_l)
    cv.imshow('diff - right',diff_r)
    cv.imshow('crop left',crop_l)
    cv.imshow('crop right',crop_r)
    key = cv.waitKey(0)
    if key == ord('q'):
        break

    # age data
    if x_l != 0 and y_l != 0:
        Xl += int(x_l - 25)
        Yl += int(y_l - 20)
        Xr += int(x_r - w/2)
        Yr += int(y_r - 20)
        w += 1
        h += 1
    else:
        w = 50
        h = 40
    if Xl < 0:
        Xl = 0
    if Yl < 0:
        Yl = 0
    if Xr < 0:
        Xr = 0
    if Yr < 0:
        Yr = 0

cv.destroyAllWindows()
print('Program has Ended ...')

#params = cv2.SimpleBlobDetector_Params()
#params.filterByArea = True
#params.minArea = 150
#params.maxArea = 1000
#params.filterByCircularity = True
#params.minCircularity = 0.1
#params.filterByInertia = False
#params.filterByConvexity = False
#params.filterByColor = True
#params.minThreshold = 150
#
#detector = cv2.SimpleBlobDetector_create(params)
#keypts = detector.detect(erode_img)
