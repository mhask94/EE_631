import sys
rm_python2 = []
for p in sys.path:
    if p.find('python2') != -1:
        rm_python2.append(p)
for p in rm_python2:
    sys.path.remove(p)

import cv2
import numpy as np

def runThreshold(frame):
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret,new_frame = cv2.threshold(frame,15,255,cv2.THRESH_BINARY)
    return new_frame

def runCornerSubPix(frame):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    ret,dst = cv2.threshold(dst,0.1*dst.max(),255,0)
    dst = np.uint8(dst)

    ret,labels,stats,centroids = cv2.connectedComponentsWithStats(dst)

    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    corners = np.int0(corners)
    #frame = frame - frame
    for x,y in corners:
        cv2.circle(frame,(x,y),2,(0,0,255))
    
    return frame

#cv2.namedWindow('Haskell',cv2.WINDOW_NORMAL)

fourCC = cv2.VideoWriter_fourcc(*'MPEG')
#fourCC = video.getBackendName()
out = cv2.VideoWriter('Haskell_HW1_task3.avi',fourCC,25.0,(640,480))
write = False

path = 'imgs/1'
ext = '.jpg'
left = 'L'
right = 'R'
count = 6
LL = True
baseL = cv2.imread(path+left+'05'+ext)
baseR = cv2.imread(path+right+'05'+ext)

while (True):
    if LL:
        if count < 10:
            filename = path+left+'0'+str(count)+ext
            frame = cv2.imread(filename)
        else:
            frame = cv2.imread(path+left+str(count)+ext)
        diff = cv2.absdiff(frame,baseL)
        count += 1
        if count > 40:
            count = 5
            LL = False
    else:
        if count < 10:
            frame = cv2.imread(path+right+'0'+str(count)+ext)
        else:
            frame = cv2.imread(path+right+str(count)+ext)
        diff = cv2.absdiff(frame,baseR)
        count += 1
        if count > 40:
            count = 5
            break

    key = cv2.waitKey(0)
    if key == ord('k'):
        mode = 0
    elif key == ord('g'):
        mode = 1
    elif key == ord('w'):
        write = True
        print('Writing')
    elif key == ord('s'):
        write = False
        print('Stopped Writing')
    elif key == ord('q'):
        break

    dt = runThreshold(diff)
    kernel = np.ones((5,5),np.uint8)
    erode_img = cv2.erode(dt,kernel,iterations=1)
    dilate_img = cv2.dilate(erode_img,kernel,iterations=1)
    dst = np.uint8(dilate_img)

    ret,labels,stats,centroids = cv2.connectedComponentsWithStats(dst)
    for x,y in centroids:
        x = int(x)
        y = int(y)
        if (x < 316 or x > 321 and y > 241 or y < 238):
#        if dilate_img[x,y] == 255:
#            print('white centroid')
#            frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
            cv2.circle(frame,(x,y),10,(0,0,255))

#    print('stats: \n',stats)
#    print('labels',labels)

#    params = cv2.SimpleBlobDetector_Params()
#    params.filterByArea = True
#    params.minArea = 5
#    params.maxArea = 1000
#    params.filterByCircularity = True
#    params.minCircularity = 0.1
#    params.filterByInertia = False
#    params.filterByConvexity = False
#    params.filterByColor = True
#    params.minThreshold = 150

#    detector = cv2.SimpleBlobDetector_create()
#    keypts = detector.detect(erode_img)
#    print(keypts)
#    key_img = cv2.drawKeypoints(dilate_img,keypts,np.array([]),(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('Haskell',frame)
    if write:
        out.write(frame)

out.release()
cv2.destroyAllWindows()


