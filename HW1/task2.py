import sys
rm_python2 = []
for p in sys.path:
    if p.find('python2') != -1:
        rm_python2.append(p)
for p in rm_python2:
    sys.path.remove(p)

import cv2
import numpy as np

def runAbsDiff(video,old_frame):
    ret,frame = video.read()
    diff = cv2.absdiff(frame,old_frame)
    return diff

def runThreshold(video,frame):
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret,new_frame = cv2.threshold(frame,125,255,cv2.THRESH_BINARY)
    return new_frame

def runCanny(video,frame):
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(frame,50,150)
    return edges

def runCornerSubPix(video,frame):
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

def runHoughLines(video,frame):
    edges = runCanny(video,frame)
    lines = cv2.HoughLines(edges,1,np.pi/180,150)
    try:
        lines[0] == None
    except TypeError:
        return frame
    for line in lines:
        for r,theta in line:
            c = np.cos(theta)
            s = np.sin(theta)
            x0 = r*c
            y0 = r*s
            x1 = int(x0+1000*-s)
            y1 = int(y0+1000*c)
            x2 = int(x0-1000*-s)
            y2 = int(y0-1000*c)
            cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
    return frame

#cv2.namedWindow('Haskell',cv2.WINDOW_NORMAL)
video = cv2.VideoCapture(-1)

#prev_ret,prev_frame = video.read()
mode = 0

fourCC = cv2.VideoWriter_fourcc(*'MPEG')
#fourCC = video.getBackendName()
out = cv2.VideoWriter('Haskell_HW1.avi',fourCC,25.0,(640,480))
write = False

while (True):
    ret,frame = video.read()
    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    key = cv2.waitKey(1)
    if key == ord('n'):
        mode = 0
    elif key == ord('g'):
        mode = 1
    elif key == ord('d'):
        mode = 2
    elif key == ord('t'):
        mode = 3
    elif key == ord('c'):
        mode = 4
    elif key == ord('p'):
        mode = 5
    elif key == ord('l'):
        mode = 6
    elif key == ord('w'):
        write = True
        print('Writing')
    elif key == ord('s'):
        write = False
        print('Stopped Writing')
    elif key == ord('q'):
        break

    if mode == 1:
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    elif mode == 2:
        image = runAbsDiff(video,frame)
    elif mode == 3:
        image = runThreshold(video,frame) 
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    elif mode == 4:
        image = runCanny(video,frame)
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    elif mode == 5:
        image = runCornerSubPix(video,frame)
    elif mode == 6:
        image = runHoughLines(video,frame)
    else:
        image = frame

    cv2.imshow('Haskell',image)
    if write:
#        print('writing frame')
        out.write(image)

video.release()
out.release()
cv2.destroyAllWindows()


