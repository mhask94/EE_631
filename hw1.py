import sys
rm_python2 = []
for p in sys.path:
    if p.find('python2') != -1:
        rm_python2.append(p)
for p in rm_python2:
    sys.path.remove(p)

import cv2

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
    edges = cv2.Canny(frame,100,200)
    return edges

video = cv2.VideoCapture(-1)

#prev_ret,prev_frame = video.read()
mode = 0

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
    elif key == ord('q'):
        break

    if mode == 1:
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    elif mode == 2:
        image = runAbsDiff(video,frame)
    elif mode == 3:
        image = runThreshold(video,frame) 
    elif mode == 4:
        image = runCanny(video,frame)
    else:
        image = frame

    cv2.imshow('Haskell',image)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

video.release()
cv2.destroyAllWindows()


