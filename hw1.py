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

video = cv2.VideoCapture(-1)

#prev_ret,prev_frame = video.read()
mode = 0

while (True):
    ret,frame = video.read()
    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    key = cv2.waitKey(1)
    if key == ord('d'):
        print('d pressed')
        mode = 2
    elif key == ord('g'):
        print('g pressed')
        mode = 1
    elif key == ord('q'):
        print('q pressed')
        break
    elif key == ord('n'):
        print('n pressed')
        mode = 0

    if mode == 1:
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    elif mode == 2:
        image = runAbsDiff(video,frame)
    else:
        image = frame

    cv2.imshow('Haskell',image)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

video.release()
cv2.destroyAllWindows()


