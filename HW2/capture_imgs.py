import cv2 as cv

video = cv.VideoCapture(-1)

path = 'my_imgs/img'
count = 0
ext = '.jpg'

while (True):
    ret,frame = video.read()

    key = cv.waitKey(1)
    if key > 0 and not key == ord('q'):
        cv.imwrite(path+str(count)+ext, frame)
        count += 1
    elif key == ord('q'):
        break

    cv.imshow('Task 5',frame)

print('Closing Application')
video.release()
cv.destroyAllWindows()
