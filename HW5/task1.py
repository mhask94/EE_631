import cv2 as cv
import numpy as np

video = cv.VideoCapture('MotionFieldVideo.mp4')

while (True):
    ret,frame = video.read()
    cv.imshow('Video',frame)
    key = cv.waitKey(30)
    if key == ord('q'):
        break
