import cv2 as cv

# task 2 vids
vid_names = ['Task2_skip0.avi','Task2_skip20.avi']

fps = 20
size = (1920,1080)
fourCC = cv.VideoWriter_fourcc(*'MPEG')
writer = cv.VideoWriter('Task2.avi',fourCC,fps,size,isColor=1)

for name in vid_names:
    print('processing '+name)
    video = cv.VideoCapture(name)
    ret, frame = video.read()
    writer.write(frame)
    while (ret):
        ret, frame = video.read()
        writer.write(frame)
    video.release()

print('Finished merging videos')
writer.release()
