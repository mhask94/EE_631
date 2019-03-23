import cv2 as cv

# task 1 vids
## 1st video
#vid_names = ['Task1_level0_skip0.avi','Task1_level0_skip20.avi']
## 2nd video
#vid_names = ['Task1_level3_skip0.avi','Task1_level3_skip20.avi']

# task 2 vids
#vid_names = ['Task2_skip0.avi','Task2_skip20.avi']

# task 3 vids
vid_names = ['Task3_skip1.avi','Task3_skip20.avi']

fps = 20
size = (1920,1080)
fourCC = cv.VideoWriter_fourcc(*'MPEG')
#writer = cv.VideoWriter('Task1_level0.avi',fourCC,fps,size,isColor=1)
#writer = cv.VideoWriter('Task1_level3.avi',fourCC,fps,size,isColor=1)
#writer = cv.VideoWriter('Task2.avi',fourCC,fps,size,isColor=1)
writer = cv.VideoWriter('Task3.avi',fourCC,fps,size,isColor=1)

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
