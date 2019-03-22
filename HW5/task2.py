import cv2 as cv
import numpy as np

def featureMatch(num_frames, show, record):
    print('Processing video with '+str(num_frames)+' skipped frames')
    video = cv.VideoCapture('MotionFieldVideo.mp4')
    fourCC = cv.VideoWriter_fourcc(*'MPEG')
    size = (1920,1080)
    fps = 15.0
    if record:
        writer = cv.VideoWriter('Task2_skip'+str(num_frames)+'.avi',fourCC,fps,size,isColor=1)
    max_corners = 500
    ts = 5
    ss = 11 * ts
    template_size = (ts, ts)
    search_size = (ss, ss)
    method = cv.TM_SQDIFF_NORMED

    quality = 0.01
    min_dist = 10.0
    min_eig = 0.01
    win_size = (21,21)
    crit = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER,50,0.001)
    ret, prev_frame = video.read()
    prev_frame_g = cv.cvtColor(prev_frame,cv.COLOR_BGR2GRAY)
    prev_corners = cv.goodFeaturesToTrack(prev_frame_g,max_corners,quality,min_dist)

    print(type(prev_corners))
    print(prev_corners.item(0))
    return

    count = 0
    while (ret):
        # skip the desired number of frames
        for k in range(num_frames):
            ret, frame = video.read()
        # process the desired frame
        ret,frame = video.read()
        if ret:
            frame_g = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            #something
            for arr in prev_corners:
                x,y = prev_corners[0]


            count = 0
            for i in range(len(prev_corners)):
                if status[i][0] and err[i][0] < 21:
                    count += 1
                    cv.circle(frame, tuple(prev_corners[i][0]),2,(0,255,0),-1)
                    cv.line(frame,tuple(prev_corners[i][0]),tuple(corners[i][0]),(0,0,255),1)
            print('percent found: ',count / len(prev_corners) * 100)
            if record:
                writer.write(frame)
            if show:
                cv.imshow('Video',frame)
                key = cv.waitKey(30)
                if key == ord('q'):
                    break
            prev_frame_g = frame_g.copy()
            prev_corners = cv.goodFeaturesToTrack(prev_frame_g,max_corners,quality,min_dist)
        else:
            break
    print('Finished processing video')

    # release objects
    video.release()
    if record:
        writer.release()

# main script
view = True
write = True
featureMatch(0,view,write)

cv.destroyAllWindows()
