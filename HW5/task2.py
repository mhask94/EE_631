import cv2 as cv
import numpy as np

def getPoint(img_size, pt, side):
    x = 0
    y = 0
    if (pt[0] > img_size[1] - side/2.0):
        x = img_size[1] - side
    elif (pt[0] < side/2.0):
        x = 0
    else:
        x = pt[0] - side/2.0

    if (pt[1] > img_size[0] - side/2.0):
        y = img_size[0] - side
    elif (pt[1] < side/2.0):
        y = 0
    else:
        y = pt[1] - side/2.0

    return [x,y]


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
    tmp_size = (ts, ts)
    search_size = (ss, ss)
    method = cv.TM_SQDIFF_NORMED
    quality = 0.01
    min_dist = 10.0

    ret, prev_frame = video.read()
    prev_frame_g = cv.cvtColor(prev_frame,cv.COLOR_BGR2GRAY)
    prev_corners = cv.goodFeaturesToTrack(prev_frame_g,max_corners,quality,min_dist)

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
                x,y = arr[0]
                pt = getPoint(frame_g.shape, [x,y], ts)
                temp = prev_frame_g[int(pt[1]):int(pt[1])+tmp_size[1],\
                        int(pt[0]):int(pt[0])+tmp_size[0]]

                s_pt = getPoint(frame_g.shape, [x,y], ss)
                search_img = frame_g[int(s_pt[1]):int(s_pt[1])+search_size[1],\
                        int(s_pt[0]):int(s_pt[0])+search_size[0]]

                cols = search_img.shape[0]-temp.shape[0] + 1
                rows = search_img.shape[1]-temp.shape[1] + 1

                result = cv.matchTemplate(search_img,temp,method)
#                result = cv.normalize(result,None,alpha=0,beta=1,norm_type=cv.NORM_MINMAX,dtype=-1)
                min_val,max_val,min_loc,max_loc = cv.minMaxLoc(result)
                match_loc_x = pt[0] - cols/2.0 + min_loc[0]
                match_loc_y = pt[1] - rows/2.0 + min_loc[1]

                cv.circle(frame,(int(x),int(y)),2,(0,255,0),-1)
                cv.line(frame,(int(x),int(y)),(int(match_loc_x),int(match_loc_y)),(0,0,255),1)

#            print('percent found: ',count / len(prev_corners) * 100)
            if record:
                writer.write(frame)
            if show:
                cv.imshow('Video',frame)
                key = cv.waitKey(3)
                if key == ord('q'):
                    break
            prev_frame_g = frame_g.copy()
            prev_corners = cv.goodFeaturesToTrack(frame_g,max_corners,quality,min_dist)
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
featureMatch(20,view,write)

cv.destroyAllWindows()
