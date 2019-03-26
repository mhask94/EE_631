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
        writer = cv.VideoWriter('Task3_skip'+str(num_frames)+'.avi',fourCC,fps,size,isColor=1)
    max_corners = 500
    ts = 5
    ss = 11 * ts
    tmp_size = (ts, ts)
    search_size = (ss, ss)
    method = cv.TM_SQDIFF_NORMED
    quality = 0.01
    min_dist = 10.0

    ret = True
    while (ret):
        # skip the desired number of frames
        for k in range(num_frames):
            ret, prev_frame = video.read()
            if ret == False:
                break
            prev_frame_g = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
            prev_corners = cv.goodFeaturesToTrack(prev_frame_g,max_corners,quality,min_dist)
            orig_corners = prev_corners

            count = 0
            for i in range(num_frames):
                ret, frame = video.read()
                if ret == False:
                    break
                frame_g = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                new_corners = []

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
                    min_val,max_val,min_loc,max_loc = cv.minMaxLoc(result)
                    match_loc_x = pt[0] - cols/2.0 + min_loc[0]
                    match_loc_y = pt[1] - rows/2.0 + min_loc[1]

                    new_corners.append([match_loc_x,match_loc_y])

                prev_corners = np.array(prev_corners).reshape(len(prev_corners),1,2)
                new_corners = np.array(new_corners).reshape(len(prev_corners),1,2)
                F,status = cv.findFundamentalMat(prev_corners,new_corners,\
                        cv.FM_RANSAC,3,0.99,None)

                prev_corners = []
                tmp = []
                for m in range(len(status)):
                    if (status[m][0] == 1):
                        prev_corners.append(new_corners[m])
                        tmp.append(orig_corners[m])
                orig_corners = tmp

                count += 1
                prev_frame_g = frame_g
            
            for i in range(len(prev_corners)):
                pt_o = (int(orig_corners[i].item(0)),int(orig_corners[i].item(1)))
                pt_p = (int(prev_corners[i].item(0)),int(prev_corners[i].item(1)))

                cv.circle(frame,pt_o,2,(0,255,0),-1)
                cv.line(frame,pt_o,pt_p,(0,0,255),1)

            if record and not (type(frame)==type(None))>0:
#                if not frame.shape[0]==0:
                    writer.write(frame)
            if show and not (type(frame)==type(None)):
#                if not frame.shape[0]==0:
                    cv.imshow('Video',frame)
                    key = cv.waitKey(3)
                    if key == ord('q'):
                        break
    print('Finished processing video')

    # release objects
    video.release()
    if record:
        writer.release()

# main script
view = True
write = True
featureMatch(1,view,write)
featureMatch(20,view,write)

cv.destroyAllWindows()
