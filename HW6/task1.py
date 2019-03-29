import cv2 as cv
import numpy as np
import glob
from copy import deepcopy as dc
from IPython.core.debugger import Pdb

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

def featureMatch(location, show):
    print('Processing frames from '+location)
    imgs = sorted(glob.glob(location+'/*.jpg'))
    max_corners = 500
    ts = 5
    ss = 11 * ts
    tmp_size = (ts, ts)
    search_size = (ss, ss)
    method = cv.TM_SQDIFF_NORMED
    quality = 0.01
    min_dist = 25.0

    prev_frame = cv.imread(imgs[0])
    prev_frame_g = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
    prev_corners = cv.goodFeaturesToTrack(prev_frame_g,max_corners,quality,min_dist)
    orig_corners = dc(prev_corners)
    orig_frame = dc(prev_frame)
    orig_frame_g = dc(prev_frame_g)

    Pdb().set_trace()

    for k in range(1,len(imgs)):
        frame = cv.imread(imgs[k])
        temp_frame = dc(frame)
        frame_g = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        new_corners = []
        for arr in prev_corners:

            x,y = arr[0]
            pt = getPoint(frame_g.shape, [x,y], ts)
            temp = prev_frame_g[int(pt[1]):int(pt[1])+ts,\
                      int(pt[0]):int(pt[0])+ts]

            s_pt = getPoint(frame_g.shape, [x,y], ss)
            search_img = frame_g[int(s_pt[1]):int(s_pt[1])+ss,\
                    int(s_pt[0]):int(s_pt[0])+ss]

            cols = search_img.shape[1]-temp.shape[1] + 1
            rows = search_img.shape[0]-temp.shape[0] + 1

            result = cv.matchTemplate(search_img,temp,method)
#            result = cv.normalize(result,None,alpha=1,beta=0,norm_type=cv.NORM_MINMAX)
            min_val,max_val,min_loc,max_loc = cv.minMaxLoc(result)
            match_loc_x = pt[0] - cols/2.0 + min_loc[0]
            match_loc_y = pt[1] - rows/2.0 + min_loc[1]

            new_corners.append([match_loc_x,match_loc_y])

        prev_corners = np.array(prev_corners).reshape(len(prev_corners),1,2)
        new_corners = np.array(new_corners).reshape(len(prev_corners),1,2)
        F,status = cv.findFundamentalMat(prev_corners,new_corners,\
                cv.FM_RANSAC,3,0.99,None)

        Pdb().set_trace()

        prev_corners = []
        tmp = []
        for m in range(len(status)):
            if (status[m][0] == 1):
                prev_corners.append(new_corners[m])
                tmp.append(orig_corners[m])
        orig_corners = tmp

        prev_frame_g = frame_g

        for i in range(len(prev_corners)):
            pt_o = (int(orig_corners[i].item(0)),int(orig_corners[i].item(1)))
            pt_p = (int(prev_corners[i].item(0)),int(prev_corners[i].item(1)))

            cv.circle(temp_frame,pt_o,2,(0,255,0),-1)
            cv.line(temp_frame,pt_o,pt_p,(0,0,255),1)
        
        if show:
            cv.imshow(location,temp_frame)
            key = cv.waitKey(0)
        
            
    for i in range(len(prev_corners)):
        pt_o = (int(orig_corners[i].item(0)),int(orig_corners[i].item(1)))
        pt_p = (int(prev_corners[i].item(0)),int(prev_corners[i].item(1)))

        cv.circle(frame,pt_o,2,(0,255,0),-1)
        cv.line(frame,pt_o,pt_p,(0,0,255),1)

    if show:
        cv.imshow(location,frame)
        key = cv.waitKey(0)

# main script
view = True
featureMatch('Parallel_Cube',view)
featureMatch('Parallel_Real',view)
featureMatch('Turned_Cube',view)
featureMatch('Turned_Real',view)

input('Press ENTER to close ...')

cv.destroyAllWindows()
