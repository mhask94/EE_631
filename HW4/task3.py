import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt

imgs_l = sorted(glob.glob('imgs/baseball/ballL*.bmp'))
imgs_r = sorted(glob.glob('imgs/baseball/ballR*.bmp'))

base_l = cv.imread(imgs_l[0])
base_r = cv.imread(imgs_r[0])
base_l = cv.cvtColor(base_l, cv.COLOR_BGR2GRAY)
base_r = cv.cvtColor(base_r, cv.COLOR_BGR2GRAY)

# starting location for the ROI to reduce processing
Xl = 345
Yl = 85
Xr = 265
Yr = 85
w = 50
h = 40

# Read in params
fs_read_l = cv.FileStorage('params/left_cam.yml',cv.FILE_STORAGE_READ)
mat_l = fs_read_l.getNode("intrinsic").mat()
dist_l = fs_read_l.getNode("distortion").mat()
fs_read_l.release()

fs = cv.FileStorage('params/rectify.yml',cv.FILE_STORAGE_READ)
Q = fs.getNode("Q").mat()
R1 = fs.getNode("R1").mat()
R2 = fs.getNode("R2").mat()
P1 = fs.getNode("P1").mat()
P2 = fs.getNode("P2").mat()
fs.release()

# variable to store the data for plotting
plot_vals = []

for file_l,file_r in zip(imgs_l,imgs_r):
    img_l = cv.imread(file_l)
    img_r = cv.imread(file_r)
    
    # crop base image to the current ROI
    crop_base_l = base_l[Yl:Yl+h,Xl:Xl+w]
    crop_base_r = base_r[Yr:Yr+h,Xr:Xr+w]

    # crop current image to the current ROI
    crop_l = img_l[Yl:Yl+h,Xl:Xl+w]
    crop_r = img_r[Yr:Yr+h,Xr:Xr+w]

    # convert to single channel image to binarize
    gray_l = cv.cvtColor(crop_l, cv.COLOR_BGR2GRAY)
    gray_r = cv.cvtColor(crop_r, cv.COLOR_BGR2GRAY)
    
    # take absolute difference of ROI and binarize
    diff_l = cv.absdiff(crop_base_l,gray_l)
    diff_r = cv.absdiff(crop_base_r,gray_r)
    _, diff_l = cv.threshold(diff_l,20,255,cv.THRESH_BINARY)
    _, diff_r = cv.threshold(diff_r,20,255,cv.THRESH_BINARY)

    # get rid of noise by eroding
    mask = np.ones((5,5),np.uint8)
    diff_l = cv.erode(diff_l,mask,iterations=1)
    diff_r = cv.erode(diff_r,mask,iterations=1)
#    diff_l = cv.dilate(diff_l,mask,iterations=1)
#    diff_r = cv.dilate(diff_r,mask,iterations=1)

    # calculate the moments and then the centroids
    M_l = cv.moments(diff_l)
    M_r = cv.moments(diff_r)

    if not M_l["m00"] == 0:
        x_l = M_l["m10"] / M_l["m00"]
        y_l = M_l["m01"] / M_l["m00"]
    else:
        x_l = 0.0
        y_l = 0.0
    if not M_r["m00"] == 0:
        x_r = M_r["m10"] / M_r["m00"]
        y_r = M_r["m01"] / M_r["m00"]
    else:
        x_r = 0.0
        y_r = 0.0

    if x_l != 0 and y_l != 0:
        ball_xl = x_l + Xl
        ball_yl = y_l + Yl
        ball_xr = x_r + Xr
        ball_yr = y_r + Yr
        
        img_pt = np.array([[[ball_xl,ball_yl]]])
        dst_img_pt = cv.undistortPoints(img_pt,mat_l,dist_l,R=R1,P=P1)
        dst_pt = np.array([[[dst_img_pt.item(0),dst_img_pt.item(1),ball_xl-ball_xr]]])
        ball_pos = cv.perspectiveTransform(dst_pt, Q)
        plot_vals.append(ball_pos[0])
#        print('ball_pos: \n',ball_pos[0])

        cv.circle(img_l,(int(ball_xl),int(ball_yl)),5,(0,0,255))
        cv.circle(img_r,(int(ball_xr),int(ball_yr)),5,(0,0,255))

    cv.imshow('left',img_l)
    cv.imshow('right',img_r)
#    cv.imshow('diff - left', diff_l)
#    cv.imshow('diff - right',diff_r)
#    cv.imshow('crop left',crop_l)
#    cv.imshow('crop right',crop_r)
    key = cv.waitKey(1)
    if key == ord('q'):
        break

    # calculate ROI for the next time through the loop
    if x_l != 0 and y_l != 0:
        Xl += int(x_l - 25)
        Yl += int(y_l - 20)
        Xr += int(x_r - w/2)
        Yr += int(y_r - 20)
        w += 1
        h += 2
    else:
        w = 50
        h = 40
    if Xl < 0:
        Xl = 0
    if Yl < 0:
        Yl = 0
    if Xr < 0:
        Xr = 0
    if Yr < 0:
        Yr = 0

plot_vals = np.array(plot_vals)
plot_vals = plot_vals.reshape((plot_vals.shape[0],3))
#print('plot_vals: \n',plot_vals)

# convert points from camera frame to catcher frame
x = plot_vals[:,0] - 10.0
y = plot_vals[:,1] - 29.9
z = plot_vals[:,2] - 21.0

# fit a least squares line or parabola to the data
time = np.linspace(0,len(x)-1,len(x)) / 60
time2 = np.linspace(0,0.819215528755)

A = np.vstack([time**2,time,np.ones(len(x))]).T
c = np.linalg.inv(A.T@A)@A.T@x.reshape(len(x),1)
x_fit = c.item(0)*time2**2 + c.item(1)*time2 + c.item(2)

c = np.linalg.inv(A.T@A)@A.T@y.reshape(len(y),1)
y_fit = c.item(0)*time2**2 + c.item(1)*time2 + c.item(2)

A = np.vstack([time,np.ones(len(x))]).T
c = np.linalg.inv(A.T@A)@A.T@z.reshape(len(z),1)
z_fit = c.item(0)*time2 + c.item(1)

print('Final x location: ',x_fit[-1],' (in)')
print('Final y location: ',y_fit[-1],' (in)')

#plt.figure(1)
#plt.plot(time,x,'b-',lw=2)
#plt.plot(time2,x_fit,'r--')
#plt.ylabel("X (in)")
#plt.xlabel("Time (s)")
#
#plt.figure(2)
#plt.plot(time,y,'b-',lw=2)
#plt.plot(time2,y_fit,'r--')
#plt.ylabel("Y (in)")
#plt.xlabel("Time (s)")
#
#plt.figure(3)
#plt.plot(time,z,'b-',lw=2)
#plt.plot(time2,z_fit,'r--')
#plt.ylabel("Z (in)")
#plt.xlabel("Time (s)")
#plt.show()

plt.figure(1)
plt.plot(x,z,'b-',lw=2)
plt.plot(x_fit,z_fit,'r--')
plt.plot(x_fit[-1],z_fit[-1],'ko')
plt.ylabel('z (in)')
plt.xlabel('x (in)')
plt.title('x vs. z trajectory')

plt.figure(2)
plt.plot(z,-y,'b-',lw=2)
plt.plot(z_fit,-y_fit,'r--')
plt.plot(z_fit[-1],-y_fit[-1],'ko')
plt.ylabel('-y (in)')
plt.xlabel('z (in)')
plt.title('z vs. -y trajectory')
plt.show()

cv.destroyAllWindows()
print('Program has Ended ...')

