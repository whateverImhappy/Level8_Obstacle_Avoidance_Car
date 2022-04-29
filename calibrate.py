import numpy as np 
import cv2
from tqdm import tqdm

# Set the path to the images captured by the left and right cameras
#pathL = "/home/pi/small_car/data/stereoL/"
#pathR = "/home/pi/small_car/data/stereoR/"
pathL = "/home/pi/small_car/data/L/"
pathR = "/home/pi/small_car/data/R/"
print("Extracting image coordinates of respective 3D pattern ....\n")
CHECKBOARD = (5,7)
# Termination criteria for refining the detected corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

img_ptsL = []
img_ptsR = []
obj_pts = []

objp = np.zeros((1,CHECKBOARD[0]*CHECKBOARD[1] ,3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKBOARD[0],0:CHECKBOARD[1]].T.reshape(-1,2)



for i in tqdm(range(1,15)):
    imgL = cv2.imread(pathL+"img%d.png"%i,0)
    imgR = cv2.imread(pathR+"img%d.png"%i,0)
    imgL_gray = cv2.imread(pathL+"img%d.png"%i,0)
    imgR_gray = cv2.imread(pathR+"img%d.png"%i,0) #cv2.COLOR_BGR2GRAY

    outputL = imgL.copy()
    outputR = imgR.copy()

    retR, cornersR =  cv2.findChessboardCorners(outputR,CHECKBOARD ,None)#cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    retL, cornersL = cv2.findChessboardCorners(outputL,CHECKBOARD,None)#cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if retR and retL:
        obj_pts.append(objp)
        #cornersR2 = cv2.cornerSubPix(imgR_gray,cornersR,(11,11),(-1,-1),criteria)
        #cornersL2 = cv2.cornerSubPix(imgL_gray,cornersL,(11,11),(-1,-1),criteria)
        cv2.cornerSubPix(imgR_gray,cornersR,(11,11),(-1,-1),criteria)
        cv2.cornerSubPix(imgL_gray,cornersL,(11,11),(-1,-1),criteria)
        #img_ptsL.append(cornersL2)
        #img_ptsR.append(cornersR2)
        cv2.drawChessboardCorners(outputR,CHECKBOARD,cornersR,retR)
        cv2.drawChessboardCorners(outputL,CHECKBOARD,cornersL,retL)
        #outputR2 = cv2.drawChessboardCorners(outputR,CHECKBOARD,cornersR2,retR)
        #outputL2 =cv2.drawChessboardCorners(outputL,CHECKBOARD,cornersL2,retL)
        #cv2.imshow('cornersR',outputR2)
        #cv2.imshow('cornersL',outputL2)
        #cv2.waitKey(0)
    

        img_ptsL.append(cornersL)
        img_ptsR.append(cornersR)
    #cv2.destroyAllWindows()
    '''
    obj_pts.append(objp)
    cornersR2 = cv2.cornerSubPix(imgR_gray,cornersR,(11,11),(-1,-1),criteria)
    cornersL2 = cv2.cornerSubPix(imgL_gray,cornersL,(11,11),(-1,-1),criteria)

    img_ptsL.append(cornersL2)
    img_ptsR.append(cornersR2)
        
    outputR2 = cv2.drawChessboardCorners(outputR,CHECKBOARD,cornersR2,retR)
    outputL2 =cv2.drawChessboardCorners(outputL,CHECKBOARD,cornersL2,retL)
    #cv2.imshow('cornersR',outputR2)
    #cv2.imshow('cornersL',outputL2)
    #cv2.waitKey(0)
    

    img_ptsL.append(cornersL)
    img_ptsR.append(cornersR)
    '''
print("Calculating left camera parameters ... ")
# Calibrating left camera

#print(type(imgL_gray.shape[::-1]))
hL,wL= imgL_gray.shape[:2]
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts,img_ptsL,imgL_gray.shape[::-1],None,None)

new_mtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))

print("Calculating right camera parameters ... ")
# Calibrating right camera
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts,img_ptsR,imgL_gray.shape[::-1],None,None)
hR,wR= imgR_gray.shape[:2]
new_mtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,(wR,hR),1,(wR,hR))


print("Stereo calibration .....")
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same 

criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(obj_pts,
                                                                                    img_ptsL,
                                                                                    img_ptsR,
                                                                                    new_mtxL,
                                                                                    distL,
                                                                                    new_mtxR,
                                                                                    distR,
                                                                                    imgL_gray.shape[::-1],
                                                                                    criteria_stereo,
                                                                                    flags)

# Once we know the transformation between the two cameras we can perform stereo rectification
# StereoRectify function
rectify_scale= 1 # if 0 image croped, if 1 image not croped
rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR= cv2.stereoRectify(new_mtxL,
                                                                         distL,
                                                                         new_mtxR,
                                                                         distR,
                                                                         imgL_gray.shape[::-1],
                                                                         Rot,
                                                                         Trns,
                                                                         rectify_scale,
                                                                         (0,0))

# Use the rotation matrixes for stereo rectification and camera intrinsics for undistorting the image
# Compute the rectification map (mapping between the original image pixels and 
# their transformed values after applying rectification and undistortion) for left and right camera frames
Left_Stereo_Map= cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l,
                                             imgL_gray.shape[::-1], cv2.CV_16SC2)
Right_Stereo_Map= cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r,
                                              imgR_gray.shape[::-1], cv2.CV_16SC2)


print("Saving paraeters ......")
#cv_file = cv2.FileStorage("data/params_py.xml", cv2.FILE_STORAGE_WRITE)
cv_file = cv2.FileStorage("/home/pi/car2/stereo_rectify_maps.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("Left_Stereo_Map_x",Left_Stereo_Map[0])
cv_file.write("Left_Stereo_Map_y",Left_Stereo_Map[1])
cv_file.write("Right_Stereo_Map_x",Right_Stereo_Map[0])
cv_file.write("Right_Stereo_Map_y",Right_Stereo_Map[1])
cv_file.release()
print("finish")
