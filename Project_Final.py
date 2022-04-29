# Haohan Zhu - Smart Video Car Kit - Obstacle Avoidance Car
from picar import front_wheels, back_wheels
from picar.SunFounder_PCA9685 import Servo
import picar
import time
import cv2
import numpy as np
import picar
import os
picar.setup()
db_file = "/home/pi/small_car/SunFounder_PiCar-V/remote_control/remote_control/driver/config"
fw = front_wheels.Front_Wheels(debug=False, db=db_file)
bw = back_wheels.Back_Wheels(debug=False, db=db_file)
pan_servo = Servo.Servo(1)
tilt_servo = Servo.Servo(2)
bw.ready()
fw.ready()
#env = None
cv_file = cv2.FileStorage("stereo_rectify_maps.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()

#environment_state='safe'
def left_image(left_detection):
    if left_detection == True:
        print("begin capture the left image")
        pan_angle = 95
        tilt_angle = 90
        pan_servo.write(pan_angle)
        tilt_servo.write(tilt_angle)
        
        capture_duration = 3
        cap = cv2.VideoCapture(0)
        left_fourcc = cv2.VideoWriter_fourcc(*'XVID')
        left_out = cv2.VideoWriter('/home/pi/car2/data/video/L/left.avi', left_fourcc, 30.0, (640,480))
        
        start_time = time.time()
        while(int(time.time() - start_time) < capture_duration):
            ret,frame_left = cap.read()
            if not ret:
                print("Not ret")
                break
            timer = capture_duration - int(time.time() - start_time)
            imgL_temp = frame_left.copy()
            left_out.write(imgL_temp)
            cv2.imshow("left_orig",frame_left)
            cv2.waitKey(2)
        cap.release()
        left_out.release()
        cv2.destroyAllWindows()
        print("left image done")
    else:
        print("left image pass")
        pass

def right_image(right_detection):
    if right_detection == True:
        print("capture right img")
        pan_angle = 85
        tilt_angle = 90
        pan_servo.write(pan_angle)
        tilt_servo.write(tilt_angle)
        
        capture_duration = 3
        cap = cv2.VideoCapture(0)
        right_fourcc = cv2.VideoWriter_fourcc(*'XVID')
        right_out = cv2.VideoWriter('/home/pi/car2/data/video/R/right.avi', right_fourcc, 30.0, (640,480))
        start_time = time.time()
        while(cap.isOpened):
            ret,frame_right = cap.read()
            if not ret:
                print("Not ret")
                break
            right_out.write(frame_right)
            cv2.imshow("right_orig",frame_right)
            cv2.waitKey(1)
            if time.time() - start_time > capture_duration:
                break
        cap.release()
        right_out.release()
        cv2.destroyAllWindows()
        print("right image was captured done")
    else:
        print("right image pass")
        pass
    
def capture_obstacle_image(env):
    if env == 'danger':
        print("begin to capture the obstacle image")
        time.sleep(0.1)
        left_detection= True
        left_image(left_detection)
        time.sleep(0.1)
        right_detection= True
        right_image(right_detection)
        time.sleep(0.1)
        pan_servo.write(90)
        tilt_servo.write(90)
    else:
        print("capture_obstacle_image pass")
        pass
    
    
disparity = None
depth_map = None


def obstacle_distance_detection():
    def mouse_click(event,x,y,flags,param):
        global Z
        if event == cv2.EVENT_LBUTTONDBLCLK:
            #print(disparity[y,x])
            if (disparity[y,x]) > 0:
                depth_map = 22.5/disparity
                print("Distance = %.2f cm"%depth_map[y,x])
    cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('disp',600,600)
    cv2.setMouseCallback('disp',mouse_click)
    # Creating an object of StereoBM algorithm
    cv2.createTrackbar('num','disp',9,30, lambda x: None)
    cv2.createTrackbar('blockSize','disp',15,255,lambda x: None)
    output_canvas = None
    max_depth = 60 # maximum distance the setup can measure (in cm)
    min_depth = 15 # minimum distance the setup can measure (in cm)
    depth_thresh = 30 
    def obstacle_avoid():
        mask = cv2.inRange(depth_map, 5, depth_thresh)
        if np.sum(mask)/255.0 > 0.01*mask.shape[0]*mask.shape[1]:
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(contours, key=cv2.contourArea, reverse=True)

            # Check if detected contour is significantly large (to avoid multiple tiny regions)
            if cv2.contourArea(cnts[0]) > 0.01*mask.shape[0]*mask.shape[1]:
                x,y,w,h = cv2.boundingRect(cnts[0])

                # finding average depth of region represented by the largest contour 
                mask2 = np.zeros_like(mask)
                cv2.drawContours(mask2, cnts, 0, (255), -1)

                # Calculating the average depth of the object closer than the safe distance
                depth_mean, _ = cv2.meanStdDev(depth_map, mask=mask2)
                
                # Display warning text
                cv2.putText(output_canvas, "WARNING !", (x+5,y-40), 1, 2, (0,0,255), 2, 2)
                cv2.putText(output_canvas, "Object at", (x+5,y), 1, 2, (100,10,25), 2, 2)
                cv2.putText(output_canvas, "%.2f cm"%depth_mean, (x+5,y+40), 1, 2, (0,255,0), 2, 2)
                #env = "danger"
        else:
            cv2.putText(output_canvas, "SAFE!", (100,100),1,3,(0,255,0),2,3)
            #env = "safe"

        cv2.imshow('output_canvas',output_canvas)
        #return env
        
    CamL = cv2.VideoCapture("/home/pi/car2/data/video/L/left.avi")
    CamR = cv2.VideoCapture("/home/pi/car2/data/video/R/right.avi")
    while True:
        retR, imgR= CamR.read()
        retL, imgL= CamL.read()
        if retL and retR:
            output_canvas = imgL.copy()
            Left_nice= cv2.remap(imgL,
                                 Left_Stereo_Map_x,
                                 Left_Stereo_Map_y,
                                 cv2.INTER_LANCZOS4,
                                 cv2.BORDER_CONSTANT,
                                 0)
            Right_nice= cv2.remap(imgR,
                                  Right_Stereo_Map_x,
                                  Right_Stereo_Map_y,
                                  cv2.INTER_LANCZOS4,
                                  cv2.BORDER_CONSTANT,
                                  0)
            imgR_gray = cv2.cvtColor(Right_nice,cv2.COLOR_BGR2GRAY)
            imgL_gray = cv2.cvtColor(Left_nice,cv2.COLOR_BGR2GRAY)
            num = cv2.getTrackbarPos('num','disp')
            blockSize = cv2.getTrackbarPos('blockSize','disp')
            numDisparities = 16*num
            if blockSize % 2 == 0:
                blockSize += 1
            if blockSize <5:
                blockSize = 5
            stereo = cv2.StereoBM_create(numDisparities = 16*num,
                                         blockSize = 31)
            disparity = stereo.compute(imgL_gray,imgR_gray)
            disparity = disparity.astype(np.float32)
            disparity = (disparity/16.0 )/numDisparities
            
            depth_map = 22.5/disparity
            cv2.resizeWindow("disp",700,700)
            cv2.imshow("disp",disparity)
            max_depth = 60 # maximum distance the setup can measure (in cm)
            min_depth = 15 # minimum distance the setup can measure (in cm)
            mask_temp = cv2.inRange(depth_map,min_depth,max_depth)
            depth_map = cv2.bitwise_and(depth_map,depth_map,mask=mask_temp)
            #print(depth_map.shape)
            #cv2.imshow("mask",depth_map)
            obstacle_avoid()
            cv2.waitKey(20)
        else:
            break
        #following_F(obstacle_avoid())
        #time.sleep(1)
        #bw.speed = 0
        #bw.stop()
    
detection =None
following =None
#def lane_following(detection,following):
def detection_F(detection):
    if detection == True:
        print("detect if there are any obj")
        environment_state = None
        pan_angle = 90 # greater than 90 left ; less than 90 right
        tilt_angle = 90 
        pan_servo.write(pan_angle)
        tilt_servo.write(tilt_angle)
        
        num_lane_point = 6
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        capture_duration = 5
        while cap.isOpened():
            ret,frame = cap.read()
            if not ret:
                print("Function lane_following cannot read the ret")
                break
            original_frame = np.copy(frame)
            cv2.imshow("original_frame",original_frame)
            img = cv2.blur(original_frame,(5,5))
            _, _, red_img = cv2.split(img)
            _, dst = cv2.threshold(red_img, 120, 255, cv2.THRESH_BINARY)
            height, width = dst.shape # [480,640]
            half_width = int (width/2) # 320
            right_line_pos = np.zeros((num_lane_point, 1)) # num_lane_point = 4
            left_line_pos = np.zeros((num_lane_point, 1))
            img_out = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)
            cv2.imshow('img_out', img_out)
            
            for i in range(num_lane_point):   # each detected point on the lane
                detect_height = 440 - 20 * (i+1)
                detect_area_left = dst[detect_height, 0: half_width - 1]  # divide the image into two parts: left and right (this may cause problems, which can be optimized in the future)
                detect_area_right = dst[detect_height, half_width: width-1]
            line_left = np.where(detect_area_left == 0)   # extract  zero pixels' index
            line_right = np.where(detect_area_right == 0)
            if len(line_left[0]):
                left_line_pos[i] = int(np.max(line_left))  # set the most internal pixel as the lane point
            else:
                left_line_pos[i] = 0  # if haven't detected any zero pixel, set the lane point as 0

            if len(line_right[0]):
                right_line_pos[i] = int(np.min(line_right))
            else:
                right_line_pos[i] = half_width - 1               

            if left_line_pos[i] != 0:   # draw the lane points on the binary image
                img_out = cv2.circle(img_out, (int(left_line_pos[i]), int(detect_height)),
                                     4, (0, 0, 255), thickness=10)
            if right_line_pos[i] != half_width - 1:
                img_out = cv2.circle(img_out, (int(half_width + right_line_pos[i]), int(detect_height)),
                                     4, (0, 0, 255), thickness=10)
            cv2.imshow("output",img_out)
            left_max = np.max(left_line_pos)
            right_min = np.max(right_line_pos)
            #print("left dist:",left_max)
            #print("right dist:",half_width + right_min)
            #print(half_width)
            
            if left_max == 0 and right_min==half_width-1:
                environment_state ='danger'
            #elif half_width+right_min - left_max < 340:
            #    environment_state ='danger'
            elif half_width+right_min - left_max < 10:
                environment_state ='danger'
            else:
                environment_state = 'safe'

            cv2.waitKey(1)
            if time.time() - start_time > capture_duration:
                break
        cap.release()
        cv2.destroyAllWindows()
        if environment_state == 'danger':
            print("There is an obstacle in the safe distance")
            bw.stop()
            env = environment_state
            #return env 
        else:
            print("safe")
            env = environment_state
        
        print("environment_state in the dete func:",env)
        return env
            
    else:
        pass

def following_F(env):
    print("access the following function")
    if env == 'safe':
        print("environment_state in the follow func:",env)
        print("Following Function begin to execute")
        num_lane_point = 4 # detect the particular point in a area
        SPEED = 60
        pan_angle = 90 # greater than 90 left ; less than 90 right
        tilt_angle = 90 
        pan_servo.write(pan_angle)
        tilt_servo.write(tilt_angle)
        bw_state = 'forward'
        fw_state = 'straight'
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("no Video input")
            original_frame = np.copy(frame)
            
            img = cv2.blur(original_frame,(5,5))
            #cv2.imshow('blur', img)
            
            _, _, red_img = cv2.split(img)
            #cv2.imshow("red_channel", red_img)
            
            _, dst = cv2.threshold(red_img, 120, 255, cv2.THRESH_BINARY)
            #cv2.imshow("gray",dst)
            #print(str(dst.shape))
            height, width = dst.shape # [480,640]
            half_width = int (width/2) # 320
            
            right_line_pos = np.zeros((num_lane_point, 1)) # num_lane_point = 4
            left_line_pos = np.zeros((num_lane_point, 1))
            
            img_out = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)
            cv2.imshow('img_out', img_out)
            
            for i in range(num_lane_point):   # each detected point on the lane
                detect_height = height - 20 * (i+1)
                detect_area_left = dst[detect_height, 0: half_width - 1]  # divide the image into two parts: left and right (this may cause problems, which can be optimized in the future)
                detect_area_right = dst[detect_height, half_width: width-1]
            line_left = np.where(detect_area_left == 0)   # extract  zero pixels' index
            line_right = np.where(detect_area_right == 0)
            if len(line_left[0]):
                left_line_pos[i] = int(np.max(line_left))  # set the most internal pixel as the lane point
            else:
                left_line_pos[i] = 0  # if haven't detected any zero pixel, set the lane point as 0

            if len(line_right[0]):
                right_line_pos[i] = int(np.min(line_right))
            else:
                right_line_pos[i] = half_width - 1               

            if left_line_pos[i] != 0:   # draw the lane points on the binary image
                img_out = cv2.circle(img_out, (int(left_line_pos[i]), int(detect_height)),
                                     4, (0, 0, 255), thickness=10)
            if right_line_pos[i] != half_width - 1:
                img_out = cv2.circle(img_out, (int(half_width + right_line_pos[i]), int(detect_height)),
                                     4, (0, 0, 255), thickness=10)
            cv2.imshow("output",img_out)
            left_max = np.max(left_line_pos)
            right_min = np.max(right_line_pos)
            print(left_max)
            print(right_min)
            #print(half_width)
                
            if left_max == 0 and right_min == half_width - 1:
                    pass
            elif left_max == 0: 
                if right_min > half_width - 100:
                    fw_state = 'Straight'
                    bw_state = 'Forward'
                    print("forward and turn straight")
                elif right_min < 100:
                    fw_state = 'Left'
                    bw_state = 'Brake'
                    print("stop and turn left")
                else:   
                    fw_state = 'Left'
                    bw_state = 'Forward'
                    print("forward and turn left")
            elif right_min == half_width - 1:
                if left_max <100:
                    fw_state = 'Straight'
                    bw_state = 'Forward'
                    print("forward and turn straight")
                elif left_max > half_width - 100:
                    fw_state = 'Right'
                    bw_state = 'Brake'
                    print("stop and turn right")
                else:
                    fw_state = 'Right'
                    bw_state = 'Forward'
                    print("forward and turn right")
            else:
                fw_state = 'Straight'
                bw_state = 'Forward'
                print("Go straight!")
                     
            #motion control
            if bw_state == 'Brake':
                if fw_state == 'Left':
                    bw.speed = SPEED - SPEED
                    bw.stop()
                    fw.turn_left()
                elif fw_state == 'Right':
                    bw.speed = SPEED - SPEED
                    bw.stop()
                    fw.turn_right()
                elif fw_state == 'Straight':
                    bw.speed = SPEED - SPEED
                    bw.stop()
                    fw.turn_straight()
            elif bw_state == 'Forward':
                if fw_state == 'Left':
                    bw.speed = SPEED
                    bw.forward()
                    fw.turn_left()
                elif fw_state == 'Right':
                    bw.speed = SPEED
                    bw.forward()
                    fw.turn_right()
                elif fw_state == 'Straight':
                    bw.speed = SPEED
                    bw.forward()
                    fw.turn_straight()
            elif bw_state == 'Backward':
                if fw_state == 'Left':
                    bw.speed = SPEED
                    bw.backward()
                    fw.turn_left()
                elif fw_state == 'Right':
                    bw.speed = SPEED
                    bw.backward()
                    fw.turn_right()
                elif fw_state == 'Brake':
                    bw.speed = SPEED
                    bw.backward()
                    fw.turn_straight()
            
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break
    else:
        print("environment_state in the follow func: ", env, "stop follow")
        print("the state of the following is False")
        pass

if __name__ == "__main__":
    detection = True
    #detection_F(detection)
    
    #print("environment_state in main:",env)
    following_F(detection_F(detection))
    capture_obstacle_image(detection_F(detection))
    obstacle_distance_detection()
