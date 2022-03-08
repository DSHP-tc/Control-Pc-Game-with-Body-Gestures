from concurrent.futures import thread
import math
import pydirectinput as pd
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
import threading
x=0
thread_running=False
last_value=320
walk_flag=False
def sendInputToGame():
    global x
    while True:
        if thread_running==True:
            if x==0:
                pd.move(xOffset=-2,yOffset=0,duration=0,relative=True)
            else:
                pd.move(xOffset=2,yOffset=0,duration=0,relative=True)
        else:
            break

th = threading.Thread(target=sendInputToGame())




def checkState(landmarks, output_image, display=False):
    global last_value,walk_flag,thread_running,th,x
    
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Not Running'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)

    # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Get the angle between the right shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
    

    if (left_elbow_angle<90 or left_elbow_angle>300) and (right_elbow_angle<90 or right_elbow_angle>270) :
        label="Running"
    toe_right_y=landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value][1]
    toe_left_y=landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value][1]

    rthumb_y=landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value][1]
    lthumb_y=landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value][1]
    rthumb_x=landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value][0]
    lthumb_x=landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value][0]
    nose_y=landmarks[mp_pose.PoseLandmark.NOSE.value][1]

    if rthumb_y<nose_y:
        pd.move(xOffset=-5,yOffset=0,duration=0,relative=True)
    elif lthumb_y<nose_y:
        pd.move(xOffset=5,yOffset=0,duration=0,relative=True)
        
            
  

    if label=="Running":
        if abs(toe_right_y-toe_left_y)>5 or abs(toe_left_y-toe_right_y)>5:
            walk_flag=True
        else:
            walk_flag=False
    else:
        if abs(toe_right_y-toe_left_y)<5:
            walk_flag=False
     

    if walk_flag==True:
        pd.keyDown("w")
    else:
        pd.keyUp("w")
        


    if label != 'Not Running':
        
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    # Check if the resultant image is specified to be displayed.
    if display:
    
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1])
        plt.title("Output Image")
        plt.axis('off')
        
    else:
        
        # Return the output image and the classified label.
        return output_image, label

def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than zero.
    if angle < 0:

        # Add 360 to the found angle.
        angle += 360
    
    # Return the calculated angle.
    return angle

def detectPose(image, pose, display=True):
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
    
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
    # Otherwise
    else:
        
        # Return the output image and the found landmarks.
        return output_image, landmarks

######################################################################################################
mp_pose = mp.solutions.pose

mp_drawing = mp.solutions.drawing_utils

pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

video = cv2.VideoCapture(0)

camera_video = cv2.VideoCapture(0)
######################################################################################################



while camera_video.isOpened():
    
    ok, frame = camera_video.read()
    
    if not ok:
        continue

    frame = cv2.flip(frame, 1)
  
    frame_height, frame_width, _ =  frame.shape

    frame, landmarks = detectPose(frame, pose_video, display=False)
    
    # Check if the landmarks are detected.
    if landmarks:
        frame, _ = checkState(landmarks, frame, display=False)
  

    cv2.line(img=frame, pt1=(frame_width//2,0), pt2=(frame_width//2, frame_height), 
    color=(255, 0, 0), thickness=2)
    cv2.line(img=frame, pt1=(0,frame_height//2), pt2=(frame_width, frame_height//2), 
    color=(255, 0, 0), thickness=2)


    cv2.imshow('Pose Classification', frame)
    
    # Wait until a key is pressed.
    # Retreive the ASCII code of the key pressed
    k = cv2.waitKey(1) & 0xFF
    
    # Check if 'ESC' is pressed.
    if(k == 27):
        
        # Break the loop.
        break

# Release the VideoCapture object and close the windows.
camera_video.release()
cv2.destroyAllWindows()