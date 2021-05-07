#!/usr/bin/env python3
# Description:
# - Subscribes to real-time streaming video from your built-in webcam.
#
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com
 
# Import the necessary libraries
import rospy # Python library for ROS
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
 
# import sys
# import time
# import logging
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2

# from tf_pose import common
# from tf_pose.estimator import TfPoseEstimator
# from tf_pose.networks import get_graph_path, model_wh


pub = rospy.Publisher('view_this', Image, queue_size=1)


# """
# https://learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/
# """

# # Specify the paths for the 2 files
# protoFile = "/home/fabian/ros/catkin_ws/resources/cv2_net/pose/mpi/pose_deploy_linevec.prototxt"
# weightsFile = "/home/fabian/ros/catkin_ws/resources/cv2_net/pose/mpi/pose_iter_160000.caffemodel"
# # Read the network into Memory
# net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# inWidth = 368
# inHeight = 368

# threshold = 0.6

def process_frame(frame):


    # frame = cv2.resize(frame, (inHeight, inWidth))

    # # frame = cv2.resize(frame, (inHeight, inWidth))
    # # Prepare the frame to be fed to the network
    # inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    # # Set the prepared object as the input blob of the network
    # net.setInput(inpBlob)


    # output = net.forward()

    # rospy.loginfo(output.shape)


    # H = output.shape[2]
    # W = output.shape[3]

    # # Empty list to store the detected keypoints
    # points = []
    # # 44 for mpi
    # for i in range(44):
    #     # confidence map of corresponding body's part.
    #     probMap = output[0, i, :, :]

    #     # Find global maxima of the probMap.
    #     minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    #     # Scale the point to fit on the original image
    #     x = (inWidth * point[0]) / W
    #     y = (inHeight * point[1]) / H
    #     if prob > threshold :
    #         cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
    #         # cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)
    #         # Add the point to the list if the probability is greater than the threshold
    #         points.append((int(x), int(y)))
    #     else :
    #         points.append(None)



    # cv2.imshow("Output-Keypoints",frame)
    # cv2.waitKey()

        
    # for pair in POSE_PAIRS:
    #     partA = pair[0]
    #     partB = pair[1]
    #     if points[partA] and points[partB]:
    #         cv2.line(frameCopy, points[partA], points[partB], (0, 255, 0), 3)





    
   

    # from rgb to bgr to show change
    return frame[:,:,::-1]

def callback(data):
    
    # Used to convert between ROS and OpenCV images
    br = CvBridge()
    
    # Output debugging information to the terminal
    rospy.loginfo("receiving video frame")

    # Convert ROS Image message to OpenCV image
    received_frame = br.imgmsg_to_cv2(data)

    rospy.loginfo('processing received image')

    processed_frame = process_frame(received_frame)

    pub.publish(br.cv2_to_imgmsg(processed_frame))

def receive_message():
    
    # Tells rospy the name of the node.
    # Anonymous = True makes sure the node has a unique name. Random
    # numbers are added to the end of the name. 
    rospy.init_node('image_processor', anonymous=True)
    
    # Node is subscribing to the video_frames topic
    rospy.Subscriber('webcam_frames', Image, callback)
    

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
    
    # Close down the video stream when done
    cv2.destroyAllWindows()
  
if __name__ == '__main__':
    
    receive_message()
