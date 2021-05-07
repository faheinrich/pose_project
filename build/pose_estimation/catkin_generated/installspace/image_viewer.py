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


def callback(data):
  
    # Used to convert between ROS and OpenCV images
    br = CvBridge()

    # Output debugging information to the terminal
    rospy.loginfo("receiving video frame")

    # Convert ROS Image message to OpenCV image
    received_frame = br.imgmsg_to_cv2(data)
    new_shape = (received_frame.shape[1]*2, received_frame.shape[0]*2)
    received_frame = cv2.resize(received_frame, new_shape)

    # Display image
    cv2.imshow("webcam", received_frame)
    cv2.waitKey(1)
      
def receive_message():
  
    # Tells rospy the name of the node.
    # Anonymous = True makes sure the node has a unique name. Random
    # numbers are added to the end of the name. 
    rospy.init_node('image_viewer', anonymous=True)
      
    # Node is subscribing to the video_frames topic
    rospy.Subscriber('view_this', Image, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

    # Close down the video stream when done
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    receive_message()