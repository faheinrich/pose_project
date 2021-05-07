#!/usr/bin/env python3

import cv2
import numpy as np
import time

"""
Simple script to capture, process and display wecam image
"""




# def process_frame_with_cv2_net(frame):

#     """
#     https://learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/
#     """

#     # Specify the paths for the 2 files
#     protoFile = "/home/fabian/ros/cv2_net/pose/mpi/pose_deploy_linevec.prototxt"
#     weightsFile = "/home/fabian/ros/cv2_net/pose/mpi/pose_iter_160000.caffemodel"
#     # Read the network into Memory
#     net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

#     inWidth = 1000
#     inHeight = 1000

#     threshold = 0.6


#     inHeight = frame.shape[0]
#     inWidth = frame.shape[1]


#     # frame = cv2.resize(frame, (inHeight, inWidth))
#     # Prepare the frame to be fed to the network
#     inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
#     # Set the prepared object as the input blob of the network
#     net.setInput(inpBlob)

#     output = net.forward()


#     H = output.shape[2]
#     W = output.shape[3]

#     # Empty list to store the detected keypoints
#     points = []
#     # 44 for mpi
#     for i in range(44):
#         # confidence map of corresponding body's part.
#         probMap = output[0, i, :, :]

#         # Find global maxima of the probMap.
#         minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

#         # Scale the point to fit on the original image
#         x = (inWidth * point[0]) / W
#         y = (inHeight * point[1]) / H
#         if prob > threshold :
#             cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
#             # cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)
#             # Add the point to the list if the probability is greater than the threshold
#             points.append((int(x), int(y)))
#         else :
#             points.append(None)

#     return frame


def process_frame(frame):
    start_time = time.time()

    frame = frame[:,::-1,::-1]

    print("Processing time in sec:", "%.8f" % (time.time()-start_time))
    return frame




if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()


        processed_frame = process_frame(frame)

        cv2.imshow("Webcam", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == 27: # use ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()
