#!/usr/bin/env python3

import cv2
import numpy as np
import time

# import tensorflow as tf
import torch


import detectron2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog



# print("Tensorflow: Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# if torch.cuda.is_available():
#     print("Torch: num available:", torch.cuda.device_count(), torch.cuda.get_device_name(0))
#     dev = "cuda:0" 
# else:
#     print("torch got no gpu")
#     dev = "cpu"  
# device = torch.device(dev)  





# model_yaml = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# model_yaml = "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"
# model_yaml = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"

model_yaml = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"

score_threshold = 0.7

cfg = get_cfg()   # get a fresh new config
cfg.merge_from_file(model_zoo.get_config_file(model_yaml))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_yaml)
predictor = DefaultPredictor(cfg)





def process_frame(frame):


    outputs = predictor(frame)
    v = Visualizer(frame, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))


    return out.get_image()



def main():

    cap = cv2.VideoCapture(0)

    while True:
        print()

        ret, frame = cap.read()
        orig_shape = frame.shape

        start_time = time.time()

        frame = process_frame(frame)
        
        if not ret:
      	    print("webcam failed")
      	    break

        print("Processing time in sec:", "%.8f" % (time.time()-start_time))

        frame = cv2.resize(frame, (orig_shape[1]*3, orig_shape[0]*3))
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == 27: # use ESC to quit
            break


    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()