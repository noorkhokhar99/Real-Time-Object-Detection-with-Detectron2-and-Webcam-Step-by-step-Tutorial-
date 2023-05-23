from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys, signal

class Detector:
    def __init__(self, model_type = "IS"):      # Preselected to Instance Segmentation
        self.cfg = get_cfg()
        self.model_type = model_type

        # Load model configuration and pretrained model type
        if model_type == "IS":      # Instance Segmenetation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        
        elif model_type == "PS":    # Panoptic Segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")

        elif model_type == "OD":    # Object Detection
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

        elif model_type == "KD":    # Keypoint Detection
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
        self.cfg.MODEL.DEVICE = "cpu"
        self.predictor = DefaultPredictor(self.cfg)

    def predictWebcam(self, path = 0):
        
        if path != 0:
            capture = cv2.VideoCapture(path)
        else:
            capture = cv2.VideoCapture("demo.mp4") #0 #1

        if (capture.isOpened()==False):
            print("Error with video file")
            return
        
        (read, frame) = capture.read()

        plt.ion()
        cam_plot = plt.imshow(frame)

        signal.signal(signal.SIGINT, self.exitWindow)
        exit_signal = False

        while read:
            (read, frame) = capture.read()

            if self.model_type == "PS":
                predictions, segments = self.predictor(frame)["panoptic_seg"]
                v = Visualizer(frame[:,:,::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
                output = v.draw_panoptic_seg_predictions(predictions.to("cpu"), segments)

            else:
                predictions = self.predictor(frame)
                v = Visualizer(frame[:,:,::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode = ColorMode.IMAGE)
                output = v.draw_instance_predictions(predictions["instances"].to("cpu"))

            pframe = cv2.cvtColor(output.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)    #predicted frame in RGB
            
            cam_plot.set_data(pframe)
            plt.draw()

            try:
                plt.pause(0.1)
            except Exception:
                pass

            if exit_signal: # Hit ctrl+c in terminal to exit
                break
            
        capture.release()
        cv2.destroyAllWindows()
        sys.exit(0)   

    def exitWindow(self, signal):
        global exit_signal
        exit_signal = True


if __name__ == "__main__":
    # Detector model types: Object Detection = "OD" / Instance Segmentation = "IS" / Panoptic Segmentation = "PS" / Keypoint Detection = "KD"
    detector = Detector(model_type="IS")
    detector.predictWebcam()