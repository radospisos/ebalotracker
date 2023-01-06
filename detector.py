import numpy as np
from yoloface.face_detector import YoloDetector

'''
Class with detection system.
'''

class Detector:
    def __init__(self):
        self.model = YoloDetector(target_size=720,min_face=90)

    def predict(self, image):
        bboxes,points = self.model.predict(image)
        return bboxes[0][0]
