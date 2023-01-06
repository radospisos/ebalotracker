import cv2

'''
Tracker class.

    Attributes:
    -----------
    parameters_path : string
        Path to JSON file with tracker parameters (absolute path is recommended).
    bbox: list
        Bounding box.
    tracker: cv2.legacy.TrackerMedianFlow
        MedianFlow tracker

    Methods:
    --------
    init(frame, bbox)
        Initialize tracker on given frame with given bounding box.
    update(frame)
        Update object position on given frame, returns predicted bbox.
'''

class Tracker:
    def __init__(self):
        self.parameters_path="medianflow_params.json"
        self.bbox = []
        self.tracker = cv2.legacy.TrackerMedianFlow_create()
        fs = cv2.FileStorage(self.parameters_path, cv2.FileStorage_READ)
        self.tracker.read(fs.getFirstTopLevelNode())

    def init(self, frame, bbox):
        self.bbox = bbox
        self.tracker.init(frame, self.bbox)
        print('MedianFlow tracker was inited successfully.')

    def update(self, frame):
        ok, self.bbox = self.tracker.update(frame)
        print('Tracking flag:', ok, 'bbox:', self.bbox)
        return ok, self.bbox
