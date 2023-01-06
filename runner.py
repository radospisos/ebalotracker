import cv2
from tracker import Tracker
from detector import Detector

'''
Class running detection and tracking system. Getting video stream from camera #0.

    Attributes:
    -----------
    cap : VideoCapture
        cv2.VideoCapture object of video stream.
    tracking_flag: bool
        if flag is True => there is a detected object which should be tracked, if False => there aren't any detected objects.
    tracker: Tracker
        Tracker object
    detector: Detector
        Detector object

    Methods:
    --------
    run()
        Runs detection & tracking system.

'''
class Runner:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.tracking_flag = False
        self.tracker = Tracker()
        self.detector = Detector()

    def run(self):
        # Video streaming loop.
        while True:
            # Getting frame.
            ret, frame = self.cap.read()
            if not ret:
                break

            # CNN and tracker entry points.
            if self.tracking_flag:
                # If tracking was started, updating object position.
                ok, bbox = self.tracker.update(frame)

                if ok:
                    # If tracker succesfully updated object position, draw bounding box.
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0, 0, 255), 1)
                else:
                    # else stop tracking => on the next frame try to detect again.
                    self.tracking_flag = False
            else:
                # If tracking was not started, trying to detect an object.
                bbox = self.detector.predict(frame)

                # If detector find an object.
                if len(bbox) > 0:
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0, 255, 255), 1)
                    self.tracker.init(frame, bbox)
                    self.tracking_flag = True

            # Show video.
            cv2.imshow('stream', frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
