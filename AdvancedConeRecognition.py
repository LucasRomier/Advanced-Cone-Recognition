"""

Author: Lucas Romier

"""

import cv2
import numpy
# from OpenCVRecognition import OpenCVRecognition
from DarknetRecognition import DarknetRecognition
import os


class AdvancedConeRecognition:
    videoCapture = None
    running = False
    darknet_recognition = None

    def __init__(self):
        cwd = os.path.dirname(__file__)
        res_wd = os.path.join(cwd, 'res').replace("\\", "/")
        os.environ['PATH'] = res_wd + ";" + cwd + ';' + os.environ['PATH']

        cfg_path = os.path.join(res_wd, 'cones-yolo.cfg')
        weight_path = os.path.join(res_wd, 'cones-yolo_1000.weights')
        dll_path = os.path.join(res_wd, 'yolo_cpp_dll.dll')

        """ Read out labels and generate random colors """
        names_path = os.path.join(res_wd, 'cones.names')
        self.LABELS = open(names_path).read().strip().split("\n")
        numpy.random.seed(42)
        self.COLORS = numpy.random.randint(0, 255, size=(len(self.LABELS), 3), dtype="uint8")

        # self.openCV_recognition = OpenCVRecognition(cfg_path, weight_path)
        self.darknet_recognition = DarknetRecognition(dll_path, cfg_path, weight_path)
        self.videoCapture = cv2.VideoCapture(0)

        if self.videoCapture.isOpened():
            self.img_w = self.videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
            self.img_h = self.videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

    def start(self, render=False, verbose=False):
        self.running = True

        """ Start reading video """
        while self.videoCapture.isOpened() and self.running:
            (grabbed, frame) = self.videoCapture.read()
            if not grabbed:
                break

            # detections = self.openCV_recognition.predict_boxes(frame)
            detections = self.darknet_recognition.predict_boxes(frame, len(self.LABELS))

            # ensure at least one detection exists
            if len(detections) > 0:
                # loop over the indexes we are keeping
                for (clazz, probability, (x, y, w, h)) in detections:

                    (x1, y1, x2, y2) = self.darknet_recognition.to_usable(x, y, w, h)
                    x = int(x)
                    y = int(y)
                    w = int(w)
                    h = int(h)

                    if verbose:
                        print(x, y, w, h, self.LABELS[clazz])

                    if render:
                        # draw a bounding box rectangle and label on the frame
                        color = [int(c) for c in self.COLORS[clazz]]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.circle(frame, (x, y), 10, color, -1)
                        text = "{}: {:.2f}".format(self.LABELS[clazz], probability)
                        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if render:
                cv2.imshow('Currently processed image', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        """ Free video capture """
        self.videoCapture.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False


if __name__ == '__main__':
    advanced_cone_recognition = AdvancedConeRecognition()
    advanced_cone_recognition.start(render=True, verbose=True)
