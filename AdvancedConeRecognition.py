"""

Author: Lucas Romier
Required packages: numpy, opencv-contrib-python, ffmpeg

"""
from datetime import datetime
import cv2
import numpy
# from OpenCVRecognition import OpenCVRecognition
from DarknetRecognition import DarknetRecognition
from SimpleDistanceCalculation import SimpleDistanceCalculation
import os


class AdvancedConeRecognition:
    videoCapture = None
    running = False
    out = None

    # openCV_recognition = None
    darknet_recognition = None

    distance_calculation = None

    def __init__(self):
        self.cwd = os.path.dirname(__file__)
        res_wd = os.path.join(self.cwd, 'res').replace("\\", "/")
        os.environ['PATH'] = res_wd + ";" + self.cwd + ';' + os.environ['PATH']

        cfg_path = os.path.join(res_wd, 'cones-yolo.cfg').replace("\\", "/")
        weight_path = os.path.join(res_wd, 'cones-yolo_1000.weights').replace("\\", "/")
        dll_path = os.path.join(res_wd, 'yolo_cpp_dll.dll').replace("\\", "/")

        """ Read out labels and generate random colors """
        names_path = os.path.join(res_wd, 'cones.names').replace("\\", "/")
        self.LABELS = open(names_path).read().strip().split("\n")
        numpy.random.seed(42)
        self.COLORS = numpy.random.randint(0, 255, size=(len(self.LABELS), 3), dtype="uint8")

        # self.openCV_recognition = OpenCVRecognition(cfg_path, weight_path)
        self.darknet_recognition = DarknetRecognition(dll_path, cfg_path, weight_path)

        self.videoCapture = cv2.VideoCapture(0)
        # self.videoCapture = cv2.VideoCapture('D:/Documents/Personal_Data/Workspaces/HSK/Development/AdvancedConeRecognition/res/drive_KA_racing_30FPS.avi')

        if self.videoCapture.isOpened():
            self.img_w = int(self.videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.img_h = int(self.videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_fps = int(self.videoCapture.get(cv2.CAP_PROP_FPS))

            print("Video specs: width=" + str(self.img_w) + " height=" + str(self.img_h) + " FPS=" + str(self.video_fps))

            self.distance_calculation = SimpleDistanceCalculation(228.0, 325.0, -11111111111111111111111111111,
                                                                  self.img_w,
                                                                  self.img_h)  # TODO: Change to proper value, then calculate dist and angle

    def start(self, render=False, verbose=False, write_output=False):
        self.running = True

        if write_output:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            path = os.path.join(self.cwd, "output.avi").replace("\\", "/")
            self.out = cv2.VideoWriter(path, fourcc, self.video_fps, (self.img_w, self.img_h))

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

                    # draw a bounding box rectangle and label on the frame
                    color = [int(c) for c in self.COLORS[clazz]]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, (x, y), 10, color, -1)
                    text = "{}: {:.2f}".format(self.LABELS[clazz], probability)
                    cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if verbose:
                        print(x, y, w, h, self.LABELS[clazz])

            if write_output:
                self.out.write(frame)

            if render:
                cv2.imshow('Currently processed image', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        """ Free video capture """
        if write_output:
            self.out.release()

        self.videoCapture.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False


if __name__ == '__main__':
    advanced_cone_recognition = AdvancedConeRecognition()
    advanced_cone_recognition.start(render=True, verbose=False, write_output=True)
