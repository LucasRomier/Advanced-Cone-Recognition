"""

Author: Lucas Romier

"""

import cv2
import numpy
from OpenCVRecognition import OpenCVRecognition
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

        self.darknet_recognition = OpenCVRecognition(cfg_path, weight_path)
        self.videoCapture = cv2.VideoCapture(0)

    def start(self, render=False, verbose=False):
        self.running = True

        """ Start reading video """
        while self.videoCapture.isOpened() and self.running:
            (grabbed, frame) = self.videoCapture.read()
            if not grabbed:
                break

            (idxs, boxes, confidences, class_ids) = self.darknet_recognition.predict_boxes(frame)

            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    if verbose:
                        print(x, y, w, h, self.LABELS[class_ids[i]])

                    if render:
                        # draw a bounding box rectangle and label on the frame
                        color = [int(c) for c in self.COLORS[class_ids[i]]]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        text = "{}: {:.4f}".format(self.LABELS[class_ids[i]], confidences[i])
                        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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
    advanced_cone_recognition.start(render=True, verbose=False)
