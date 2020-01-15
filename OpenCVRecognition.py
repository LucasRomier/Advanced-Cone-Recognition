"""

Author: Lucas Romier
Based on: https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

"""

import numpy
import cv2


class OpenCVRecognition:

    def __init__(self, cfg_path, weight_path):
        """ Load YOLO in OpenCV """
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weight_path)
        self.layer_names = self.net.getLayerNames()
        self.layer_names = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        self.W = self.H = None

    def predict_boxes(self, frame, threshold=0.8, hier_threshold=0.5):
        """ Grb dimensions if not set """
        if self.W is None or self.H is None:
            (self.W, self.H) = frame.shape[:2]

        """ Create blob from image for processing """
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.layer_names)

        """ Empty holding variables """
        boxes = []
        confidences = []
        class_ids = []

        """ Iterate over each output layer """
        for output in layer_outputs:

            """ Check every detection"""
            for detection in output:

                """ Extract class and probability """
                scores = detection[5:]
                class_id = numpy.argmax(scores)
                confidence = scores[class_id]

                """ Only consider recognitions above certain threshold """
                if confidence > threshold:
                    """ Scale bounding box back to readable format """
                    box = detection[0:4] * numpy.array([self.W, self.H, self.W, self.H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        """ Non-maxima-suppression to overlapping boxes """
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, hier_threshold)

        return idxs, boxes, confidences, class_ids
