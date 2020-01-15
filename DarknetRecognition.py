"""

Author: Lucas Romier
Based on: https://github.com/abhigarg/darknet-yolo-python/blob/master/darknet_demo.py

"""

from ctypes import *
import cv2
import numpy


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


class DarknetRecognition:

    def __init__(self, yolo_path, cfg_path, weight_path):
        self.lib = CDLL(yolo_path, RTLD_GLOBAL)

        self.lib.network_width.argtypes = [c_void_p]
        self.lib.network_width.restype = c_int
        self.lib.network_height.argtypes = [c_void_p]
        self.lib.network_height.restype = c_int

        self.copy_image_from_bytes = self.lib.copy_image_from_bytes
        self.copy_image_from_bytes.argtypes = [IMAGE, c_char_p]

        self.predict = self.lib.network_predict_ptr
        self.predict.argtypes = [c_void_p, POINTER(c_float)]
        self.predict.restype = POINTER(c_float)

        self.set_gpu = self.lib.cuda_set_device
        self.set_gpu.argtypes = [c_int]

        self.init_cpu = self.lib.init_cpu

        self.make_image = self.lib.make_image
        self.make_image.argtypes = [c_int, c_int, c_int]
        self.make_image.restype = IMAGE

        self.get_network_boxes = self.lib.get_network_boxes
        self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int,
                                           POINTER(c_int), c_int]
        self.get_network_boxes.restype = POINTER(DETECTION)

        self.make_network_boxes = self.lib.make_network_boxes
        self.make_network_boxes.argtypes = [c_void_p]
        self.make_network_boxes.restype = POINTER(DETECTION)

        self.free_detections = self.lib.free_detections
        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

        self.free_ptrs = self.lib.free_ptrs
        self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        self.network_predict = self.lib.network_predict_ptr
        self.network_predict.argtypes = [c_void_p, POINTER(c_float)]

        self.reset_rnn = self.lib.reset_rnn
        self.reset_rnn.argtypes = [c_void_p]

        self.load_net = self.lib.load_network
        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype = c_void_p

        self.load_net_custom = self.lib.load_network_custom
        self.load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
        self.load_net_custom.restype = c_void_p

        self.do_nms_obj = self.lib.do_nms_obj
        self.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.do_nms_sort = self.lib.do_nms_sort
        self.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.free_image = self.lib.free_image
        self.free_image.argtypes = [IMAGE]

        self.letterbox_image = self.lib.letterbox_image
        self.letterbox_image.argtypes = [IMAGE, c_int, c_int]
        self.letterbox_image.restype = IMAGE

        self.load_meta = self.lib.get_metadata
        self.lib.get_metadata.argtypes = [c_char_p]
        self.lib.get_metadata.restype = METADATA

        self.load_image = self.lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        self.rgbgr_image = self.lib.rgbgr_image
        self.rgbgr_image.argtypes = [IMAGE]

        self.predict_image = self.lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)

        self.predict_image_letterbox = self.lib.network_predict_image_letterbox
        self.predict_image_letterbox.argtypes = [c_void_p, IMAGE]
        self.predict_image_letterbox.restype = POINTER(c_float)

        """ Init main darknet components """
        self.net_main = self.load_net_custom(cfg_path.encode("ascii"), weight_path.encode("ascii"), 0,
                                             1)  # batch size = 1

    def network_width(self, net):
        return self.lib.network_width(net)

    def network_height(self, net):
        return self.lib.network_height(net)

    @staticmethod
    def __array_to_image(array):
        # need to return old values to avoid python freeing memory
        array = array.transpose(2, 0, 1)
        c = array.shape[0]
        h = array.shape[1]
        w = array.shape[2]
        array = numpy.ascontiguousarray(array.flat, dtype=numpy.float32) / 255.0
        data = array.ctypes.data_as(POINTER(c_float))
        image = IMAGE(w, h, c, data)
        return image, array

    @staticmethod
    def to_usable(x, y, w, h):
        w = int(w)
        h = int(h)
        x_coord = int(x - w / 2)
        y_coord = int(y - h / 2)

        return x_coord, y_coord, x_coord + w, y_coord + h

    """ threshold is value for valid detection. hier_threshold determines how specific the result is. nms takes care of overlapping boxes """
    def predict_boxes(self, frame, classes_count, threshold=0.45, hier_threshold=0.7, nms=0.45):
        """ Extract data from frame and convert to array """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = cv2.resize(rgb_frame,
                               (self.lib.network_width(self.net_main), self.lib.network_height(self.net_main)),
                               interpolation=cv2.INTER_LINEAR)
        image, array = self.__array_to_image(rgb_frame)

        """ Create pointer for later fetching """
        prediction_count = c_int(0)
        pointer_prediction_count = pointer(prediction_count)

        """ Prepare detection and fetch """
        self.predict_image(self.net_main, image)
        letter_box = 0
        # predict_image_letterbox(net, im)
        # letter_box = 1
        detections = self.get_network_boxes(self.net_main, rgb_frame.shape[1], rgb_frame.shape[0], threshold,
                                            hier_threshold, None, 0, pointer_prediction_count, letter_box)

        """ Get amount of predictions """
        prediction_count = pointer_prediction_count[0]
        if nms:
            self.do_nms_sort(detections, prediction_count, classes_count, nms)

        """ Check predictions """
        ret = []
        for prediction in range(prediction_count):
            for clazz in range(classes_count):
                if detections[prediction].prob[clazz] > 0:
                    b = detections[prediction].bbox
                    ret.append((clazz, detections[prediction].prob[clazz], (b.x, b.y, b.w, b.h)))

        """ Sort and free detection """
        ret = sorted(ret, key=lambda x: -x[1])
        self.free_detections(detections, prediction_count)

        return ret
