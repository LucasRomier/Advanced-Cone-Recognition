import math


class SimpleDistanceCalculation:

    def __init__(self, known_cone_width, known_cone_height, focal_length, image_width, image_height):
        self.known_cone_width = known_cone_width
        self.known_cone_height = known_cone_height
        self.focal_length = focal_length
        self.image_width = image_width
        self.image_width_half = image_width / 2.
        self.image_height = image_height

    def calculate_focal_length(self, known_distance, cone_w, cone_h):
        focal_width = (cone_w * known_distance) / self.known_cone_width
        focal_height = (cone_h * known_distance) / self.known_cone_height
        return (focal_width + focal_height) / 2

    def calculate_distance(self, cone_w, cone_h):
        distance_width = (self.known_cone_width * self.focal_length) / cone_w
        distance_height = (self.known_cone_height * self.focal_length) / cone_h
        return (distance_width + distance_height) / 2

    def calculate_relative_angle(self, cone_x, cone_y):
        if cone_x >= self.image_width_half:
            opposite = cone_x - self.image_width / 2.
        else:
            opposite = self.image_width / 2. - cone_x
        adjacent = self.image_height - cone_y
        return math.tan(opposite / adjacent)
