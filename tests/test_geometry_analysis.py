import unittest
from shapely.geometry import Polygon
from shapely.affinity import rotate
from math import pi
import numpy as np

from trustParetoFronts.geometry_analysis import (
    calculate_elongation,
    calculate_shape_factor,
    calculate_concavity,
    compactness,
    calculate_building_direction,
)


class TestGeometryAnalysis(unittest.TestCase):

    def setUp(self):
        self.rectangle = Polygon([(0, 0), (4, 0), (4, 2), (0, 2), (0, 0)])
        self.square = Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
        self.L_shape = Polygon([(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3), (0, 0)])
        self.concave_shape = Polygon([(0, 0), (4, 0), (2, 2), (4, 4), (0, 4), (0, 0)])
        self.rotated_rectangle = Polygon([(0, 0), (2, 2), (5, -1), (3, -3), (0, 0)])
        self.height = 10.0

    def test_calculate_elongation_rectangle(self):
        elongation = calculate_elongation(self.rectangle)
        self.assertAlmostEqual(elongation, 2.0)

    def test_calculate_elongation_square(self):
        elongation = calculate_elongation(self.square)
        self.assertAlmostEqual(elongation, 1.0)

    def test_calculate_elongation_L_shape(self):
        elongation = calculate_elongation(self.L_shape)
        self.assertAlmostEqual(elongation, 1.5)

    def test_calculate_shape_factor_rectangle(self):
        shape_factor = calculate_shape_factor(self.rectangle, self.height)
        expected_value = (self.rectangle.area + self.height * self.rectangle.length) / (
            self.rectangle.area * self.height
        )
        self.assertAlmostEqual(shape_factor, expected_value)

    def test_calculate_shape_factor_square(self):
        shape_factor = calculate_shape_factor(self.square, self.height)
        expected_value = (self.square.area + self.height * self.square.length) / (
            self.square.area * self.height
        )
        self.assertAlmostEqual(shape_factor, expected_value)

    def test_calculate_concavity_convex(self):
        concavity = calculate_concavity(self.rectangle)
        self.assertAlmostEqual(concavity, 1.0)

    def test_calculate_concavity_L_shape(self):
        concavity = calculate_concavity(self.L_shape)
        expected_value = self.L_shape.area / self.L_shape.convex_hull.area
        self.assertAlmostEqual(concavity, expected_value)

    def test_calculate_concavity_concave(self):
        concavity = calculate_concavity(self.concave_shape)
        expected_value = self.concave_shape.area / self.concave_shape.convex_hull.area
        self.assertAlmostEqual(concavity, expected_value)

    def test_compactness_square(self):
        compactness_value = compactness(self.square, self.height)
        volume = self.square.area * self.height
        surface_area = self.square.length * self.height + self.square.area
        r = (volume / pi) ** (1 / 3)
        h = (volume / pi) ** (1 / 3)
        min_surface_area = 2 * pi * r * h + pi * r**2
        expected_value = surface_area / min_surface_area
        self.assertAlmostEqual(compactness_value, expected_value)

    def test_calculate_building_direction_rectangle(self):
        direction = calculate_building_direction(self.rectangle)
        self.assertAlmostEqual(direction, 90.0)

    def test_calculate_building_direction_rotated(self):
        direction = calculate_building_direction(self.rotated_rectangle)
        self.assertAlmostEqual(direction, 135.0, places=1)

    def test_calculate_building_direction_rotated_30(self):
        # Rotate a rectangle by 135 degrees
        rectangle = Polygon([(0, 0), (4, 0), (4, 2), (0, 2)])
        rotated_rectangle = rotate(rectangle, 30, origin=(0, 0), use_radians=False)
        direction = calculate_building_direction(rotated_rectangle)
        self.assertAlmostEqual(direction, 60, places=1)

    def test_calculate_building_direction_rotated_90(self):
        # Rotate a rectangle by 135 degrees
        rectangle = Polygon([(0, 0), (4, 0), (4, 2), (0, 2)])
        rotated_rectangle = rotate(rectangle, 90, origin=(0, 0), use_radians=False)
        direction = calculate_building_direction(rotated_rectangle)
        self.assertAlmostEqual(direction, 0, places=1)

    def test_calculate_building_direction_T_shape_rotated_random(self):
        # Rotate a T-shape by 30 degrees
        H_shape = Polygon(
            [
                (0, 0),
                (3, 0),
                (3, 1),
                (2, 1),
                (2, 5),
                (3, 5),
                (3, 6),
                (0, 6),
                (0, 5),
                (1, 5),
                (1, 1),
                (0, 1),
                (0, 0),
            ]
        )
        for i in range(360):
            angle = i
            rotated_H_shape = rotate(H_shape, -angle, origin=(0, 0), use_radians=False)
            direction = calculate_building_direction(rotated_H_shape)
            angle_predicted = angle % 180
            self.assertAlmostEqual(direction, angle_predicted, places=1)


if __name__ == "__main__":
    unittest.main()
