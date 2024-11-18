import unittest
import numpy as np
from trustParetoFronts.pareto_analysis import ParetoFront


class TestParetoFront(unittest.TestCase):
    def test_initialization_valid(self):
        xs = np.array([1, 2, 3, 4, 5])
        ys = np.array([6, 5, 4, 3, 2])
        pf = ParetoFront(xs, ys)
        np.testing.assert_array_equal(pf.xy_values, np.column_stack((xs, ys)))

    def test_randomized_initialization(self):
        xs = np.array([3, 4, 5, 1, 2])
        ys = np.array([4, 3, 2, 6, 5])
        pf = ParetoFront(xs, ys)
        sorted = np.column_stack((np.array([1, 2, 3, 4, 5]), np.array([6, 5, 4, 3, 2])))
        np.testing.assert_array_equal(pf.xy_values, sorted)

    def test_initialization_invalid_length(self):
        xs = np.array([1, 2, 3])
        ys = np.array([3, 2])
        with self.assertRaises(ValueError):
            ParetoFront(xs, ys)

    def test_initialization_invalid_shape(self):
        xs = np.array([[1, 2], [3, 4]])
        ys = np.array([5, 4, 3, 2])
        with self.assertRaises(ValueError):
            ParetoFront(xs, ys)

    def test_initialization_non_numeric(self):
        xs = np.array([1, 2, 3, 4, 5])
        ys = np.array(["a", "b", "c", "d", "e"])
        with self.assertRaises(ValueError):
            ParetoFront(xs, ys)

    def test_x_values(self):
        xs = np.array([1, 2, 3, 4, 5])
        ys = np.array([6, 5, 4, 3, 2])
        pf = ParetoFront(xs, ys)
        np.testing.assert_array_equal(pf.x_values(), xs)

    def test_y_values(self):
        xs = np.array([1, 2, 3, 4, 5])
        ys = np.array([6, 5, 4, 3, 2])
        pf = ParetoFront(xs, ys)
        np.testing.assert_array_equal(pf.y_values(), ys)

    def test_x_range(self):
        xs = np.array([1, 2, 3, 4, 5])
        ys = np.array([6, 5, 4, 3, 2])
        pf = ParetoFront(xs, ys)
        self.assertEqual(pf.x_range(), 4.0)

    def test_x_relative_range(self):
        xs = np.array([1, 2, 3, 4, 5])
        ys = np.array([6, 5, 4, 3, 2])
        pf = ParetoFront(xs, ys)
        self.assertEqual(pf.x_range(rel=True), 0.8)

    def test_y_range(self):
        xs = np.array([1, 2, 3, 4, 5])
        ys = np.array([6, 5, 4, 3, 2])
        pf = ParetoFront(xs, ys)
        self.assertEqual(pf.y_range(), 4.0)

    def test_y_relative_range(self):
        xs = np.array([1, 2, 3, 4, 5])
        ys = np.array([6, 5, 4, 3, 2])
        pf = ParetoFront(xs, ys)
        self.assertAlmostEqual(pf.y_range(rel=True), 4 / 6, places=5)

    def test_slope(self):
        xs = np.array([1, 2, 3, 4, 5])
        ys = np.array([6, 5, 4, 3, 2])
        pf = ParetoFront(xs, ys)
        self.assertEqual(pf.slope(), 1.0)

    def test_curvature(self):
        xs = np.array([1, 2, 3, 4, 5])
        ys = np.array([6, 5, 4, 3, 2])
        pf = ParetoFront(xs, ys)
        self.assertAlmostEqual(pf.curvature(), 1.0, places=5)

    def test_length(self):
        xs = np.array([1, 2, 3, 4, 5])
        ys = np.array([6, 5, 4, 3, 2])
        pf = ParetoFront(xs, ys)
        self.assertAlmostEqual(pf.length(), 5.65685, places=5)

    def test_get_collided_points(self):
        xs = np.array([1, 2, 2, 4, 5])
        ys = np.array([6, 4, 4, 3, 2])
        pf = ParetoFront(xs, ys)
        np.testing.assert_array_equal(pf.get_collided_points(), np.array([[2, 4]]))

    def test_get_collided_points_multiple_collisions(self):
        xs = np.array([1, 2, 2, 4, 4, 5])
        ys = np.array([6, 5, 5, 3, 3, 2])
        pf = ParetoFront(xs, ys)
        np.testing.assert_array_equal(
            pf.get_collided_points(), np.array([[2, 5], [4, 3]])
        )


if __name__ == "__main__":

    unittest.main()
