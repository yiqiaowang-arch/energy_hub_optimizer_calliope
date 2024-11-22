import numpy as np
from numpy.typing import ArrayLike
from sympy import sequence


class ParetoFront:
    def __init__(self, xs: ArrayLike, ys: ArrayLike):
        if len(xs) != len(ys):
            raise ValueError("xs and ys must have the same length.")
        if len(xs.shape) != 1 or len(ys.shape) != 1:
            raise ValueError("xs and ys must be 1D arrays.")
        if not np.issubdtype(xs.dtype, np.number) or not np.issubdtype(
            ys.dtype, np.number
        ):
            raise ValueError("xs and ys must be numeric arrays.")

        sorted_sequence = np.argsort(xs)
        xs = xs[sorted_sequence]
        ys = ys[sorted_sequence]

        self.xy_values = np.column_stack((xs, ys))

        if np.any(np.diff(ys) > 0):  # y should be monotonically increasing
            raise ValueError(
                "the combination of x and y values does not represent a Pareto front, which should point to the top left and bottom right corners of the plot."
            )

    def x_values(
        self, ignore_left_endpoint=False, ignore_right_endpoint=False
    ) -> np.array:
        xs = self.xy_values[:, 0]
        if ignore_left_endpoint:
            xs = xs[1:]
        if ignore_right_endpoint:
            xs = xs[:-1]
        return xs

    def y_values(
        self, ignore_left_endpoint=False, ignore_right_endpoint=False
    ) -> np.array:
        ys = self.xy_values[:, 1]
        if ignore_left_endpoint:
            ys = ys[1:]
        if ignore_right_endpoint:
            ys = ys[:-1]
        return ys

    def x_range(
        self, ignore_left_endpoint=False, ignore_right_endpoint=False, rel=False
    ) -> float:
        range = np.ptp(self.x_values(ignore_left_endpoint, ignore_right_endpoint))
        if rel:
            max = np.max(self.x_values())
            return range / max
        return range

    def y_range(
        self, ignore_left_endpoint=False, ignore_right_endpoint=False, rel=False
    ) -> float:
        range = np.ptp(self.y_values(ignore_left_endpoint, ignore_right_endpoint))
        if rel:
            max = np.max(self.y_values())
            return range / max
        return range

    def slope(self, ignore_left_endpoint=False, ignore_right_endpoint=False) -> float:
        return self.y_range(ignore_left_endpoint, ignore_right_endpoint) / self.x_range(
            ignore_left_endpoint, ignore_right_endpoint
        )

    def curvature(
        self, ignore_left_endpoint=False, ignore_right_endpoint=False
    ) -> float:
        """
        Calculates the total length of the pareto front curve, deviding with the length of
        the straight line between the endpoints.

        :param ignore_endpoints: end points normally is optimized with one objective, thus is biased and
        unrealistic. Setting this toggle to true to ignore both endpoints, defaults to False
        :type ignore_endpoints: bool, optional
        :return: ratio between the length of the curve (polyline) and the length of the straight line between the endpoints.
        :rtype: float
        """
        return self.length(ignore_left_endpoint, ignore_right_endpoint) / np.sqrt(
            self.x_range(ignore_left_endpoint, ignore_right_endpoint) ** 2
            + self.y_range(ignore_left_endpoint, ignore_right_endpoint) ** 2
        )

    def length(self, ignore_left_endpoint=False, ignore_right_endpoint=False) -> float:
        """
        Calculate the length of the pareto front curve.

        :param ignore_endpoints: end points normally is optimized with one objective, thus is biased and unrealistic.
        Setting this toggle to true to ignore both endpoints, defaults to False
        :type ignore_endpoints: bool, optional
        :return: the length of the pareto front polyline
        :rtype: float
        """
        x_values = self.x_values(ignore_left_endpoint, ignore_right_endpoint)
        y_values = self.y_values(ignore_left_endpoint, ignore_right_endpoint)

        return np.sum(np.sqrt(np.diff(x_values) ** 2 + np.diff(y_values) ** 2))

    def get_collided_points(
        self, ignore_left_endpoint=False, ignore_right_endpoint=False
    ) -> np.array:
        """if some of the points are physically in the same place, report their values. if no collision, return empty array.

        :param ignore_endpoints: nd points normally is optimized with one objective, thus is biased and unrealistic.
        Setting this toggle to true to ignore both endpoints, defaults to False
        :type ignore_endpoints: bool, optional
        :return: an array of collided points, with each row being a point with its (x, y) coordinates.
            First column is x, second column is y. The points are sorted by their x values with an ascending order.
            If no collision, return empty array.
        :rtype: np.array
        """
        x_values = self.x_values(ignore_left_endpoint, ignore_right_endpoint)
        y_values = self.y_values(ignore_left_endpoint, ignore_right_endpoint)

        collided_points = []

        for i in range(1, len(x_values)):
            xi = x_values[i]
            yi = y_values[i]
            if xi == x_values[i - 1] and (xi, yi) not in collided_points:
                collided_points.append((xi, yi))

        return np.array(collided_points)
