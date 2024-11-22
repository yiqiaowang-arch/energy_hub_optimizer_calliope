from shapely.geometry import LineString, Polygon, LinearRing
import math


def calculate_elongation(geometry: Polygon) -> float:
    """calculate the elongation of a polygon. First, take the minimal bounding box of the polygon; then, calculate the ratio between the length and the width of the minimal bounding box.

    :param geometry: a polygon that describes the building's master plan (unit is meters).
    :type geometry: Polygon
    :return: elongation of the polygon (unitless)
    :rtype: float
    """
    min_rectangle: LinearRing = geometry.minimum_rotated_rectangle.exterior

    # Calculate the length and the width of the minimum rectangle
    a = LineString([min_rectangle.coords[0], min_rectangle.coords[1]]).length
    b = LineString([min_rectangle.coords[1], min_rectangle.coords[2]]).length
    # find which one is the length and which one is the width
    length = max(a, b)
    width = min(a, b)

    # Calculate the elongation
    elongation = length / width
    return elongation


def calculate_shape_factor(geometry: Polygon, height: float) -> float:
    """calculate the shape factor of a building, which is the ratio between surface area and volume.

    :param geometry: a polygon that describes the building's master plan (unit is meters).
    :type geometry: Polygon
    :param height: height of building above ground level in meters.
    :type height: float
    :return: the shape factor of the building (1/m)
    :rtype: float
    """
    # Calculate the area of the polygon
    area = geometry.area

    # Calculate the perimeter of the polygon
    perimeter = geometry.length

    # Calculate the shape factor, which is total exposed surface area divided by the total volume
    shape_factor = (area + height * perimeter) / (area * height)
    return shape_factor


def calculate_concavity(geometry: Polygon) -> float:
    """calculate the concavity of a polygon, which is the ratio between the area of the polygon and the area of its convex hull.

    :param geometry: a polygon that describes the building's master plan (unit is meters).
    :type geometry: Polygon
    :return: concavity of the polygon (unitless)
    :rtype: float
    """
    # Calculate the area of the polygon
    area = geometry.area

    # Calculate the area of the convex hull of the polygon
    convex_hull_area = geometry.convex_hull.area

    # Calculate the concavity, which is the ratio of area and convex hull area
    concavity = area / convex_hull_area
    return concavity


def compactness(geometry: Polygon, height: float) -> float:
    """calculate the compactness of a building, which is the ratio between the surface area and the minimum surface area that can contain the volume.
    The minimum surface area that can contain the volume is a cylinder with the same volume as the building.
    If we define the volume to be `V`, radius of the cylinder as `r` and the height of the cylinder as `h`, then the surface area of the
    cylinder is `S = 2*pi*r*h + pi*r^2` (excluding the bottom surface).
    Therefore, `h = V/(pi*r^2)` and `S = 2*pi*r*V/(pi*r^2) + pi*r^2 = 2*V/r + pi*r^2`.
    for minimum S, we need the derivative of S with respect to r to be 0,
    so `dS/dr = -2*V/r^2 + 2*pi*r = 0`, which gives `r = (V/pi)^(1/3)`.
    Then `h = V/(pi*r^2) = V/(pi*(V/pi)^(2/3)) = (V/pi)^(1/3)`.

    :param geometry: _description_
    :type geometry: Polygon
    :param height: _description_
    :type height: float
    :return: _description_
    :rtype: float
    """
    # Calculate the volume of the polygon
    volume = geometry.area * height
    surface_area = geometry.length * height + geometry.area

    # calculate the minimum surface area that can contain the volume
    r = (volume / math.pi) ** (1 / 3)
    h = volume / (math.pi * r**2)
    min_surface_area = 2 * math.pi * r * h + math.pi * r**2

    # calculate the compactness
    compactness = surface_area / min_surface_area
    return compactness


def calculate_building_direction(geometry: Polygon) -> float:
    """calculate the direction of the building, which is the angle between the longest side of the building and the north direction.

    :param geometry: a polygon that describes the building's master plan (unit is meters).
    :type geometry: Polygon
    :return: the direction of the building in degrees (0-180).
        If the building is pointing to northeast/southwest, the angle is between 0 and 90;
        if the building is pointing to southeast/northwest, the angle is between 90 and 180.
    :rtype: float
    """
    # get the minimal bounding box of the building
    min_rectangle: LinearRing = geometry.minimum_rotated_rectangle.exterior

    # calculate the angle between the longest side of the building and the north direction
    a = LineString([min_rectangle.coords[0], min_rectangle.coords[1]]).length
    b = LineString([min_rectangle.coords[1], min_rectangle.coords[2]]).length
    if a > b:
        # a is the length, we need to find the angle between a and the north direction.
        # the coordinates of points is (lat, lon), so if we use asin(delta_lat / a) we get the angle in radians.
        # Then we compare the coordinates of the two points: if the lat of first point is smaller but lon is larger or vice versa, then the line is pointing towards northeast/southwest.
        # Then the angle is between 90 and 180, so we need to add pi/2 to the angle.
        # finally, we convert the angle to degrees.
        lat1, lon1 = min_rectangle.coords[0]
        lat2, lon2 = min_rectangle.coords[1]
        angle = math.degrees(math.asin(abs(lat2 - lat1) / a))
        if (lat1 < lat2) ^ (lon1 < lon2):
            angle += 90
    else:
        # b is the length, we need to find the angle between b and the north direction.
        lat1, lon1 = min_rectangle.coords[1]
        lat2, lon2 = min_rectangle.coords[2]
        angle = math.degrees(math.asin(abs(lat2 - lat1) / b))
        if (lat1 <= lat2) ^ (lon1 <= lon2):
            angle = 180 - angle
    return angle
