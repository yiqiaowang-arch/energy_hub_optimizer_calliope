from typing import Union, Tuple


def calculate_slope_interception(
    func, min_value: Union[float, int], max_value: Union[float, int]
) -> Tuple[float, float]:
    ymin = func(min_value)
    ymax = func(max_value)
    slope = (ymax - ymin) / (max_value - min_value)
    interception = ymin - slope * min_value
    return slope, interception


def power(
    a: Union[float, int],
    b: Union[float, int],
    min_value: Union[float, int],
    max_value: Union[float, int],
) -> Tuple[float, float]:
    power_func = lambda x: a * x**b
    return calculate_slope_interception(power_func, min_value, max_value)


def quadratic(
    a: Union[float, int],
    b: Union[float, int],
    c: Union[float, int],
    min_value: Union[float, int],
    max_value: Union[float, int],
) -> Tuple[float, float]:
    quadratic_func = lambda x: a * x**2 + b * x + c
    return calculate_slope_interception(quadratic_func, min_value, max_value)
