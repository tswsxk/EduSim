# coding: utf-8
# 2020/5/12 @ tongshiwei

import functools
from typing import overload
import numpy as np


def as_array(obj, skip_type=(int, float)):
    if isinstance(obj, skip_type):
        return obj

    if isinstance(obj, list):
        return np.asarray(obj)

    return obj


@overload
def exponential_forgetting_curve(item_difficulty: (list, np.ndarray), elapsed_time: (list, np.ndarray),
                                 memory_strength: (list, np.ndarray)) -> np.ndarray:
    ...


@overload
def exponential_forgetting_curve(item_difficulty: float, elapsed_time: float,
                                 memory_strength: float) -> float:
    ...


def exponential_forgetting_curve(item_difficulty: (float, list, np.ndarray), elapsed_time: (float, list, np.ndarray),
                                 memory_strength: (float, list, np.ndarray)) -> (float, np.ndarray):
    """

    Parameters
    ----------
    item_difficulty: float or list or array
    elapsed_time: float or list or array
        the time elapsed since the item was last reviewed by the student
    memory_strength: float or list or array
        the student’s memory strength for the item

    Returns
    -------
    z: float or array
        the probability of recalling an item

    Examples
    --------
    >>> exponential_forgetting_curve(1, 0, 1)
    1.0
    >>> exponential_forgetting_curve(2, 0, 1)
    1.0
    >>> round(exponential_forgetting_curve(1, 1, 1), 2)
    0.37
    >>> round(exponential_forgetting_curve(2, 1, 1), 2)
    0.14
    >>> round(exponential_forgetting_curve(2, 10, 1), 2)
    0.0
    >>> round(exponential_forgetting_curve(2, 10, 10), 2)
    0.14
    >>> [round(v, 2) for v in exponential_forgetting_curve([1, 2], [10, 1], [10, 1])]
    [0.37, 0.14]
    """
    theta = as_array(item_difficulty)
    d = as_array(elapsed_time)
    s = as_array(memory_strength)

    return np.exp(- theta * (d / s))


efc = exponential_forgetting_curve


@overload
def half_life_regression(elapsed_time: float, feature_coefficients: (list, np.ndarray),
                         features: (list, np.ndarray)) -> float:
    pass


@overload
def half_life_regression(elapsed_time: list, feature_coefficients: (list, np.ndarray),
                         features: (list, np.ndarray)) -> np.ndarray:
    pass


def half_life_regression(elapsed_time, feature_coefficients, features):
    """

    Parameters
    ----------
    elapsed_time
        the time elapsed since the item was last reviewed by the student
    feature_coefficients

    features


    Returns
    -------
    z:
        the probability of recalling an item


    Examples
    --------
    >>> feature_coefficients = [1, 1, 1, 1]
    >>> round(half_life_regression(1, feature_coefficients, [1, 2, 1, 2]), 3)
    0.998
    >>> [round(v, 2) for v in half_life_regression([0, 2], feature_coefficients, [[0, 0, 0, 0], [1, 1, 1, 0]])]
    [1.0, 0.91]
    """
    feature_coefficients = as_array(feature_coefficients)
    features = as_array(features).T
    s = np.exp(feature_coefficients @ features)
    d = as_array(elapsed_time)

    return np.exp(- d / s)


hlr = half_life_regression


@overload
def generalized_power_law(
        student_ability: float,
        additional_student_ability: float,
        item_difficulty: float,
        additional_item_difficulty: float,
        elapsed_time: float,
        decay_rate: float,
        correct_times: (list, np.ndarray) = None,
        attempts: (list, np.ndarray) = None,
        window_correct_coefficients: (list, np.ndarray) = None,
        window_attempt_coefficients: (list, np.ndarray) = None,
) -> float:
    pass


@overload
def generalized_power_law(
        student_ability: (list, np.ndarray),
        additional_student_ability: (list, np.ndarray),
        item_difficulty: (list, np.ndarray),
        additional_item_difficulty: (list, np.ndarray),
        elapsed_time: (list, np.ndarray),
        decay_rate: float,
        correct_times: (list, np.ndarray) = None,
        attempts: (list, np.ndarray) = None,
        window_correct_coefficients: (list, np.ndarray) = None,
        window_attempt_coefficients: (list, np.ndarray) = None,
) -> np.ndarray:
    pass


def generalized_power_law(
        student_ability,
        additional_student_ability,
        item_difficulty,
        additional_item_difficulty,
        elapsed_time,
        decay_rate: float,
        correct_times=None,
        attempts=None,
        window_correct_coefficients=None,
        window_attempt_coefficients=None,
):
    """

    Parameters
    ----------
    student_ability: float, list, array
    additional_student_ability: float, list, array
    item_difficulty: float, list, array
    additional_item_difficulty: float, list, array
    elapsed_time: float, list, array
        the time elapsed since the item was last reviewed by the student
    decay_rate: float
         a constant that controls the decay rate
    correct_times: list, array
         the number of times the student correctly recalled the item in window w
    attempts: list, array
        the number of times the student attempt to recall the item in window w
    window_correct_coefficients: list, array
        window specific weights for correct features
    window_attempt_coefficients: list, array
        window specific weights for attempt features

    Returns
    -------
    z
        the probability of recalling an item


    """
    r = decay_rate
    a = as_array(student_ability)
    d = as_array(item_difficulty)
    t = as_array(elapsed_time)

    if window_correct_coefficients is None:
        m = 1 / (1 + np.exp(a - d))
    else:
        theta_cw = np.asarray(window_correct_coefficients)
        theta_nw = np.asarray(window_attempt_coefficients)
        cw = np.asarray(correct_times).T
        nw = np.asarray(attempts).T
        h = theta_cw @ cw + theta_nw @ nw
        m = 1 / (1 + np.exp(a - d + h))

    f = np.exp(additional_student_ability - additional_item_difficulty)

    return m * (1 + r * t) ** (-f)


gpl = generalized_power_law
dash = functools.partial(generalized_power_law, correct_times=None, attempts=None,
                         window_correct_coefficients=None, window_attempt_coefficients=None)
