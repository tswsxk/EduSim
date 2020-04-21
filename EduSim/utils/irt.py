# coding: utf-8
# 2019/11/25 @ tongshiwei

__all__ = ["irt"]

import math


def irt(ability, difficulty, discrimination=5, c=0.25):
    """
    Examples
    --------
    >>> round(irt(3, 1), 2)
    1.0
    >>> round(irt(1, 5), 3)
    0.25
    """
    return c + (1 - c) / (1 + math.exp(-1.7 * discrimination * (ability - difficulty)))
