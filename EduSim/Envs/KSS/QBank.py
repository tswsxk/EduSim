# coding: utf-8
# 2019/11/26 @ tongshiwei

import random


class QBank(object):
    def __init__(self, exercise_num, order):
        _difficulty = sorted([random.randint(0, 5) for _ in range(exercise_num)])
        difficulty = [0] * exercise_num
        for index, j in enumerate(order):
            difficulty[j] = _difficulty[index]
        self.difficulty = difficulty

    def __getitem__(self, item):
        return self.difficulty[item]
