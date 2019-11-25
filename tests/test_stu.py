# coding: utf-8
# 2019/11/25 @ tongshiwei

from longling import as_list
import random
from EduSim.Envs.SimStu import SimStu
from EduSim.utils import irt


class IrtSimStu(SimStu):
    def __init__(self, concept_num: int, initializer: (str, None) = None):
        if initializer is None:
            self.state = [0] * concept_num
        elif initializer is "random":
            self.state = [random.random(0, 1) for _ in range(concept_num)]
        else:
            raise TypeError("unknown initializer: %s" % initializer)

        self.capacity = [random.random(0, 1) for _ in range(concept_num)]

    def learn(self, concept: (int, str, list)):
        for concept in as_list(concept):
            self.state[concept] = max(self.state[concept] + self.capacity[concept], 1)

    def test(self, concept, difficulty, discrimination=5, c=0.25):
        return irt(self.state[concept], difficulty, discrimination, c)


def test_sim_stu():
    pass
