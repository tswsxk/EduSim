# coding: utf-8
# 2019/11/27 @ tongshiwei
import pytest
from longling import as_list
import random
from EduSim.Envs.Learner import Learner
from copy import deepcopy


class IrtLearner(Learner):
    def __init__(self, concept_num: int, initializer: (str, None) = None):
        super(IrtLearner, self).__init__()

        if initializer is None:
            self.state = [0] * concept_num
        elif initializer is "random":
            self.state = [random.random() for _ in range(concept_num)]
        else:
            raise TypeError("unknown initializer: %s" % initializer)

        self.capacity = [random.random() for _ in range(concept_num)]

    def learn(self, learning_item: (int, str, list)):
        for concept in as_list(learning_item):
            self.state[concept] = max(self.state[concept] + self.capacity[concept], 1)

    def test(self, exercise):
        return self.state[exercise]

    def state_snapshot(self):
        return deepcopy(self.state)


@pytest.fixture(scope="module")
def learner():
    return IrtLearner(10)


@pytest.fixture
def agents(tmp_path) -> dict:
    from EduSim.Agent.utils import Graph
    from EduSim.Agent.agent import RandomGraphAgent

    return {
        "random_agent": RandomGraphAgent(Graph("KSS"))
    }
