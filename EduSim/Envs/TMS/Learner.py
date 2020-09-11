# coding: utf-8
# 2020/4/29 @ tongshiwei

import random
from longling.ML.utils import choice

from EduSim.Envs.meta.Learner import MetaLearner, MetaLearningModel


class TransitionMatrix(MetaLearningModel):
    def __init__(self, transition_matrix, state2list):
        self.matrix = transition_matrix
        self._state2list = state2list

    def step(self, state, learning_item_id):
        return choice(self.matrix[learning_item_id][state])

    def state2list(self, state):
        return self._state2list[state]


class Learner(MetaLearner):
    def __init__(self, transition_matrix, state2list, init_states: list, _id=None):
        super(Learner, self).__init__(_id)
        self.learning_model = TransitionMatrix(transition_matrix, state2list)
        self._state = None
        self._init_states = init_states
        self.reset()

    def reset(self):
        self._state = random.choice(self._init_states)

    @property
    def state(self):
        return self.learning_model.state2list(self._state)

    def learn(self, learning_item, *args, **kwargs):
        self._state = self.learning_model.step(self._state, learning_item)

    def test(self, exercise, *args, **kwargs):
        return self.state[exercise]
