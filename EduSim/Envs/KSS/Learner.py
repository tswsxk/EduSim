# coding: utf-8
# 2019/11/26 @ tongshiwei


import math

import networkx as nx
from EduSim.Envs.Learner import Learner as BaseLearner
from EduSim.Envs.Env import influence_control

__all__ = ["Learner"]


class Learner(BaseLearner):
    def __init__(self, initial_state, knowledge_structure: nx.DiGraph, learning_target: set,
                 learning_history=None, exercise_history=None, _id=None):

        super(Learner, self).__init__(_id=_id)

        self.structure = knowledge_structure
        self._state = initial_state
        self._target = learning_target
        self.learning_history = learning_history if learning_history else []
        self.exercise_history = exercise_history if exercise_history else []

    def learn(self, learning_item: int):
        structure = self.structure
        a = self._state

        if self.learning_history:
            if learning_item not in influence_control(
                    structure, a, self.learning_history[-1], allow_shortcut=False, target=self._target,
            )[0]:
                return

        assert isinstance(learning_item, int), learning_item
        self.learning_history.append(learning_item)

        # capacity growth function
        discount = math.exp(sum([(5 - a[node]) for node in structure.predecessors(learning_item)] + [0]))
        ratio = 1 / discount
        inc = (5 - a[learning_item]) * ratio * 0.5

        def _promote(_ind, _inc):
            a[_ind] += _inc
            if a[_ind] > 5:
                a[_ind] = 5
            for node in structure.successors(_ind):
                _promote(node, _inc * 0.5)

        _promote(learning_item, inc)

    def test(self, exercise):
        return self._state[exercise]

    @property
    def target(self):
        return self._target
