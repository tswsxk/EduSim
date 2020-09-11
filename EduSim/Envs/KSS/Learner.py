# coding: utf-8
# 2019/11/26 @ tongshiwei

import random
import math

import networkx as nx
from EduSim.Envs.meta import MetaLearner, MetaLearnerGroup, MetaLearningModel
from EduSim.Envs.shared.KSS_KES.KS import influence_control

__all__ = ["Learner"]


class LearningModel(MetaLearningModel):
    def __init__(self, state, learning_target, knowledge_structure, last_visit=None):
        self._state = state
        self._target = learning_target
        self._ks = knowledge_structure
        self._ks_last_visit = last_visit

    def step(self, state, learning_item):
        if self._ks_last_visit is not None:
            if learning_item not in influence_control(
                    self._ks, state, self._ks_last_visit, allow_shortcut=False, target=self._target,
            )[0]:
                return
        self._ks_last_visit = learning_item

        # capacity growth function
        discount = math.exp(sum([(5 - state[node]) for node in self._ks.predecessors(learning_item)] + [0]))
        ratio = 1 / discount
        inc = (5 - state[learning_item]) * ratio * 0.5

        def _promote(_ind, _inc):
            state[_ind] += _inc
            if state[_ind] > 5:
                state[_ind] = 5
            for node in self._ks.successors(_ind):
                _promote(node, _inc * 0.5)

        _promote(learning_item, inc)


class Learner(MetaLearner):
    def __init__(self, initial_state, knowledge_structure: nx.DiGraph, learning_target: set,
                 learning_history: list = None, profile=None, _id=None):
        super(Learner, self).__init__(_id=_id)

        self.lm = LearningModel(
            initial_state,
            learning_target,
            knowledge_structure,
            learning_history[-1] if learning_history is not None else None
        )

        self.structure = knowledge_structure
        self._state = initial_state
        self._target = learning_target
        self._profile = [] if profile is None else profile

    @property
    def profile(self):
        return self._profile

    def set_profile(self, profile):
        self._profile = profile

    def learn(self, learning_item: int):
        self.lm.step(self._state, learning_item)

    @property
    def state(self):
        return {_idx: self._state[_idx] for _idx in self._target}

    def test(self, exercise):
        return self._state[exercise]

    @property
    def target(self):
        return self._target


class LearnerGroup(MetaLearnerGroup):
    def __init__(self, knowledge_structure):
        super(LearnerGroup, self).__init__([])
        self.ks = knowledge_structure

    def new_learner(self):
        return Learner(
            [random.randint(-3, 0) - (0.1 * i) for i in range(10)],
            self.ks,
            set(random.sample(self.ks.nodes, random.randint(3, len(self.ks.nodes)))),
        )

    def add(self, learner):
        self._learners.append(learner)

    def __getitem__(self, item):
        return self._learners[item]

    def __iter__(self):
        for learner in self._learners:
            yield learner
