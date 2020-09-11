# coding: utf-8
# 2020/4/29 @ tongshiwei

import random
import uuid

__all__ = ["MetaLearner", "MetaLearnerGroup", "MetaLearningModel"]


class MetaLearningModel(object):
    def step(self, state, learning_item, *args, **kwargs):
        raise NotImplementedError


class MetaLearner(object):
    def __init__(self, _id=None):
        self.id = self.__id(_id)

    @classmethod
    def __id(cls, _id=None):
        return _id if _id is not None else uuid.uuid1()

    @property
    def profile(self):
        return {"id": self.id}

    @property
    def state(self):
        raise NotImplementedError

    def learn(self, learning_item, *args, **kwargs):
        """learn a new learning item, which can result in state changing"""
        raise NotImplementedError

    def test(self, exercise, *args, **kwargs) -> ...:
        """test on a certain exercise, return the answer"""
        raise NotImplementedError


class MetaLearnerGroup(object):
    def __init__(self, learners, *args, **kwargs):
        self._learners = learners

    def __getitem__(self, item):
        return self._learners[item]

    def sample(self):
        return random.choice(self._learners)

    def __len__(self):
        return len(self._learners)
