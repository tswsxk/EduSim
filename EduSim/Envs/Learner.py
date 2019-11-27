# coding: utf-8
# 2019/11/25 @ tongshiwei
import uuid

__all__ = ["Learner"]


class Learner(object):
    def __init__(self, _id=None):
        self.id = _id if _id is not None else uuid.uuid1()

        self.learning_history = []
        self.exercise_history = []

    def learn(self, learning_item, *args, **kwargs):
        """learn a new learning item, which can result in state changing"""
        raise NotImplementedError

    def test(self, exercise, *args, **kwargs) -> ...:
        """test on a certain exercise, return the answer"""
        raise NotImplementedError
