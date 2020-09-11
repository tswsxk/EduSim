# coding: utf-8
# 2020/4/29 @ tongshiwei

from .Aid import Aid
from EduSim.Envs.meta.SrcBank import EBank


class MetaExerciser(Aid):
    """to check how a learner/student perform on certain exercises"""

    def __init__(self, exercise_bank: EBank):
        self._bank = exercise_bank

    def present(self, exercise_id, *args, **kwargs):
        """present an exercise"""
        return self._bank[exercise_id]

    def feedback(self, response, *args, **kwargs):
        """receive a response from a learner and give a feedback"""
        raise NotImplementedError

    def test(self, exercise_id, *args, **kwargs) -> ...:
        """make a test on an exercise and report the result"""
        raise NotImplementedError

    def exam(self, learner, *exercise_id) -> ...:
        """make a test on several exercises and report the result"""
        raise NotImplementedError
