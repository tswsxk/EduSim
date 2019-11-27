# coding: utf-8
# 2019/11/26 @ tongshiwei

from EduSim.utils import irt
from EduSim.Envs.Tester import Tester as BaseTester

from .QBank import QBank


class Tester(BaseTester):
    def __init__(self, q_bank: QBank):
        super(Tester, self).__init__()

        self.q_bank = q_bank

    def test(self, exercise: int, binary=True, learner=None):
        _learner = learner if learner is not None else self._learner
        p = irt(self._learner.test(exercise), self.q_bank.get_difficulty(exercise))
        if binary:
            return 1 if p >= 0.5 else 0
        else:
            return p

    def exam(self, exercises=None, binary=True, learner=None):
        exercises = self._learner.target if exercises is None else exercises

        return sum([self.test(exercise, binary=binary) for exercise in exercises])
