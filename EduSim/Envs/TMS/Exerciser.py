# coding: utf-8
# 2020/5/9 @ tongshiwei

import random
from EduSim.Envs.meta.Aid import MetaExerciser
from EduSim.Envs.meta.SrcBank import EBank
from EduSim.utils import dina


class ExerciseBank(EBank):
    def __init__(self, skill_num, exercise_for_each_skill):
        self._exercise = {}
        for skill in range(skill_num):
            self._exercise[skill] = []
            for _ in range(exercise_for_each_skill):
                guessing = random.uniform(0.1, 0.3)
                skipping = random.uniform(0.1, 0.3)
                self._exercise[skill].append([guessing, skipping])

    def __getitem__(self, item):
        return self._exercise[item]


class Exerciser(MetaExerciser):
    def __init__(self, skill_num, exercise_for_each_skill):
        super(Exerciser, self).__init__(ExerciseBank(skill_num, exercise_for_each_skill))

    def feedback(self, response, exercise, *args, **kwargs):
        return dina(response, *exercise)

    def test(self, exercise_id, learner, *args, **kwargs):
        return [self.feedback([learner.state[exercise_id]], exercise) for exercise in self._bank[exercise_id]]

    def exam(self, learner, *exercise_id):
        return [self.test(eid, learner) for eid in exercise_id]

    @property
    def exercise_bank(self):
        return self._bank
