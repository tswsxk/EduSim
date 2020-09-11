# coding: utf-8
# 2020/5/8 @ tongshiwei

from EduSim.utils import irt
from EduSim.Envs.meta.Aid import MetaExerciser as _Exerciser


class Exerciser(_Exerciser):
    def __init__(self, exercise_bank, binary_feedback=True, threshold=0.5):
        super(Exerciser, self).__init__(exercise_bank)
        self._current_exercise_id = None
        self._binary_mode = binary_feedback
        self._threshold = threshold

    def present(self, exercise_id, *args, **kwargs):
        self._current_exercise_id = exercise_id
        return exercise_id

    def feedback(self, response, current_exercise_id, binary_mode=None, *args, **kwargs):
        binary_mode = binary_mode if binary_mode is not None else self._binary_mode
        p = irt(response, self._bank[current_exercise_id])
        if binary_mode:
            return 1 if p >= self._threshold else 0
        else:
            return p

    def test(self, exercise_id, learner, binary_mode=None):
        if isinstance(learner, (int, float)):
            return [exercise_id, self.feedback(learner, exercise_id, binary_mode)]
        elif isinstance(learner, dict):
            return [exercise_id, self.feedback(learner[exercise_id], exercise_id, binary_mode)]
        else:
            return [exercise_id, self.feedback(learner.test(exercise_id), exercise_id, binary_mode)]

    def exam(self, learner, *exercise_id, binary_mode=None):
        return [self.test(_exercise_id, learner, binary_mode) for _exercise_id in exercise_id]
