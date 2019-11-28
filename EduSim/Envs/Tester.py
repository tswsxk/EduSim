# coding: utf-8
# 2019/11/26 @ tongshiwei

from .Learner import Learner

from longling import config_logging


class Tester(object):
    """to check how a learner/student perform on certain exercises"""

    def __init__(self, logger=None):
        self._initial_score = None
        self._final_score = None
        self._learner = None
        self._logger = logger if logger else config_logging(logger="Tester")

    def begin_episode(self, learner: Learner):
        self._learner = learner
        self._initial_score = self.exam()

    def end_episode(self, path=None) -> dict:
        result = self.summary_episode(path)
        self._reset_episode()
        return result

    def _reset_episode(self):
        self._initial_score = None
        self._final_score = None
        self._learner = None

    def summary_episode(self, path):
        self._final_score = self.exam()

        result = {
            "learner_id": self._learner.id,
            "initial_score": self._initial_score,
            "final_score": self._final_score,
            "full_score": len(self._learner.target),
            "evaluation": self.eval(path) if path is not None else None,
        }

        self._logger.info(result)

        return result

    def test(self, exercise) -> ...:
        raise NotImplementedError

    def exam(self, exercises=None) -> ...:
        raise NotImplementedError

    def eval(self, path):
        initial_score = self._initial_score
        final_score = self._final_score
        full_score = len(self._learner.target)

        delta = final_score - initial_score
        delta_base = full_score - initial_score

        _alpha = delta / full_score
        _beta = _alpha / len(path) if len(path) > 0 else 0
        _gamma = delta / delta_base
        _theta = _gamma / len(path) if len(path) > 0 else 0

        return {
            "alpha": _alpha,
            "beta": _beta,
            "gamma": _gamma,
            "theta": _theta
        }
