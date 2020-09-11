# coding: utf-8
# 2020/5/13 @ tongshiwei


from collections import namedtuple
from .utils import efc, hlr, gpl, dash
from EduSim.Envs.meta import MetaLearner, MetaLearningModel

EFCState = namedtuple("EFCState", ["memory_strengths", "latest_review_ts"])


class EFCLearningModel(MetaLearningModel):
    def step(self, state: EFCState, learning_item, timestamp, *args, **kwargs):
        state.memory_strengths[learning_item] += 1
        state.latest_review_ts[learning_item] = timestamp


class EFCLearner(MetaLearner):
    def __init__(self, memory_strengths: list, latest_review_ts: list):
        super(EFCLearner, self).__init__()
        self._state = EFCState(memory_strengths, latest_review_ts)
        self._learning_model = EFCLearningModel()

    @property
    def state(self):
        return self._state

    def learn(self, learning_item, timestamp, *args, **kwargs):
        self._learning_model.step(self._state, learning_item, timestamp)

    def test(self, exercise_id, exercise_difficulty, timestamp, *args, **kwargs):
        return efc(
            exercise_difficulty,
            timestamp - self._state.latest_review_ts[exercise_id],
            self._state.memory_strengths[exercise_id]
        )


HLRState = namedtuple("HLRState", ["features", "latest_review_ts"])


class HLRLearningModel(MetaLearningModel):
    def step(self, state: HLRState, learning_item, outcome, timestamp, *args, **kwargs):
        state.features[learning_item][0] += 1
        state.features[learning_item][1 if outcome == 1 else 2] += 1
        state.latest_review_ts[learning_item] = timestamp


class HLRLearner(MetaLearner):
    def __init__(self, features: list, latest_review_ts: list, memory_strengths: list):
        super(HLRLearner, self).__init__()
        self._state = HLRState(features, latest_review_ts)
        self._learning_model = HLRLearningModel()
        self._memory_strengths = memory_strengths

    @property
    def state(self):
        return HLRState([f[:3] for f in self._state.features], self._state.latest_review_ts)

    def learn(self, learning_item, timestamp, *args, **kwargs):
        self._learning_model.step(self._state, learning_item, self.test(learning_item, timestamp), timestamp)

    def test(self, exercise_id, timestamp, *args, **kwargs):
        return hlr(
            timestamp - self._state.latest_review_ts[exercise_id],
            self._memory_strengths,
            self._state.features[exercise_id],
        )


GPLState = namedtuple("GPLState", ["n_correct", "n_attempts", "latest_review_ts", "window_index"])


class GPLLearningModel(MetaLearningModel):
    def __init__(self, window_size, n_items):
        self._window_size = window_size
        self.n_items = n_items

    def step(self, state: GPLState, learning_item, correct, timestamp, *args, **kwargs):
        state.window_index += 1
        if state.window_index // self._window_size >= len(state.n_correct):
            state.n_correct.append([0] * self.n_items)
        if correct:
            state.n_correct[-1][learning_item] += 1
        state.n_attempts[-1][learning_item] += 1
        state.latest_review_ts[learning_item] = timestamp


class GPLLearner(MetaLearner):
    def __init__(self, latest_review_ts: list, abilities: list, additional_abilities: list,
                 decay_rate: float, window_correct_coefficients: list = None, window_attempt_coefficients: list = None,
                 window_size=40):
        super(GPLLearner, self).__init__()
        self._state = GPLState(
            [[0] * len(latest_review_ts)],
            [[0] * len(latest_review_ts)],
            latest_review_ts,
            0
        )
        self._learning_model = GPLLearningModel(window_size, len(latest_review_ts))
        self._abilities = abilities
        self._additional_abilities = additional_abilities
        self._decay_rate = decay_rate
        self._window_correct_coefficients = window_correct_coefficients
        self._window_attempt_coefficients = window_attempt_coefficients
        self._window_size = window_size

    def test(self, exercise, item_difficulty, item_additional_difficulty, timestamp, *args, **kwargs):
        return gpl(
            self._abilities[exercise],
            self._additional_abilities[exercise],
            item_difficulty,
            item_additional_difficulty,
            timestamp - self._state.latest_review_ts[exercise],
            self._decay_rate,
            self._state.n_correct[self._state.window_index // self._window_size][exercise],
            self._state.n_attempts[self._state.window_index // self._window_size][exercise],
            self._window_correct_coefficients,
            self._window_attempt_coefficients
        )


    def learn(self, learning_item, timestamp, *args, **kwargs):
        self.test(learning_item, timestamp)