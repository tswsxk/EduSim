# coding: utf-8
# create by tongshiwei on 2019/6/25

import json
import contextlib

import gym

from longling import config_logging
from longling.lib.structure import AttrDict

try:
    from .Reward import get_reward
    from .Learner import Learner
    from .Tester import Tester
except (SystemError, ModuleNotFoundError):  # pragma: no cover
    from Reward import get_reward
    from Learner import Learner
    from Tester import Tester


class Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, logger=None, log_rf=None, *args, **kwargs):
        # components
        self.tester = None
        self.reward = None
        self._logger = logger if logger is not None else config_logging(logger="Env", console_log_level="info")
        if log_rf is None:
            self._log_rf = config_logging(
                logger="Env_RF", console_log_level="info", log_format=json.dumps("%(message)s")
            )
        elif isinstance(log_rf, str):
            self._log_rf = config_logging(
                filename=log_rf, logger="Env_RF", file_format=json.dumps("%(message)s")
            )
        else:
            self._log_rf = log_rf

        # episode relevant
        self._learner = None
        self._path = None

    def render(self, mode='human'):
        print(self._learner)

    def reset(self):
        self._reset_episode()
        return self.begin_episode()

    def learn(self, learning_item):
        self._learner.learn(learning_item)

    def run_test(self, learner, exercise, *args, **kwargs) -> ...:
        return self.tester.test(exercise, learner=learner, *args, **kwargs)

    def test(self, exercise) -> dict:
        """
        In test procedure, a learner will be given an exercise, and the answer will be measured by tester.
        After testing, the result will be exposed to the learner.
        """
        raise NotImplementedError

    def exam(self, exercises=None):
        """
        In exam procedure, a learner will be given one or more exercises, and the answer will be measured by tester.
        """
        raise NotImplementedError

    def step(self, learning_item, exercise=None, *args, **kwargs) -> [object, float, bool, dict]:
        """simulate a learner learn and test on a certain learning item and exercise"""
        self._path.append(learning_item)
        self.learn(learning_item)
        exercise = learning_item if exercise is None else exercise
        performance = self.test(exercise)
        return AttrDict({
            "performance": performance,
            "reward": self.step_reward()
        }), 0., False, {}

    def n_step(self, learning_items, exercises=None, *args, **kwargs):
        """n step learning and testing simulation"""
        exercises = learning_items if exercises is None else exercises
        return [self.step(learning_item, exercise, *args, **kwargs) for learning_item, exercise in
                zip(learning_items, exercises)]

    @contextlib.contextmanager
    def episode(self, *args, **kwargs):
        learner = self.begin_episode(*args, **kwargs)
        yield learner
        self._reset_episode()

    def begin_episode(self, *args, **kwargs) -> Learner:
        """invoked at the most beginning of an episode (initialize state)"""
        raise NotImplementedError

    def _reset_episode(self, *args, **kwargs):
        self._learner = None
        self._path = None

    def summary_episode(self, *args, **kwargs) -> dict:
        _learner = self._learner
        _path = self._path

        summary = {
            "learner_id": None,
            "path": _path,
            "reward": None,
            "evaluation": None
        }

        self._logger.info(summary)
        self._log_rf.info(summary)
        return summary

    def end_episode(self, *args, **kwargs) -> ...:
        """Three required key in returned value: learner, path and reward, evaluation"""
        summary = self.summary_episode()
        self._reset_episode()

        return summary

    def is_valid_sample(self, learner):
        return sum([self.run_test(learner, t) for t in learner.target]) < len(learner.target)

    def remove_invalid_sample(self, idx):
        raise NotImplementedError

    def step_reward(self):
        return None

    def episode_reward(self, initial_score=None, final_score=None, full_score=None):
        if self._path:
            return self.reward.episode_reward(initial_score, final_score, full_score)
        else:
            return 0

    def eval(self, agent, max_steps, max_episode_num=None, n_step=False):
        train_eval(agent, self, max_steps, max_episode_num, n_step, train=False)

    def train(self, agent, max_steps, max_episode_num=None, n_step=False):
        train_eval(agent, self, max_steps, max_episode_num, n_step, train=True)


def train_eval(agent, env, max_steps, max_episode_num=None, n_step=False, train=False):
    episode = 0

    while True:
        if max_episode_num is not None and episode > max_episode_num:
            break

        try:
            agent.begin_episode(env.begin_episode())
            episode += 1
        except ValueError:  # pragma: no cover
            break

        # recommend and learn
        if n_step is True:
            # generate a learning path
            learning_path = agent.n_step(max_steps)
            env.n_step(learning_path)
        else:
            # generate a learning path step by step
            for _ in range(max_steps):
                try:
                    learning_item = agent.step()
                except ValueError:  # pragma: no cover
                    break
                interaction, _, _, _ = env.step(learning_item)
                agent.observe(**interaction["performance"])

        # test the learner to see the learning effectiveness
        agent.episode_reward(env.end_episode()["reward"])
        agent.end_episode()

        if train is True:
            agent.tune()
