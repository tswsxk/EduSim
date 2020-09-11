# coding: utf-8
# 2020/5/13 @ tongshiwei

import logging
import numpy as np
import random
from EduSim.Envs.meta import Env
from .Learner import EFCLearner, HLRLearner
from EduSim.SimOS import train_eval
from EduSim.utils import SummaryWriter, board_episode_callback, reward_summary_callback


def mbs_train_eval(agent, env: Env, max_steps: int = None, max_episode_num: int = None, n_step=False,
                   train=False,
                   logger=logging, level="episode", board_dir=None):
    sw = None
    if board_dir is not None:
        sw = SummaryWriter(board_dir)

        def episode_callback(episode, reward, *args):
            return board_episode_callback(episode, reward, sw)

    else:
        episode_callback = None

    train_eval(
        agent, env,
        max_steps, max_episode_num, n_step, train,
        logger, level,
        episode_callback=episode_callback,
        summary_callback=reward_summary_callback,
    )

    if board_dir:
        sw.close()


def sample(sample_type="const", a=None, b=None):
    if sample_type == "const":
        return a
    elif sample_type == "randint":
        return random.randint(a, b)
    elif sample_type == "random":
        return random.random() * (b - a) + a
    else:
        raise TypeError("unknown sample type: %s" % sample_type)


class Reward(object):
    def __init__(self, reward_func="likelihood"):
        self.reward_func = reward_func

    @staticmethod
    def likelihood(probabilities):
        return np.asarray(probabilities).mean()

    @staticmethod
    def log_likelihood(probabilities, eps=1e-9):
        return np.log(eps + np.asarray(probabilities)).mean()

    def __call__(self, probabilities, *args, **kwargs):
        if self.reward_func == "likelihood":
            return self.likelihood(probabilities)
        elif self.reward_func == "log_likelihood":
            return self.log_likelihood(probabilities)
        else:
            raise TypeError("unknown reward function: %s" % self.reward_func)


class MetaEnv(Env):
    def __init__(self, n_items=30, reward_func="likelihood", threshold=0.5, *args, **kwargs):
        self.n_items = n_items
        self.item_difficulties = self.sample_item_difficulties(self.n_items)
        self.timestamp = 0
        self._learner = None
        self._reward = Reward(reward_func=reward_func)
        self._threshold = threshold

    @staticmethod
    def sample_memory_strengths(n_items, sample_type="const", a=1, b=None):
        return [sample(sample_type, a, b) for _ in range(n_items)]

    @staticmethod
    def sample_delay(sample_type="const", a=5, b=None):
        return sample(sample_type, a, b)

    @staticmethod
    def sample_item_difficulties(n_items):
        raise NotImplementedError

    @staticmethod
    def sample_init_review_time(n_items, sample_type="const"):
        if sample_type == "const":
            return [-np.inf] * n_items
        elif sample_type == "normal":
            return np.exp(np.random.normal(0, 1, n_items)).tolist()
        else:
            raise TypeError("unknown sample type: %s" % sample_type)

    # @staticmethod
    # def sample_item_difficulties(n_items, item_difficulty_mean=1, item_difficulty_std=1):
    #     return np.random.normal(item_difficulty_mean, item_difficulty_std, n_items).tolist()

    def test(self, item, timestamp):
        raise NotImplementedError

    def step(self, learning_item_id, *args, **kwargs):
        timestamp = self.timestamp

        probabilities = [
            self.test(item, timestamp) for item in range(self.n_items)
        ]

        self._learner.learn(learning_item_id, timestamp)

        observation = [
            learning_item_id, 1 if random.random() < probabilities[learning_item_id] else 0, timestamp
        ]

        reward = self._reward(probabilities)
        done = all([p > self._threshold for p in probabilities])
        info = {}

        self.timestamp += self.sample_delay()
        return observation, reward, done, info

    def n_step(self, learning_path, *args, **kwargs):
        for learning_item_id in learning_path:
            self.step(learning_item_id)

    def end_episode(self, *args, **kwargs):
        probabilities = [
            self.test(item, self.timestamp) for item in range(self.n_items)
        ]
        observation = [
            [
                learning_item_id, 1 if random.random() < probabilities[learning_item_id] else 0, self.timestamp
            ] for learning_item_id in range(self.n_items)
        ]
        reward = self._reward(probabilities)
        done = all([p > self._threshold for p in probabilities])
        info = {}

        return observation, reward, done, info

    def reset(self):
        self._learner = None
        self.timestamp = 0

    def render(self, mode='human'):
        if mode == "log":
            return self._learner.state


class EFCEnv(MetaEnv):
    @staticmethod
    def sample_item_difficulties(n_items):
        return np.exp(np.random.normal(np.log(0.077), 1, n_items)).tolist()

    def begin_episode(self, *args, **kwargs):
        memory_strengths = self.sample_memory_strengths(self.n_items)
        self._learner = EFCLearner(
            memory_strengths,
            # latest_review_ts=[-np.inf] * self.n_items，
            latest_review_ts=self.sample_init_review_time(self.n_items)
        )

    def test(self, item, timestamp):
        return self._learner.test(item, self.item_difficulties[item], timestamp)


class HLREnv(MetaEnv):
    def __init__(self, n_items=30, reward_func="likelihood", threshold=0.5, *args, **kwargs):
        super(HLREnv, self).__init__(n_items, reward_func, threshold, *args, **kwargs)
        self._memory_strengths = np.concatenate(
            (np.asarray([1, 1, 0]), np.random.normal(0, 1, n_items))
        ).tolist()

    @staticmethod
    def sample_item_difficulties(n_items):
        return np.exp(np.random.normal(np.log(0.077), 1, n_items)).tolist()

    def begin_episode(self, *args, **kwargs):
        # n_attempts, n_correct, n_incorrect
        features = np.zeros((self.n_items, 3))
        features = np.concatenate((features, np.eye(self.n_items)), axis=1).tolist()
        self._learner = HLRLearner(features, [-np.inf] * self.n_items, self._memory_strengths)

    def test(self, item, timestamp):
        return self._learner.test(item, timestamp)
