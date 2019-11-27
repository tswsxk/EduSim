# coding: utf-8
# 2019/11/25 @ tongshiwei


"""generate reward based on the performance of learners/students"""

import numpy as np


def get_reward():
    return GreedyExpReward()


class Reward(object):
    def __call__(self, initial_score, final_score, full_score, path, terminal_tag=False, rs=None):
        path_len = len(path)

        delta_base = full_score - initial_score
        delta = final_score - initial_score

        bias = 0

        def global_reward_decay(_global_reward):
            _global_reward = 0
            return _global_reward

        global_reward = delta
        normalize_factor = delta_base

        start = path_len - 1
        reward_values = [0] * path_len
        if terminal_tag:
            reward = global_reward
            reward_values[-1] = reward / normalize_factor
            start -= 1

        # bias
        defualt_r = bias

        rs = [defualt_r] * path_len if rs is None else rs

        assert len(rs) == path_len

        for i in range(start, -1, -1):
            # weight
            reward = global_reward + rs[i]
            global_reward = global_reward_decay(global_reward)
            reward_values[i] = reward / normalize_factor

        return reward_values

    @staticmethod
    def mature_reward(reward_values):
        reward_values = np.array(reward_values)
        eps = np.finfo(reward_values.dtype).eps.item()
        reward_values = (reward_values - reward_values.mean()) / (reward_values.std() + eps)
        return reward_values


class GreedyExpReward(Reward):
    def __call__(self, initial_score, final_score, full_score, path, terminal_tag=False, rs=None):
        path_len = len(path)

        delta_base = full_score - initial_score
        delta = final_score - initial_score

        bias = 0

        def global_reward_decay(_global_reward):
            _global_reward *= 0.99
            return _global_reward

        global_reward = delta
        normalize_factor = full_score

        start = path_len - 1
        reward_values = [0] * path_len
        if terminal_tag:
            reward = global_reward
            reward_values[-1] = reward / normalize_factor
            start -= 1

        # bias
        defualt_r = bias

        rs = [defualt_r] * path_len if rs is None else rs

        assert len(rs) == path_len

        for i in range(start, -1, -1):
            # weight
            reward = global_reward + rs[i]
            global_reward = global_reward_decay(global_reward)
            reward_values[i] = reward / normalize_factor

        return reward_values
