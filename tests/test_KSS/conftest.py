# coding: utf-8
# 2019/11/27 @ tongshiwei

import pytest
import gym


@pytest.fixture(scope="module")
def env():
    return gym.make('KSS-v1', learner_num=20)
