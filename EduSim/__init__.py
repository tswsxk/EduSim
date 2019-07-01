# coding: utf-8
# create by tongshiwei on 2019/6/25

from gym.envs.registration import register
from .Agent import *

register(
    id='KSS-v0',
    entry_point='EduSim.envs:KSS',
)
