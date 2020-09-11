# coding: utf-8
# create by tongshiwei on 2019/6/25

from gym.envs.registration import register
from .Envs import *
from .SimOS import train_eval, MetaAgent

register(
    id='KSS-v1',
    entry_point='EduSim.Envs:KSSEnv',
)

register(
    id='TMS-v0',
    entry_point='EduSim.Envs:TMSEnv',
)
