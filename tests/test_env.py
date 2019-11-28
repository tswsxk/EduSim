# coding: utf-8
# 2019/11/28 @ tongshiwei

from longling import config_logging, path_append

from EduSim.Envs import Env
from EduSim.Envs.base import return_key_env_summary_episode


def test_env(tmp_path):
    log_rf = path_append(tmp_path, "kss_rf.json", to_str=True)

    Env(log_rf=log_rf)
    Env()
    _env = Env(log_rf=config_logging(logger="test_env"))

    _env.render()
    _env.reset()
    assert set(_env.summary_episode()) == set(return_key_env_summary_episode)

    assert True
