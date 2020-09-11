# coding: utf-8
# 2019/11/27 @ tongshiwei

import random
from EduSim import TMSEnv
from EduSim.Envs.TMS.Env import NAME, MODE
from longling import path_append
import pytest
import gym


@pytest.mark.parametrize("name", NAME)
@pytest.mark.parametrize("mode", list(MODE.keys()))
def test_api(name, mode):
    env = TMSEnv(name, mode=mode)

    assert set(env.description.keys()) == {"ks", "action_space"}

    learner_profile = env.begin_episode()

    assert isinstance(learner_profile, dict)

    action = random.choice(env.description["action_space"])
    env.step(action)
    env.end_episode()


@pytest.mark.parametrize("name", NAME)
@pytest.mark.parametrize("mode", list(MODE.keys()))
def test_env(name, mode, tmp_path):
    from EduSim.Envs.TMS import tms_train_eval, TMSAgent

    env = gym.make("TMS-v0", name=name, mode=mode)

    tms_train_eval(
        TMSAgent(env.action_num),
        env,
        20, 10,
        level="step",
        board_dir=path_append(tmp_path, "tms_logs", to_str=True)
    )
