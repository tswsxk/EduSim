# coding: utf-8
# 2019/11/27 @ tongshiwei

import random
import pytest
from EduSim.Envs.KSS import KSSEnv
from EduSim.Envs.KSS.Env import MODE
from longling import path_append


@pytest.mark.parametrize("mode", list(MODE.keys()))
def test_api(mode):
    env = KSSEnv(learner_num=20, mode=mode)

    assert set(env.description.keys()) == {"ks", "action_space"}

    learner_profile = env.begin_episode()

    assert isinstance(learner_profile, list)

    action = random.choice(env.description["action_space"])
    env.step(action)
    env.end_episode()


@pytest.mark.parametrize("n_step", [True, False])
def test_env(env, tmp_path, n_step):
    from EduSim.Envs.KSS import kss_train_eval, KSSAgent

    kss_train_eval(
        KSSAgent(env.action_num),
        env,
        20, 10,
        level="step",
        n_step=n_step,
        board_dir=path_append(tmp_path, "kss_logs", to_str=True)
    )
