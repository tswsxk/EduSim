# coding: utf-8
# 2019/11/28 @ tongshiwei


from EduSim.Envs.Reward import get_reward


def test_reward():
    reward = get_reward()

    assert isinstance(reward.step_reward(), (int, float))

    assert reward.episode_reward(0, 1, 1) == 1
    assert reward.episode_reward(0, 100, 100) == 1
    assert reward.episode_reward(90, 100, 100) == 0.1
    assert reward.episode_reward(6, 10, 10) == 0.4
    assert reward.episode_reward(0.6, 1, 1) == 0.4
