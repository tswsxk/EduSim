# coding: utf-8
# 2019/11/27 @ tongshiwei

from EduSim.Envs.base import return_key_tester


def test_tester(tester, learner):
    tester.begin_episode(learner)
    assert isinstance(tester.test(5), int)
    assert isinstance(tester.exam(), int)
    result = tester.summary_episode([])

    for key in return_key_tester:
        assert key in result
