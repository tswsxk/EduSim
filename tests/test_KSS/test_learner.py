# coding: utf-8
# 2019/11/27 @ tongshiwei
# potential bugs


def test_learner(tester, learner):
    tester.begin_episode(learner)
    initial_ability = tester.exam(binary=False)
    assert isinstance(initial_ability, (int, float))
    assert initial_ability == tester.exam(binary=False)
    for i in learner.target:
        learner.learn(i)
    assert initial_ability != tester.exam(binary=False)
